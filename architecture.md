# Recommendation Engine V2 — System Architecture & Data Flow

This document details the multi-stage architecture, data flow, and training pipeline of Recommendation Engine V2.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph ClientLayer["Clients & Upstream Services"]
        WebClient["Web / Mobile App"]
        AdminService["Catalog / ERP System"]
        PromService["Prometheus Scraper"]
    end

    subgraph APILayer["FastAPI Gateway & Observability"]
        TimingMW["RequestTimingMiddleware"]
        Router["APIRouter (/v1)"]
        ReadyProbe["/ready & /health"]
        MetricsEp["/metrics"]
        MetricsStore["MetricsStore (In-Memory Counters & Histograms)"]
    end

    subgraph CoreLayer["Orchestration Engine (Recommender)"]
        EngineCache["Engine Cache (TTL + Version Invalidation)"]
        Funnel["3-Stage Recommendation Funnel"]
        FacetedFilter["Faceted Filter (Category, Brand, Price, Stock)"]
    end

    subgraph EnginesLayer["Retrieval Engines (Stage 1)"]
        DenseEng["DenseEngine (TF-IDF + TruncatedSVD)"]
        ContentEng["ContentEngine (Sparse TF-IDF)"]
        BehaviorEng["BehaviorEngine (User Affinity & Co-occurrence)"]
        PopEng["PopularityEngine (Time-Decayed Engagement)"]
        FreshEng["FreshnessEngine (Exponential Age Decay)"]
        GeoEng["GeoEngine (Haversine Distance Decay)"]
    end

    subgraph MLLayer["Machine Learning Layer (Stage 2 & Background)"]
        Ranker["MLRanker (HistGradientBoostingClassifier)"]
        FeatExtract["FeatureExtractor (14 Numerical Features)"]
        Trainer["Async Trainer (Negative Mining, AUC & NDCG)"]
        ModelStore[("models/ranker.joblib")]
    end

    subgraph DiversityLayer["Post-Processing (Stage 3)"]
        MMR["MMR Diversity Engine (Category & Brand Penalization)"]
    end

    subgraph StorageLayer["Data Layer (SQLite WAL Mode)"]
        Store["Store with Read/Write Locks"]
        ProductsTable[("products (id, JSON data)")]
        EventsTable[("events (user, session, product, type, timestamp)")]
    end

    %% Client flows
    WebClient -->|"/v1/recommendations, /v1/events"| TimingMW
    AdminService -->|"/v1/catalog/products"| TimingMW
    PromService -->|"/metrics"| MetricsEp

    %% Gateway
    TimingMW --> Router
    TimingMW -.->|Record Latency| MetricsStore
    MetricsEp --> MetricsStore
    Router --> ReadyProbe
    ReadyProbe -.->|Check Connection| Store

    %% Ingestion
    Router -->|Upsert Products| Store
    Router -->|Record Events| Store
    Router -->|POST /v1/models/train| Trainer

    %% Storage
    Store --- ProductsTable
    Store --- EventsTable

    %% Core Recommender
    Router -->|Recommend Request| Funnel
    Funnel --> EngineCache
    EngineCache --> EnginesLayer
    EnginesLayer --> Store

    %% 3-Stage Pipeline
    EnginesLayer -->|Candidates + Blended Heuristics| FacetedFilter
    FacetedFilter -->|Filtered Candidates| FeatExtract
    FeatExtract --> Ranker
    Ranker -.->|Load Weights| ModelStore
    Ranker -->|Ranked by Engagement Probability| MMR
    MMR -->|Diversified Top-K| WebClient

    %% Training flow
    Trainer -->|Extract Historical Interactions| Store
    Trainer -->|Save Model| ModelStore
    Trainer -.->|Reload Weights| Ranker
```

---

## 2. 3-Stage Recommendation Funnel (Online Flow)

Every `POST /v1/recommendations` request passes through a 3-stage funnel:

```mermaid
flowchart TD
    Req["Incoming Request: user_id, context, filters, diversity_factor"] --> S1

    subgraph S1["Stage 1: Multi-Channel Retrieval & Candidate Generation"]
        direction TB
        C1["Dense Semantic Retrieval (TruncatedSVD)"]
        C2["Sparse Text Matching (TF-IDF)"]
        C3["Personalized Affinity & Collaborative Co-occurrence"]
        C4["Time-Decayed Popularity & Trending"]
        C5["Catalog Freshness Boost"]
        C6["Geo-Distance Proximity"]

        C1 & C2 & C3 & C4 & C5 & C6 --> Union["Candidate Union ~200 items"]
        Union --> Filter["Apply Faceted Filters: Brand, Category, Price Range, Availability"]
    end

    Filter --> FallbackCheck{"MLRanker Available?"}

    subgraph S2["Stage 2: Precision Ranking"]
        direction TB
        ExtractFeats["Extract 14 Features per Candidate: User Stats, Product Stats, Cross Signals, Context"]
        PredictProba["HistGradientBoostingClassifier Vectorized Inference"]
        ExtractFeats --> PredictProba
        PredictProba --> ReScore["Replace Score with Engagement Probability"]
    end

    FallbackCheck -- "Yes" --> S2
    FallbackCheck -- "No (Fallback)" --> Heuristic["Use Normalized Heuristic Blended Score"]

    ReScore --> S3
    Heuristic --> S3

    subgraph S3["Stage 3: Maximal Marginal Relevance (MMR) Diversity"]
        direction TB
        MMRCalc["Calculate MMR balancing relevance and category/brand diversity"]
        SortSelect["Greedy Selection of Top-K Items"]
        MMRCalc --> SortSelect
    end

    SortSelect --> Resp["Final Response (UUID, Diversified Recommendations, Reasons, Sources)"]
```

---

## 3. Asynchronous Model Training Pipeline (Offline / Background)

```mermaid
sequenceDiagram
    autonumber
    participant Admin as Client / Cron
    participant API as FastAPI Router
    participant Trainer as app.ml.trainer
    participant Store as SQLite Store
    participant Disk as models/ranker.joblib
    participant Engine as Recommender Core

    Admin->>API: POST /v1/models/train
    API-->>Admin: 200 {"status": "training_started"}
    Note over API,Trainer: Non-blocking execution via run_in_executor

    Trainer->>Store: Query all products & events (up to 100k)
    Store-->>Trainer: Products catalog + Historical events

    Note over Trainer: 1. Group events by user identity<br/>2. Mine positive signals (purchase: 3x, cart: 2x, click: 1x)<br/>3. Verify minimum 50 positive samples<br/>4. Random negative sampling of unseen products

    Trainer->>Trainer: Extract 14 features per sample
    Trainer->>Trainer: 80/20 Train/Validation Split (Stratified)
    Trainer->>Trainer: Fit HistGradientBoostingClassifier(max_iter=200, class_weight='balanced')
    Trainer->>Trainer: Evaluate ROC-AUC & NDCG@10

    Trainer->>Disk: Persist model + evaluation metadata (joblib.dump)
    Trainer->>Engine: reload_ranker()
    Trainer->>Trainer: Increment recsys_train_runs_total counter
```

---

## 4. 14 Engineered ML Features

| Index | Feature Name | Description | Group |
|---|---|---|---|
| 0 | `user_click_count` | Total views & clicks recorded for this user | User History |
| 1 | `user_cart_count` | Total add-to-cart actions recorded | User History |
| 2 | `user_purchase_count` | Total purchases completed by user | User History |
| 3 | `user_avg_price_viewed`| Average price of products interacted with | User History |
| 4 | `product_popularity_score`| Time-decayed popularity score | Product Stats |
| 5 | `product_freshness_score` | Exponential decay freshness score (`exp(-age/7)`) | Product Stats |
| 6 | `product_price` | Product catalog price | Product Stats |
| 7 | `product_availability` | 1.0 if in stock, else 0.0 | Product Stats |
| 8 | `category_affinity` | 1.0 if category matches user history | Cross Signals |
| 9 | `brand_affinity` | 1.0 if brand matches user history | Cross Signals |
| 10 | `dense_cosine_sim` | Semantic LSA cosine similarity with context | Cross Signals |
| 11 | `price_ratio` | Ratio of user's average viewed price to product price | Cross Signals |
| 12 | `is_product_context` | 1.0 if context is `product`, else 0.0 | Context |
| 13 | `is_search_context` | 1.0 if context is `search` or query provided | Context |
