# Recommendation Engine V2 (Production-Grade)

Standalone, e-commerce-agnostic **Recommendation-as-a-Service** built with FastAPI, SQLite, Scikit-Learn, and Docker.

---

## What's New in V2

- **Dense Semantic Retrieval (LSA)** — TF-IDF + TruncatedSVD(128) dense projection pipeline capturing semantic similarity (e.g. "wireless earbuds" matching "Bluetooth headphones").
- **Self-Training ML Ranker** — `HistGradientBoostingClassifier` model trained directly on user interactions (purchases, carts, views) using 14 engineered features. Silently and gracefully falls back to heuristic blending if no model is trained yet.
- **MMR Diversity Re-Ranking** — Maximal Marginal Relevance pass avoiding repetitive results by penalizing category and brand duplication, controllable via `diversity_factor`.
- **Full Observability** — In-process Prometheus `/metrics` exporter (counters, latency histograms), Kubernetes-compatible `/ready` probe, request timing middleware, and structured JSON logs.
- **Containerized & Production Ready** — Zero new mandatory dependencies, multi-stage `Dockerfile`, and `docker-compose.yml` pre-configured with Prometheus.

---

## Core Features

- **Universal catalog ingestion** — Ingest any product schema via single or bulk REST endpoints
- **Event ingestion** — Track user actions (`product_view`, `product_click`, `add_to_cart`, `purchase`, `search`, etc.)
- **Anonymous & authenticated users** — Attribution via `user_id` or `anonymous_id`
- **Session-aware intent signals** — Dynamic boosts based on recent in-session searches and interactions
- **Dual-channel retrieval** — Sparse TF-IDF cosine similarity + Dense SVD semantic retrieval
- **Collaborative filtering** — Co-occurrence matrix across positive user interactions
- **Popular & trending ranking** — Exponential time-decayed engagement scoring (`math.exp(-age / 14)`)
- **Freshness boosting** — Exponential decay boost (`math.exp(-age / 7)`) for newly added catalog products
- **Optional geo-aware ranking** — Distance-decay scoring using user coordinates and product location metadata
- **Faceted filtering** — Filter recommendations by category, brand, price ranges (`min_price`, `max_price`), availability, and attributes
- **Engine caching & lifecycle** — In-memory caching with TTL and automatic invalidation on catalog updates
- **Zero-config SQLite with WAL** — Thread-safe concurrent reads with Write-Ahead Logging

---

## System Architecture

```mermaid
flowchart TB
    subgraph ClientLayer["Clients & Upstream Services"]
        WebClient["Web / Mobile Apps"]
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

    subgraph CoreLayer["Core Orchestrator (Recommender)"]
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

    WebClient -->|"/v1/recommendations, /v1/events"| TimingMW
    AdminService -->|"/v1/catalog/products"| TimingMW
    PromService -->|"/metrics"| MetricsEp

    TimingMW --> Router
    TimingMW -.->|Record Latency| MetricsStore
    MetricsEp --> MetricsStore
    Router --> ReadyProbe
    ReadyProbe -.->|Check Connection| Store

    Router -->|Upsert Products| Store
    Router -->|Record Events| Store
    Router -->|POST /v1/models/train| Trainer

    Store --- ProductsTable
    Store --- EventsTable

    Router -->|Recommend Request| Funnel
    Funnel --> EngineCache
    EngineCache --> EnginesLayer
    EnginesLayer --> Store

    EnginesLayer -->|Candidates + Heuristic Scores| FacetedFilter
    FacetedFilter -->|Filtered Candidates| FeatExtract
    FeatExtract --> Ranker
    Ranker -.->|Load Weights| ModelStore
    Ranker -->|Ranked by Engagement Probability| MMR
    MMR -->|Diversified Top-K| WebClient

    Trainer -->|Extract Historical Interactions| Store
    Trainer -->|Save Model| ModelStore
    Trainer -.->|Reload Weights| Ranker
```

---

## 3-Stage Recommendation Funnel

```mermaid
flowchart LR
    subgraph S1["Stage 1: Multi-Channel Retrieval"]
        direction TB
        R1["Dense SVD Semantic"]
        R2["Sparse TF-IDF"]
        R3["Collaborative / Affinity"]
        R4["Popularity & Freshness"]
        R1 & R2 & R3 & R4 --> Union["Candidate Union ~200 items"]
        Union --> Filter["Faceted Filtering"]
    end

    subgraph S2["Stage 2: Precision Ranking"]
        direction TB
        Feats["14 Feature Vector Extraction"]
        ML["HistGradientBoosting Inference"]
        Feats --> ML
        ML --> PScore["Score: Engagement Probability"]
    end

    subgraph S3["Stage 3: MMR Diversity"]
        direction TB
        MMR["Maximal Marginal Relevance"]
        Penalty["Penalize Category & Brand Overlap"]
        MMR --> Penalty
        Penalty --> TopK["Top-K Diverse Results"]
    end

    S1 --> S2
    S2 --> S3
```

---

## Project Structure

```
recommender_v1/
├── app/
│   ├── config.py           # Configurable settings (DB path, candidate limits, engine weights, TTL)
│   ├── main.py             # FastAPI entrypoint, lifespan, error handlers, observability endpoints
│   ├── core/
│   │   ├── recommender.py  # Orchestrator: candidate generation, ML ranker scoring, MMR diversity
│   │   └── metrics.py      # Thread-safe in-process Prometheus metrics store & text serializer
│   ├── engines/
│   │   ├── content.py      # Sparse TF-IDF content similarity & query search
│   │   ├── dense.py        # Dense LSA (TF-IDF + TruncatedSVD) semantic retrieval
│   │   ├── diversity.py    # Maximal Marginal Relevance (MMR) re-ranking
│   │   ├── behavior.py     # User affinity & collaborative co-occurrence
│   │   ├── popularity.py   # Time-decayed popularity ranking
│   │   ├── freshness.py    # Time-decayed catalog freshness ranking
│   │   └── geo.py          # Haversine distance-based geo scoring
│   ├── ml/
│   │   ├── features.py     # 14-feature extraction pipeline (user, product, cross, context)
│   │   ├── ranker.py       # MLRanker wrapper around HistGradientBoostingClassifier
│   │   └── trainer.py      # Negative sampling, 80/20 train/val split, AUC & NDCG evaluation
│   ├── models/
│   │   └── schemas.py      # Pydantic schemas (Product, Event, Request, Response)
│   └── storage/
│       └── store.py        # SQLite storage with WAL mode, indexing, thread safety
├── tests/
│   ├── test_basic.py       # Basic health check test
│   ├── test_recommender.py # V1 regression test suite (9 tests)
│   └── test_v2.py          # V2 production test suite (8 tests)
├── Dockerfile              # Production container build
├── docker-compose.yml      # Service orchestration (API + Prometheus)
├── prometheus.yml          # Prometheus scrape configuration
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Production dependencies
└── requirements-dev.txt    # Development and testing dependencies
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service liveness probe |
| `GET` | `/ready` | Service readiness probe (validates database connectivity) |
| `GET` | `/metrics` | Prometheus metrics exposition (`recsys_requests_total`, `recsys_latency_ms`, etc.) |
| `POST` | `/v1/catalog/products` | Upsert a single product |
| `POST` | `/v1/catalog/products/bulk` | Bulk upsert a list of products |
| `POST` | `/v1/events` | Ingest user/session interaction events |
| `POST` | `/v1/recommendations` | Get ranked, diversified, filtered recommendations |
| `POST` | `/v1/models/train` | Trigger background ML ranker training from event log |
| `GET` | `/v1/models/info` | Inspect currently loaded ML model metadata (AUC, NDCG, sample count) |

---

## Quickstart

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Run tests:**
   ```bash
   python -m pytest -v
   ```

3. **Start the API server:**
   ```bash
   uvicorn app.main:app --reload
   ```

Swagger documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Running with Docker Compose

```bash
docker compose up --build
```
- API server: [http://localhost:8000](http://localhost:8000)
- Prometheus dashboard: [http://localhost:9090](http://localhost:9090)
