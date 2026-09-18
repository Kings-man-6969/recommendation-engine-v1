import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import Product, Event, RecommendationRequest, Context
from app.storage.store import Store
from app.engines.dense import DenseEngine
from app.engines.diversity import mmr_rerank
from app.ml.features import extract_user_profile, extract_features, FEATURE_NAMES
from app.ml.ranker import MLRanker
from app.ml.trainer import train_from_store
from app.core.recommender import Recommender
from app.core.metrics import MetricsStore

client = TestClient(app)

def test_dense_engine():
    products = [
        Product(id="p1", title="Wireless Bluetooth Earbuds", description="Noise cancelling in-ear audio headphones", category=["Audio"]),
        Product(id="p2", title="Over-ear Wireless Headphones", description="Studio monitor sound bluetooth headset", category=["Audio"]),
        Product(id="p3", title="Mechanical Gaming Keyboard", description="RGB backlit clicky switches", category=["Accessories"]),
    ]
    dense = DenseEngine(products, n_components=2)
    assert dense.matrix is not None

    # Query for headphones should rank audio products high
    q_res = dense.query("Bluetooth headphones")
    assert len(q_res) > 0
    top_pids = [pid for pid, _ in q_res]
    assert "p1" in top_pids or "p2" in top_pids

    # Similar items for p1 should return p2
    sims = dense.similar("p1")
    assert len(sims) > 0
    assert sims[0][0] == "p2"

def test_dense_engine_edge_cases():
    # Empty products
    d_empty = DenseEngine([])
    assert d_empty.matrix is None
    assert d_empty.similar("p1") == []
    assert d_empty.query("test") == []

    # Single product
    d_single = DenseEngine([Product(id="p1", title="Single Item")])
    assert d_single.similar("p1") == []

def test_feature_extraction():
    byid = {
        "p1": Product(id="p1", title="Laptop", category=["Computers"], brand="BrandX", price=1000.0),
        "p2": Product(id="p2", title="Phone", category=["Mobile"], brand="BrandY", price=500.0),
    }
    events = [
        {"event": "product_view", "product_id": "p1"},
        {"event": "product_click", "product_id": "p1"},
        {"event": "add_to_cart", "product_id": "p1"},
        {"event": "purchase", "product_id": "p2"},
    ]
    profile = extract_user_profile(events, byid)
    assert profile.click_count == 2
    assert profile.cart_count == 1
    assert profile.purchase_count == 1
    # 3 interactions with p1 ($1000) and 1 with p2 ($500) -> 3500 / 4 = 875.0
    assert profile.avg_price == 875.0
    assert "computers" in profile.categories
    assert "brandx" in profile.brands

    ctx = Context(type="product", product_id="p1")
    features = extract_features(
        profile,
        byid["p1"],
        context=ctx,
        pop_score=0.8,
        fresh_score=0.9,
        dense_sim=0.75
    )
    assert len(features) == len(FEATURE_NAMES)
    assert len(features) == 14
    assert features[FEATURE_NAMES.index("user_click_count")] == 2.0
    assert features[FEATURE_NAMES.index("category_affinity")] == 1.0
    assert features[FEATURE_NAMES.index("brand_affinity")] == 1.0
    assert features[FEATURE_NAMES.index("is_product_context")] == 1.0
    assert features[FEATURE_NAMES.index("is_search_context")] == 0.0

def test_mmr_diversity():
    byid = {
        "p1": Product(id="p1", title="Shoe 1", category=["Shoes"], brand="Nike"),
        "p2": Product(id="p2", title="Shoe 2", category=["Shoes"], brand="Nike"),
        "p3": Product(id="p3", title="Shirt 1", category=["Apparel"], brand="Adidas"),
        "p4": Product(id="p4", title="Hat 1", category=["Accessories"], brand="Puma"),
    }
    candidates = [
        {"product_id": "p1", "score": 0.95},
        {"product_id": "p2", "score": 0.94},  # same category & brand as p1
        {"product_id": "p3", "score": 0.90},  # different category
        {"product_id": "p4", "score": 0.85},  # different category
    ]

    # With diversity_factor = 0.0, order should strictly be score order
    pure_rel = mmr_rerank(candidates, byid, diversity_factor=0.0, limit=4)
    assert [c["product_id"] for c in pure_rel] == ["p1", "p2", "p3", "p4"]

    # With high diversity_factor, p3 (different category) should be boosted above p2
    diverse = mmr_rerank(candidates, byid, diversity_factor=0.7, limit=4)
    diverse_pids = [c["product_id"] for c in diverse]
    assert diverse_pids[0] == "p1"
    assert diverse_pids[1] == "p3"  # p3 jumped ahead of p2 due to category penalty on p2

def test_ml_ranker_fallback_and_training():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_ranker.db")
        model_path = os.path.join(tmpdir, "ranker.joblib")
        test_store = Store(db_path=db_path)

        try:
            # 1. Uninitialized ranker returns None
            ranker = MLRanker(model_path=model_path)
            assert not ranker.is_trained
            assert ranker.score([[0.0] * 14]) is None

            # 2. Add products with diverse text
            prods = [
                Product(id=f"p{i}", title=f"Product item title {i}", description=f"Detailed description of specs {i}", category=[f"Cat{i%3}"], price=float(10 * i + 1))
                for i in range(1, 11)
            ]
            test_store.upsert_products(prods)

            # 3. Add positive events for training
            for i in range(1, 20):
                test_store.add_event(Event(
                    event="purchase" if i % 2 == 0 else "product_click",
                    user_id=f"user_{i%3}",
                    product_id=f"p{(i%5)+1}"
                ))

            # Train with min_positives = 10
            result = train_from_store(test_store, min_positives=10, model_path=model_path)
            assert result is not None
            assert 0.0 <= result.auc <= 1.0
            assert result.n_samples > 0
            assert os.path.exists(model_path)

            # 4. Load trained model
            loaded_ranker = MLRanker(model_path=model_path)
            assert loaded_ranker.load()
            assert loaded_ranker.is_trained

            # 5. Score predictions
            dummy_feats = [[1.0] * 14, [0.0] * 14]
            scores = loaded_ranker.score(dummy_feats)
            assert scores is not None
            assert len(scores) == 2
            assert 0.0 <= scores[0] <= 1.0
            assert 0.0 <= scores[1] <= 1.0
        finally:
            test_store.close()

def test_recommender_with_trained_ranker():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "rec_test.db")
        model_path = os.path.join(tmpdir, "ranker.joblib")
        test_store = Store(db_path=db_path)

        try:
            prods = [
                Product(id=f"prod_{i}", title=f"Electronics device model {i}", description=f"High quality specs for unit {i}", category=["Electronics"], price=100.0)
                for i in range(1, 10)
            ]
            test_store.upsert_products(prods)

            for i in range(1, 15):
                test_store.add_event(Event(
                    event="purchase",
                    user_id="alice",
                    product_id=f"prod_{(i%3)+1}"
                ))

            rec = Recommender(test_store, model_path=model_path)
            # Without model, recommend succeeds using heuristic
            res1 = rec.recommend(RecommendationRequest(user_id="alice", limit=5))
            assert len(res1["recommendations"]) > 0

            # Now train model
            train_from_store(test_store, min_positives=5, model_path=model_path)
            rec.reload_ranker()
            assert rec.ranker.is_trained

            # With model, ml_ranker source should appear in recommendations
            res2 = rec.recommend(RecommendationRequest(user_id="alice", limit=5))
            assert len(res2["recommendations"]) > 0
            assert any("ml_ranker" in r["sources"] for r in res2["recommendations"])
        finally:
            test_store.close()

def test_metrics_store():
    m = MetricsStore()
    m.inc_counter("test_counter", 3.0, {"endpoint": "/v1/recommendations"})
    m.observe_histogram("test_latency", 45.0, {"endpoint": "/v1/recommendations"})

    text = m.generate_prometheus_text()
    assert "# TYPE test_counter counter" in text
    assert 'test_counter{endpoint="/v1/recommendations"} 3.0' in text
    assert "# TYPE test_latency histogram" in text
    assert 'test_latency_count{endpoint="/v1/recommendations"} 1' in text
    assert 'test_latency_bucket{endpoint="/v1/recommendations",le="50.0"} 1' in text

def test_api_v2_endpoints():
    # Test /ready
    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    assert r_ready.json() == {"status": "ready"}

    # Test /metrics
    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "recsys_latency_ms" in r_metrics.text
    assert "# TYPE" in r_metrics.text

    # Test /v1/models/info
    r_info = client.get("/v1/models/info")
    assert r_info.status_code == 200
    assert r_info.json()["status"] in ("ready", "no_model")

    # Test /v1/models/train
    r_train = client.post("/v1/models/train")
    assert r_train.status_code == 200
    assert r_train.json() == {"status": "training_started"}
