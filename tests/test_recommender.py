from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models.schemas import Product, Event, Context, RecommendationRequest, Location
from app.storage.store import Store
from app.engines.content import ContentEngine
from app.engines.popularity import PopularityEngine
from app.engines.freshness import FreshnessEngine
from app.core.recommender import Recommender

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}

def test_event_requires_identity():
    # Both None should raise ValidationError
    with pytest.raises(ValidationError):
        Event(event="product_view", user_id=None, anonymous_id=None, product_id="p1")

    # user_id present is valid
    e1 = Event(event="product_view", user_id="u1", product_id="p1")
    assert e1.user_id == "u1"

    # anonymous_id present is valid
    e2 = Event(event="product_view", anonymous_id="anon_1", product_id="p1")
    assert e2.anonymous_id == "anon_1"

def test_sparse_matrix_truth_value_fix():
    # >1 products previously caused ValueError: The truth value of an array is ambiguous
    p1 = Product(id="p1", title="Wireless Noise-Cancelling Headphones", category=["Audio", "Electronics"])
    p2 = Product(id="p2", title="Bluetooth Earbuds Pro", category=["Audio", "Electronics"])
    p3 = Product(id="p3", title="Coffee Maker Machine", category=["Home", "Kitchen"])
    
    engine = ContentEngine([p1, p2, p3])
    # Calling similar() should not raise ValueError
    sims = engine.similar("p1")
    assert len(sims) == 2
    assert sims[0][0] == "p2"  # Audio should be more similar to audio than coffee

    # Calling query() should not raise ValueError
    q_results = engine.query("wireless audio")
    assert len(q_results) > 0

def test_popularity_time_decay():
    now = datetime.now(timezone.utc)
    recent_ts = now.isoformat()
    old_ts = (now - timedelta(days=60)).isoformat()

    events = [
        {"product_id": "p_old", "event": "purchase", "timestamp": old_ts},
        {"product_id": "p_recent", "event": "purchase", "timestamp": recent_ts},
    ]
    pop = PopularityEngine(events)
    ranked = pop.ranked()
    assert len(ranked) == 2
    # The recent purchase must score significantly higher than the 60-day old purchase
    assert ranked[0][0] == "p_recent"
    assert ranked[0][1] > ranked[1][1]

def test_freshness_engine():
    now = datetime.now(timezone.utc)
    new_product = Product(id="p_new", title="New Smartphone", created_at=now)
    old_product = Product(id="p_old", title="Old Smartphone", created_at=now - timedelta(days=30))

    freshness = FreshnessEngine([new_product, old_product])
    ranked = freshness.ranked()
    assert len(ranked) == 2
    assert ranked[0][0] == "p_new"
    assert ranked[0][1] > ranked[1][1]

def test_recommendation_filters(tmp_path):
    test_db = str(tmp_path / "test_filter.db")
    store = Store(db_path=test_db)
    
    p1 = Product(id="p1", title="Gaming Laptop", category=["Computers"], brand="BrandX", price=1200.0)
    p2 = Product(id="p2", title="Office Laptop", category=["Computers"], brand="BrandY", price=600.0)
    p3 = Product(id="p3", title="Smartphone", category=["Phones"], brand="BrandX", price=800.0)
    store.upsert_products([p1, p2, p3])

    recommender = Recommender(store)

    # Filter by category Computers
    req_cat = RecommendationRequest(
        context=Context(type="home"),
        filters={"category": "Computers"},
        limit=10
    )
    res_cat = recommender.recommend(req_cat)
    cat_ids = [r["product_id"] for r in res_cat["recommendations"]]
    assert "p1" in cat_ids or "p2" in cat_ids
    assert "p3" not in cat_ids

    # Filter by max_price
    req_price = RecommendationRequest(
        context=Context(type="home"),
        filters={"max_price": 700.0},
        limit=10
    )
    res_price = recommender.recommend(req_price)
    price_ids = [r["product_id"] for r in res_price["recommendations"]]
    assert "p2" in price_ids
    assert "p1" not in price_ids
    assert "p3" not in price_ids

def test_engine_caching(tmp_path):
    test_db = str(tmp_path / "test_cache.db")
    store = Store(db_path=test_db)
    p1 = Product(id="p1", title="Item 1")
    store.upsert_product(p1)

    recommender = Recommender(store)
    # First call builds cache
    engines1 = recommender._get_engines()
    # Second call returns cached instance without rebuilding
    engines2 = recommender._get_engines()
    assert engines1 is engines2

    # After store change (version increments), cache invalidates
    p2 = Product(id="p2", title="Item 2")
    store.upsert_product(p2)
    engines3 = recommender._get_engines()
    assert engines3 is not engines1
    assert len(engines3[0]) == 2

def test_full_api_flow():
    # 1. Upsert product
    p_data = {
        "id": "prod_test_1",
        "title": "Ergonomic Mechanical Keyboard",
        "description": "Silent red switches for coding",
        "category": ["Peripherals", "Keyboards"],
        "brand": "KeyMaster",
        "price": 99.99
    }
    r_prod = client.post("/v1/catalog/products", json=p_data)
    assert r_prod.status_code == 201

    # 2. Record event
    e_data = {
        "event": "product_view",
        "user_id": "user_api_1",
        "product_id": "prod_test_1"
    }
    r_evt = client.post("/v1/events", json=e_data)
    assert r_evt.status_code == 201

    # 3. Request recommendations
    rec_payload = {
        "user_id": "user_api_1",
        "context": {"type": "home"},
        "limit": 5
    }
    r_rec = client.post("/v1/recommendations", json=rec_payload)
    assert r_rec.status_code == 200
    data = r_rec.json()
    assert "request_id" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
