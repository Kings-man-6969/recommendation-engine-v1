import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.model_selection import train_test_split

from app.engines.popularity import PopularityEngine
from app.engines.freshness import FreshnessEngine
from app.engines.dense import DenseEngine
from app.ml.features import extract_user_profile, extract_features, UserProfile
from app.ml.ranker import MLRanker, DEFAULT_MODEL_PATH
from app.models.schemas import Context, Product

@dataclass
class TrainResult:
    auc: float
    ndcg_at_10: float
    n_samples: int
    trained_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "auc": self.auc,
            "ndcg_at_10": self.ndcg_at_10,
            "n_samples": self.n_samples,
            "trained_at": self.trained_at,
        }

EVENT_WEIGHTS = {
    "purchase": 3.0,
    "add_to_cart": 2.0,
    "cart": 2.0,
    "checkout_start": 2.5,
    "product_click": 1.0,
    "product_view": 1.0,
}

def train_from_store(
    store: Any,
    min_positives: int = 50,
    model_path: str = DEFAULT_MODEL_PATH
) -> TrainResult | None:
    products = store.products()
    if len(products) < 2:
        return None

    byid = {p.id: p for p in products}
    all_events = store.all_events(limit=100000)
    if not all_events:
        return None

    # Group events by user identity
    user_events: dict[str, list[Any]] = {}
    user_interacted_pids: dict[str, set[str]] = {}

    for row in all_events:
        ev_name = row["event"] if hasattr(row, "__getitem__") else getattr(row, "event", None)
        identity = (row["user_id"] or row["anonymous_id"]) if hasattr(row, "__getitem__") else (getattr(row, "user_id", None) or getattr(row, "anonymous_id", None))
        pid = row["product_id"] if hasattr(row, "__getitem__") else getattr(row, "product_id", None)

        if identity:
            user_events.setdefault(identity, []).append(row)
            if pid:
                user_interacted_pids.setdefault(identity, set()).add(pid)

    # Compute UserProfiles
    user_profiles: dict[str, UserProfile] = {
        uid: extract_user_profile(evts, byid) for uid, evts in user_events.items()
    }

    # Engines for scoring features
    pop_engine = PopularityEngine(all_events)
    pop_scores = dict(pop_engine.ranked(len(products)))
    fresh_engine = FreshnessEngine(products)
    fresh_scores = dict(fresh_engine.ranked(len(products)))
    dense_engine = DenseEngine(products)

    # Collect positives
    positives: list[tuple[str, str, float]] = []  # (identity, pid, weight)
    for row in all_events:
        ev_name = row["event"] if hasattr(row, "__getitem__") else getattr(row, "event", None)
        identity = (row["user_id"] or row["anonymous_id"]) if hasattr(row, "__getitem__") else (getattr(row, "user_id", None) or getattr(row, "anonymous_id", None))
        pid = row["product_id"] if hasattr(row, "__getitem__") else getattr(row, "product_id", None)

        if ev_name in EVENT_WEIGHTS and pid and pid in byid and identity:
            weight = EVENT_WEIGHTS[ev_name]
            positives.append((identity, pid, weight))

    if len(positives) < min_positives:
        return None

    rng = random.Random(42)
    all_pids = [p.id for p in products]

    X: list[list[float]] = []
    y: list[int] = []
    sample_weights: list[float] = []

    # Positives
    for identity, pid, weight in positives:
        u_prof = user_profiles.get(identity, UserProfile())
        prod = byid[pid]
        ctx = Context(type="home")
        p_score = pop_scores.get(pid, 0.0)
        f_score = fresh_scores.get(pid, 0.0)
        feat = extract_features(u_prof, prod, ctx, pop_score=p_score, fresh_score=f_score, dense_sim=0.0)
        X.append(feat)
        y.append(1)
        sample_weights.append(weight)

        # Sample 1-2 negatives for each positive
        user_seen = user_interacted_pids.get(identity, set())
        unseen = [p for p in all_pids if p not in user_seen]
        if unseen:
            neg_pids = rng.sample(unseen, k=min(2, len(unseen)))
            for neg_pid in neg_pids:
                neg_prod = byid[neg_pid]
                neg_feat = extract_features(
                    u_prof,
                    neg_prod,
                    ctx,
                    pop_score=pop_scores.get(neg_pid, 0.0),
                    fresh_score=fresh_scores.get(neg_pid, 0.0),
                    dense_sim=0.0,
                )
                X.append(neg_feat)
                y.append(0)
                sample_weights.append(1.0)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    w_arr = np.array(sample_weights, dtype=np.float32)

    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        return None

    try:
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_arr, y_arr, w_arr, test_size=0.2, random_state=42, stratify=y_arr
        )
    except Exception:
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_arr, y_arr, w_arr, test_size=0.2, random_state=42
        )

    model = HistGradientBoostingClassifier(
        max_iter=200,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    # Evaluation
    val_probas = model.predict_proba(X_val)[:, 1]
    try:
        auc = float(roc_auc_score(y_val, val_probas))
    except Exception:
        auc = 0.5

    try:
        k = min(len(y_val), 10)
        if k > 1 and np.sum(y_val[:k]) > 0:
            ndcg = float(ndcg_score(np.array([y_val[:k]]), np.array([val_probas[:k]]), k=k))
        else:
            ndcg = float(auc)
    except Exception:
        ndcg = float(auc)

    trained_at = datetime.now(timezone.utc).isoformat()
    result = TrainResult(
        auc=round(auc, 4),
        ndcg_at_10=round(ndcg, 4),
        n_samples=len(X_arr),
        trained_at=trained_at,
    )

    ranker = MLRanker(model_path=model_path)
    ranker.model = model
    ranker.metadata = result.to_dict()
    ranker.save()

    return result
