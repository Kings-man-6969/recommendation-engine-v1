import time
import uuid
from typing import Any
import numpy as np

from app.config import settings
from app.engines.content import ContentEngine
from app.engines.dense import DenseEngine
from app.engines.behavior import BehaviorEngine
from app.engines.popularity import PopularityEngine
from app.engines.geo import GeoEngine
from app.engines.freshness import FreshnessEngine
from app.engines.diversity import mmr_rerank
from app.ml.features import extract_user_profile, extract_features
from app.ml.ranker import MLRanker, DEFAULT_MODEL_PATH
from app.core.metrics import metrics
from app.models.schemas import Product, RecommendationRequest, RecommendationResponse

class Recommender:
    def __init__(self, store, model_path: str | None = None):
        self.store = store
        self._cache = None
        self._cache_version = -1
        self._cache_ts = 0.0
        self.ranker = MLRanker(model_path=model_path or DEFAULT_MODEL_PATH)
        self.ranker.load()

    def reload_ranker(self) -> bool:
        return self.ranker.load()

    def invalidate(self):
        self._cache = None
        self._cache_version = -1
        self._cache_ts = 0.0
        self.ranker.load()

    def _get_engines(self):
        now = time.time()
        store_version = getattr(self.store, 'version', 0)
        if (
            self._cache is not None
            and self._cache_version == store_version
            and (now - self._cache_ts < settings.engine_ttl_seconds)
        ):
            return self._cache

        products = self.store.products()
        byid = {p.id: p for p in products}
        events = self.store.all_events(since_days=settings.events_window_days)

        content = ContentEngine(products)
        dense = DenseEngine(products)
        behavior = BehaviorEngine(events)
        pop = PopularityEngine(events)
        geo = GeoEngine()
        freshness = FreshnessEngine(products)

        self._cache = (products, byid, content, dense, behavior, pop, geo, freshness)
        self._cache_version = store_version
        self._cache_ts = now
        return self._cache

    def _matches_filters(self, p: Product, filters: dict[str, Any]) -> bool:
        if not filters:
            return True
        for k, v in filters.items():
            if k == "category":
                p_cats = [c.lower() for c in (p.category or [])]
                if isinstance(v, list):
                    if not any(cat.lower() in p_cats for cat in v):
                        return False
                elif isinstance(v, str):
                    if v.lower() not in p_cats:
                        return False
            elif k == "brand":
                p_brand = (p.brand or "").lower()
                if isinstance(v, list):
                    if not any(b.lower() == p_brand for b in v):
                        return False
                elif isinstance(v, str):
                    if v.lower() != p_brand:
                        return False
            elif k == "min_price":
                if p.price is None or p.price < float(v):
                    return False
            elif k == "max_price":
                if p.price is None or p.price > float(v):
                    return False
            elif k == "availability":
                if p.availability != bool(v):
                    return False
            elif k in p.attributes:
                if p.attributes[k] != v:
                    return False
        return True

    def recommend(self, req: RecommendationRequest) -> dict[str, Any]:
        metrics.inc_counter("recsys_requests_total", 1.0, {"context_type": req.context.type})

        products, byid, content, dense, behavior, pop, geo, freshness = self._get_engines()

        identity = req.user_id or req.anonymous_id
        affinity = behavior.user_affinity(identity) if identity else {}
        scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}

        def add(pid: str, value: float, source: str, weight: float = 1.0):
            if pid not in byid or pid in req.exclude_product_ids:
                return
            product = byid[pid]
            if not product.availability or not self._matches_filters(product, req.filters):
                return
            val = max(0.0, min(1.0, float(value)))
            scores[pid] = scores.get(pid, 0.0) + val * weight
            sources.setdefault(pid, set()).add(source)

        context = req.context
        if context.type == 'product' and context.product_id:
            for pid, s in content.similar(context.product_id, settings.candidate_limit):
                add(pid, s, 'content', settings.weights.get('content', 0.15))
            for pid, s in dense.similar(context.product_id, settings.candidate_limit):
                add(pid, s, 'dense', settings.weights.get('dense', 0.20))

        if context.query:
            for pid, s in content.query(context.query, settings.candidate_limit):
                add(pid, s, 'intent', settings.weights.get('intent', 0.25))
            for pid, s in dense.query(context.query, settings.candidate_limit):
                add(pid, s, 'dense_intent', settings.weights.get('dense', 0.20))

        if identity:
            for pid, s in behavior.collaborative(identity, settings.candidate_limit):
                add(pid, s, 'collaborative', settings.weights.get('collaborative', 0.10))

            for pid, s in affinity.items():
                add(pid, s, 'personalization', settings.weights.get('personalization', 0.25))

        ranked_pop = pop.ranked(settings.candidate_limit)
        for pid, s in ranked_pop:
            add(pid, s, 'popularity', settings.weights.get('popularity', 0.10))

        for pid, s in freshness.ranked(settings.candidate_limit):
            add(pid, s, 'freshness', settings.weights.get('freshness', 0.05))

        if req.location:
            for p in products:
                if p.location:
                    add(p.id, geo.score(req.location, p.location), 'geo', settings.weights.get('geo', 0.10))

        # Contextual intent boost from recent session search terms
        if req.session_id:
            recent = [e for e in self.store.events(session_id=req.session_id, limit=20) if e['event'] == 'search' and e['query']]
            for e in recent[:3]:
                for pid, s in content.query(e['query'], 50):
                    add(pid, s, 'session_intent', settings.weights.get('intent', 0.25) * 0.75)

        # Normalize heuristic blended score safely
        active_weights = sum(w for w in settings.weights.values() if w > 0)
        normalizer = active_weights if active_weights > 0 else 1.0

        candidates: list[dict[str, Any]] = []
        for pid, s in scores.items():
            src = sorted(list(sources.get(pid, [])))
            reason = self._reason(src)
            normalized_score = round(min(1.0, s / normalizer), 6)
            candidates.append({
                'product_id': pid,
                'score': normalized_score,
                'reason': reason,
                'sources': src
            })

        # Sort candidates by blended heuristic score initially
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Stage 2: ML Ranker scoring (if trained model is available)
        if self.ranker.is_trained and candidates:
            pool = candidates[:settings.candidate_limit]
            user_evts = self.store.events(
                user_id=req.user_id,
                anonymous_id=req.anonymous_id,
                session_id=req.session_id,
                limit=100
            ) if (identity or req.session_id) else []
            user_profile = extract_user_profile(user_evts, byid)
            pop_map = dict(pop.ranked(len(products)))
            fresh_map = dict(freshness.ranked(len(products)))

            feat_matrix: list[list[float]] = []
            for c in pool:
                p = byid[c['product_id']]
                d_sim = 0.0
                if req.context.product_id and dense.matrix is not None and req.context.product_id in dense.index and p.id in dense.index:
                    i_ctx = dense.index[req.context.product_id]
                    i_cand = dense.index[p.id]
                    d_sim = float(np.dot(dense.matrix[i_cand], dense.matrix[i_ctx]))
                feat = extract_features(
                    user_profile,
                    p,
                    req.context,
                    pop_score=pop_map.get(p.id, 0.0),
                    fresh_score=fresh_map.get(p.id, 0.0),
                    dense_sim=d_sim
                )
                feat_matrix.append(feat)

            t0 = time.perf_counter()
            ml_scores = self.ranker.score(feat_matrix)
            inference_ms = (time.perf_counter() - t0) * 1000.0
            metrics.observe_histogram("recsys_ranker_inference_ms", inference_ms)

            if ml_scores is not None and len(ml_scores) == len(pool):
                for idx, c in enumerate(pool):
                    c['score'] = round(float(ml_scores[idx]), 6)
                    if 'ml_ranker' not in c['sources']:
                        c['sources'].append('ml_ranker')
                        c['sources'].sort()
                candidates = pool
                candidates.sort(key=lambda x: x['score'], reverse=True)

        # Stage 3: MMR Diversity Re-ranking
        reranked = mmr_rerank(
            candidates,
            byid=byid,
            diversity_factor=req.diversity_factor,
            limit=req.limit
        )

        return {'request_id': str(uuid.uuid4()), 'recommendations': reranked}

    def _reason(self, src: list[str]) -> str:
        if 'session_intent' in src or 'intent' in src or 'dense_intent' in src:
            return 'matches_current_intent'
        if 'content' in src or 'dense' in src:
            return 'similar_to_current_product'
        if 'collaborative' in src:
            return 'popular_with_similar_users'
        if 'personalization' in src:
            return 'based_on_your_activity'
        if 'freshness' in src:
            return 'recently_added'
        if 'geo' in src:
            return 'relevant_near_your_location'
        return 'popular_products'
