from datetime import datetime, timezone
import math
from app.models.schemas import Product

class FreshnessEngine:
    def __init__(self, products: list[Product], half_life_days: float = 7.0):
        self.products = products
        self.half_life_days = half_life_days

    def score(self, product: Product, now: datetime | None = None) -> float:
        current_time = now or datetime.now(timezone.utc)
        created_at = product.created_at
        if created_at is None and product.metadata:
            meta_ca = product.metadata.get("created_at")
            if meta_ca:
                try:
                    if isinstance(meta_ca, str):
                        created_at = datetime.fromisoformat(meta_ca.replace('Z', '+00:00'))
                    elif isinstance(meta_ca, (int, float)):
                        created_at = datetime.fromtimestamp(meta_ca, tz=timezone.utc)
                except Exception:
                    pass

        if created_at is None:
            return 0.5

        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        age_days = max(0.0, (current_time - created_at).total_seconds() / 86400.0)
        return float(math.exp(-age_days / self.half_life_days))

    def ranked(self, limit: int = 100) -> list[tuple[str, float]]:
        now = datetime.now(timezone.utc)
        scores = [(p.id, self.score(p, now)) for p in self.products]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]
