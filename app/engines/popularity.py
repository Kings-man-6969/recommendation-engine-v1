from datetime import datetime, timezone
import math

EVENT_WEIGHTS = {
    'impression': 0.1,
    'product_view': 1.0,
    'product_click': 2.0,
    'wishlist': 4.0,
    'add_to_cart': 5.0,
    'purchase': 8.0
}

class PopularityEngine:
    def __init__(self, events):
        self.events = events

    def score(self, product_id: str, location=None) -> float:
        s = 0.0
        now = datetime.now(timezone.utc)
        for e in self.events:
            if e['product_id'] != product_id:
                continue
            ts_val = e['timestamp']
            ts = datetime.fromisoformat(ts_val.replace('Z', '+00:00')) if isinstance(ts_val, str) else ts_val
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = max(0.0, (now - ts).total_seconds() / 86400.0)
            s += EVENT_WEIGHTS.get(e['event'], 0.0) * math.exp(-age / 14.0)
        return s

    def ranked(self, limit: int = 100) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        now = datetime.now(timezone.utc)
        for e in self.events:
            pid = e['product_id']
            if not pid:
                continue
            w = EVENT_WEIGHTS.get(e['event'], 0.0)
            if w <= 0:
                continue
            ts_val = e['timestamp']
            ts = datetime.fromisoformat(ts_val.replace('Z', '+00:00')) if isinstance(ts_val, str) else ts_val
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = max(0.0, (now - ts).total_seconds() / 86400.0)
            scores[pid] = scores.get(pid, 0.0) + w * math.exp(-age / 14.0)

        if not scores:
            return []
        m = max(scores.values())
        if m <= 0:
            return []
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [(p, s / m) for p, s in sorted_items]
