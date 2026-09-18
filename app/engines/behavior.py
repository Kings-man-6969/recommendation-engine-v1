from collections import defaultdict, Counter

WEIGHTS = {
    'product_view': 1,
    'product_click': 2,
    'wishlist': 5,
    'add_to_cart': 8,
    'purchase': 15,
    'remove_from_cart': -5,
    'purchase_cancelled': -10
}

POSITIVE_EVENTS = {'product_view', 'product_click', 'wishlist', 'add_to_cart', 'purchase'}

class BehaviorEngine:
    def __init__(self, events):
        self.events = events

    def user_affinity(self, identity: str | None) -> dict[str, float]:
        if not identity:
            return {}
        scores = defaultdict(float)
        for e in self.events:
            if (e['user_id'] == identity or e['anonymous_id'] == identity) and e['product_id']:
                scores[e['product_id']] += WEIGHTS.get(e['event'], 0)
        if not scores:
            return {}
        m = max(scores.values()) or 1.0
        return {k: max(0.0, v / m) for k, v in scores.items()}

    def collaborative(self, identity: str | None, limit: int = 100) -> list[tuple[str, float]]:
        if not identity:
            return []

        mine = set()
        users = defaultdict(set)

        # Single pass over events
        for e in self.events:
            pid = e['product_id']
            if not pid or e['event'] not in POSITIVE_EVENTS:
                continue
            u = e['user_id'] or e['anonymous_id']
            if not u:
                continue
            users[u].add(pid)
            if u == identity or e['user_id'] == identity or e['anonymous_id'] == identity:
                mine.add(pid)

        if not mine:
            return []

        counts = Counter()
        for u, items in users.items():
            if u == identity:
                continue
            overlap = len(mine & items)
            if overlap > 0:
                for p in items - mine:
                    counts[p] += overlap

        if not counts:
            return []
        m = max(counts.values()) or 1.0
        return [(p, v / m) for p, v in counts.most_common(limit)]
