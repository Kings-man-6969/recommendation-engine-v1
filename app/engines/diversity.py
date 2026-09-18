from typing import Any
from app.models.schemas import Product

def _product_similarity(p1: Product | None, p2: Product | None) -> float:
    if p1 is None or p2 is None:
        return 0.0
    cat1 = p1.category[0].strip().lower() if (p1.category and len(p1.category) > 0 and p1.category[0]) else None
    cat2 = p2.category[0].strip().lower() if (p2.category and len(p2.category) > 0 and p2.category[0]) else None
    if cat1 is not None and cat2 is not None and cat1 == cat2:
        return 1.0

    b1 = p1.brand.strip().lower() if (p1.brand and p1.brand.strip()) else None
    b2 = p2.brand.strip().lower() if (p2.brand and p2.brand.strip()) else None
    if b1 is not None and b2 is not None and b1 == b2:
        return 0.5

    return 0.0

def mmr_rerank(
    candidates: list[dict[str, Any]],
    byid: dict[str, Product],
    diversity_factor: float = 0.3,
    limit: int = 20
) -> list[dict[str, Any]]:
    """
    Maximal Marginal Relevance (MMR) re-ranking.
    Balancing relevance score with diversity (category and brand overlap).
    diversity_factor (lambda):
      0.0 => pure relevance
      1.0 => pure diversity
    """
    if not candidates or limit <= 0:
        return []
    if diversity_factor <= 0.0 or len(candidates) <= 1:
        return candidates[:limit]

    diversity_factor = max(0.0, min(1.0, float(diversity_factor)))
    rel_weight = 1.0 - diversity_factor

    # Normalize candidate scores to [0, 1] for fair combination with overlap penalties
    max_score = max(c["score"] for c in candidates)
    min_score = min(c["score"] for c in candidates)
    score_range = max_score - min_score if max_score > min_score else 1.0

    unselected = list(candidates)
    selected: list[dict[str, Any]] = []

    # First item is always the most relevant item
    first_item = unselected.pop(0)
    selected.append(first_item)

    while unselected and len(selected) < limit:
        best_idx = 0
        best_mmr = -float("inf")

        for idx, cand in enumerate(unselected):
            norm_rel = (cand["score"] - min_score) / score_range
            cand_prod = byid.get(cand["product_id"])

            # Find max overlap with already selected items
            max_sim = 0.0
            for sel in selected:
                sel_prod = byid.get(sel["product_id"])
                sim = _product_similarity(cand_prod, sel_prod)
                if sim > max_sim:
                    max_sim = sim
                if max_sim >= 1.0:
                    break

            mmr_score = (rel_weight * norm_rel) - (diversity_factor * max_sim)
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        selected.append(unselected.pop(best_idx))

    return selected
