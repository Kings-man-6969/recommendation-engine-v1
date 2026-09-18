from typing import Any, Iterable
import numpy as np
from app.models.schemas import Product, Context

FEATURE_NAMES = [
    "user_click_count",
    "user_cart_count",
    "user_purchase_count",
    "user_avg_price_viewed",
    "product_popularity_score",
    "product_freshness_score",
    "product_price",
    "product_availability",
    "category_affinity",
    "brand_affinity",
    "dense_cosine_sim",
    "price_ratio",
    "is_product_context",
    "is_search_context",
]

class UserProfile:
    def __init__(
        self,
        click_count: int = 0,
        cart_count: int = 0,
        purchase_count: int = 0,
        avg_price: float = 0.0,
        categories: set[str] | None = None,
        brands: set[str] | None = None,
    ):
        self.click_count = click_count
        self.cart_count = cart_count
        self.purchase_count = purchase_count
        self.avg_price = avg_price
        self.categories = set(categories or [])
        self.brands = set(brands or [])

def extract_user_profile(events: Iterable[Any], byid: dict[str, Product]) -> UserProfile:
    clicks = 0
    carts = 0
    purchases = 0
    prices: list[float] = []
    categories: set[str] = set()
    brands: set[str] = set()

    for row in events:
        # Support dict, sqlite3.Row, or Event model
        ev_name = row["event"] if isinstance(row, (dict, list)) or hasattr(row, "__getitem__") else getattr(row, "event", None)
        pid = row["product_id"] if isinstance(row, (dict, list)) or hasattr(row, "__getitem__") else getattr(row, "product_id", None)

        if ev_name in ("product_click", "product_view"):
            clicks += 1
        elif ev_name in ("add_to_cart", "cart"):
            carts += 1
        elif ev_name in ("purchase", "checkout_start"):
            purchases += 1

        if pid and pid in byid:
            p = byid[pid]
            if p.price is not None and p.price > 0:
                prices.append(float(p.price))
            if p.category:
                for c in p.category:
                    categories.add(c.strip().lower())
            if p.brand:
                brands.add(p.brand.strip().lower())

    avg_price = float(np.mean(prices)) if prices else 0.0
    return UserProfile(
        click_count=clicks,
        cart_count=carts,
        purchase_count=purchases,
        avg_price=avg_price,
        categories=categories,
        brands=brands,
    )

def extract_features(
    user_profile: UserProfile,
    product: Product,
    context: Context | None = None,
    pop_score: float = 0.0,
    fresh_score: float = 0.0,
    dense_sim: float = 0.0,
) -> list[float]:
    # User history (4)
    u_click = float(user_profile.click_count)
    u_cart = float(user_profile.cart_count)
    u_purchase = float(user_profile.purchase_count)
    u_avg_price = float(user_profile.avg_price)

    # Product stats (4)
    p_pop = float(pop_score)
    p_fresh = float(fresh_score)
    p_price = float(product.price or 0.0)
    p_avail = 1.0 if product.availability else 0.0

    # Cross signals (4)
    cat_affinity = 0.0
    if product.category:
        for c in product.category:
            if c.strip().lower() in user_profile.categories:
                cat_affinity = 1.0
                break

    brand_affinity = 0.0
    if product.brand and product.brand.strip().lower() in user_profile.brands:
        brand_affinity = 1.0

    d_sim = float(dense_sim)

    if p_price > 0 and u_avg_price > 0:
        ratio = min(10.0, max(0.01, u_avg_price / p_price))
    else:
        ratio = 1.0

    # Context (2)
    ctx_type = context.type if context else "home"
    is_prod_ctx = 1.0 if ctx_type == "product" else 0.0
    is_search_ctx = 1.0 if (ctx_type == "search" or (context and bool(context.query))) else 0.0

    return [
        u_click,
        u_cart,
        u_purchase,
        u_avg_price,
        p_pop,
        p_fresh,
        p_price,
        p_avail,
        cat_affinity,
        brand_affinity,
        d_sim,
        ratio,
        is_prod_ctx,
        is_search_ctx,
    ]
