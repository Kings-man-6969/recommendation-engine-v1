from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

class Location(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

class Product(BaseModel):
    id: str
    title: str
    description: str = ""
    category: list[str] = Field(default_factory=list)
    brand: str | None = None
    price: float | None = None
    currency: str | None = None
    availability: bool = True
    stock: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    location: Location | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None

class Event(BaseModel):
    event: Literal[
        "product_view", "product_click", "search", "category_view",
        "add_to_cart", "remove_from_cart", "wishlist", "purchase",
        "review", "share", "impression", "checkout_start", "purchase_cancelled"
    ]
    user_id: str | None = None
    anonymous_id: str | None = None
    session_id: str | None = None
    product_id: str | None = None
    query: str | None = None
    category: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    location: Location | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_identity(self):
        if not self.user_id and not self.anonymous_id:
            raise ValueError("Event must have at least one of user_id or anonymous_id")
        return self

class Context(BaseModel):
    type: Literal["home", "search", "product", "category", "cart", "checkout", "wishlist"] = "home"
    product_id: str | None = None
    query: str | None = None
    category: str | None = None

class RecommendationRequest(BaseModel):
    user_id: str | None = None
    anonymous_id: str | None = None
    session_id: str | None = None
    context: Context = Field(default_factory=Context)
    location: Location | None = None
    limit: int = Field(default=20, ge=1, le=100)
    diversity_factor: float = Field(default=0.3, ge=0.0, le=1.0)
    exclude_product_ids: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)

class Recommendation(BaseModel):
    product_id: str
    score: float
    reason: str
    sources: list[str] = Field(default_factory=list)

class RecommendationResponse(BaseModel):
    request_id: str
    recommendations: list[Recommendation]
