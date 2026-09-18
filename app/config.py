import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    database_path: str = Field(default_factory=lambda: os.getenv("DATABASE_PATH", "recommendation.db"))
    candidate_limit: int = Field(default_factory=lambda: int(os.getenv("CANDIDATE_LIMIT", "200")))
    engine_ttl_seconds: int = Field(default_factory=lambda: int(os.getenv("ENGINE_TTL_SECONDS", "60")))
    events_window_days: int = Field(default_factory=lambda: int(os.getenv("EVENTS_WINDOW_DAYS", "30")))
    weights: dict[str, float] = Field(default_factory=lambda: {
        "personalization": 0.25,
        "intent": 0.25,
        "content": 0.15,
        "dense": 0.20,
        "collaborative": 0.10,
        "geo": 0.10,
        "popularity": 0.10,
        "freshness": 0.05,
    })

settings = Settings()
