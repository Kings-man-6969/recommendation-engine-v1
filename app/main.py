import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, APIRouter, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.models.schemas import Product, Event, RecommendationRequest, RecommendationResponse
from app.storage.store import store
from app.core.recommender import Recommender
from app.core.metrics import metrics
from app.ml.trainer import train_from_store

# JSON Structured Logging Formatter
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(JSONFormatter())
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Avoid duplicate handlers on reload
if not any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JSONFormatter) for h in root_logger.handlers):
    root_logger.handlers = [log_handler]

logger = logging.getLogger("recommender_api")
engine = Recommender(store)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing recommendation engine and storage...")
    try:
        _ = store.products()
        engine.reload_ranker()
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        raise
    yield
    logger.info("Shutting down recommendation engine...")

app = FastAPI(
    title="Recommendation Engine V2",
    version="2.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    metrics.observe_histogram(
        "recsys_latency_ms",
        latency_ms,
        {"endpoint": request.url.path, "method": request.method}
    )
    return response

v1_router = APIRouter(prefix="/v1")

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/ready", tags=["system"])
def ready():
    try:
        _ = store.products()
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "detail": str(e)}
        )

@app.get("/metrics", tags=["system"])
def get_metrics():
    return Response(
        content=metrics.generate_prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

@v1_router.post("/catalog/products", status_code=status.HTTP_201_CREATED, tags=["catalog"])
def upsert_single_product(p: Product):
    store.upsert_product(p)
    return {"status": "ok", "product_id": p.id}

@v1_router.post("/catalog/products/bulk", status_code=status.HTTP_201_CREATED, tags=["catalog"])
def upsert_bulk_products(items: list[Product]):
    store.upsert_products(items)
    return {"status": "ok", "count": len(items)}

@v1_router.post("/events", status_code=status.HTTP_201_CREATED, tags=["events"])
def record_event(e: Event):
    store.add_event(e)
    metrics.inc_counter("recsys_events_ingested_total", 1.0, {"event": e.event})
    return {"status": "ok"}

@v1_router.post("/recommendations", response_model=RecommendationResponse, tags=["recommendations"])
def get_recommendations(req: RecommendationRequest):
    return engine.recommend(req)

@v1_router.post("/models/train", tags=["ml"])
async def trigger_model_training():
    def _run_training():
        try:
            logger.info("Starting background ML ranker training...")
            res = train_from_store(store)
            if res:
                engine.reload_ranker()
                metrics.inc_counter("recsys_train_runs_total", 1.0, {"status": "success"})
                logger.info(f"Ranker training succeeded: {res.to_dict()}")
            else:
                metrics.inc_counter("recsys_train_runs_total", 1.0, {"status": "skipped"})
                logger.info("Ranker training skipped: insufficient data or positives.")
        except Exception as exc:
            metrics.inc_counter("recsys_train_runs_total", 1.0, {"status": "error"})
            logger.exception(f"Error during ranker training: {exc}")

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_training)
    return {"status": "training_started"}

@v1_router.get("/models/info", tags=["ml"])
def get_model_info():
    if engine.ranker.is_trained:
        return {"status": "ready", "metadata": engine.ranker.metadata}
    return {"status": "no_model"}

app.include_router(v1_router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_error", "detail": exc.errors()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "detail": "An unexpected server error occurred."}
    )
