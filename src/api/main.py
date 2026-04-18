"""
FastAPI Application Entry Point.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.core.config import settings
from src.core.db import init_db, Base
from src.core.message_bus import MessageBus, get_redis, close_redis
from src.runtime.lifecycle import LifecycleManager
from src.runtime.worker import run_worker
from src.api.routes import router
from src.api.telemetry import ws_router

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ── Global singletons (injected by DI helpers) ────────────────────────────────
message_bus: MessageBus = None   # type: ignore
lifecycle_manager: LifecycleManager = None  # type: ignore
_worker_task: asyncio.Task = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global message_bus, lifecycle_manager, _worker_task

    # Initialise DB
    logger.info("Initialising database schema...")
    await init_db()

    # Connect to Redis
    redis = await get_redis()
    message_bus = MessageBus(redis)
    lifecycle_manager = LifecycleManager(message_bus)

    # Ensure consumer group exists
    await message_bus.ensure_group()

    # Start background worker
    _worker_task = asyncio.create_task(
        run_worker(message_bus, lifecycle_manager, worker_id="worker-1")
    )
    logger.info("Background worker task started.")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    await close_redis()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Agent Orchestration Runtime",
    description="Distributed LLM Runtime — multi-agent orchestration with Redis Streams + LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

# HTTP methods that mutate state — a reset is appropriate after these.
_WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}


async def _reset_database() -> None:
    """Drop and recreate all tables, returning the database to a clean state."""
    from src.core.db import _engine
    if _engine is None:
        return
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database reset complete.")


@app.middleware("http")
async def db_reset_middleware(request: Request, call_next):
    """Reset the database after write operations (POST/PUT/DELETE/PATCH).

    GET, HEAD, and OPTIONS requests are read-only — skipping the reset for
    these methods ensures that workflow data and executive summaries written
    by the background worker remain visible when the frontend polls for them.
    """
    response = await call_next(request)
    if request.method in _WRITE_METHODS:
        await _reset_database()
    return response


# REST + WebSocket routes
app.include_router(router)
app.include_router(ws_router)

# Serve static dashboard
app.mount("/static", StaticFiles(directory="src/ui"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("src/ui/dashboard.html")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
