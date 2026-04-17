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
from sqlalchemy import text

from src.core.config import settings
from src.core.db import init_db, _resolve_engine
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


async def _reset_database() -> None:
    """Drop and recreate the public schema, giving every request a clean slate.

    Only runs against PostgreSQL — SQLite does not support schema-level resets
    and is used only as a local fallback, so we skip it silently.
    """
    import src.core.db as _db_module  # import the live module so we see the current _engine

    await _resolve_engine()  # ensure the engine is initialised
    engine = _db_module._engine
    if engine is None or engine.dialect.name != "postgresql":
        logger.debug("DB reset skipped (not PostgreSQL).")
        return

    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
        await conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
        await conn.execute(text("GRANT ALL ON SCHEMA public TO postgres"))

    # Recreate all ORM-managed tables so the next request starts with a valid schema.
    async with engine.begin() as conn:
        from src.core.db import Base
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database reset complete — public schema dropped and recreated.")


@app.middleware("http")
async def db_reset_middleware(request: Request, call_next):
    """Send the response to the client first, then reset the database."""
    response = await call_next(request)
    try:
        await _reset_database()
    except Exception:
        logger.exception("Database reset failed after request to %s", request.url.path)
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
