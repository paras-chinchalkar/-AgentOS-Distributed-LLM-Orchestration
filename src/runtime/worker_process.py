import asyncio
import logging

from src.core.config import settings
from src.core.db import init_db
from src.core.message_bus import MessageBus, get_redis, close_redis
from src.runtime.lifecycle import LifecycleManager
from src.runtime.worker import run_worker

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

async def main() -> None:
    await init_db()
    redis = await get_redis()
    bus = MessageBus(redis)
    lifecycle_manager = LifecycleManager(bus)
    await bus.ensure_group()
    try:
        await run_worker(bus, lifecycle_manager, worker_id="worker-1")
    finally:
        await close_redis()

if __name__ == "__main__":
    asyncio.run(main())
