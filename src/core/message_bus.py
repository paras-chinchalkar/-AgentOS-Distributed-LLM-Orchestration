"""
Message Bus — with optional in-memory fallback when Redis is unavailable.
Detects connection and auto-falls back to a simple asyncio Queue + dict.
"""
import asyncio
import json
import logging
from typing import Any, AsyncIterator

from src.core.config import settings

logger = logging.getLogger(__name__)

# ── Try to import real Redis ──────────────────────────────────────────────────
try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

_pool = None
_USE_MEMORY = False  # set to True after a failed Redis connect


# ── In-memory backend ─────────────────────────────────────────────────────────
_mem_queue: asyncio.Queue = asyncio.Queue()
_mem_pubsub: dict[str, list[asyncio.Queue]] = {}


async def get_redis():
    global _pool, _USE_MEMORY
    if _USE_MEMORY:
        return None
    if not _REDIS_AVAILABLE:
        _USE_MEMORY = True
        logger.warning("redis package not found — using in-memory fallback.")
        return None
    if _pool is None:
        try:
            _pool = aioredis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
            await _pool.ping()
            logger.info("Connected to Redis at %s", settings.redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s) — using in-memory fallback.", exc)
            _pool = None
            _USE_MEMORY = True
    return _pool


async def close_redis() -> None:
    global _pool
    if _pool:
        await _pool.aclose()
        _pool = None


# ── Message Bus ───────────────────────────────────────────────────────────────
class MessageBus:
    def __init__(self, redis=None):
        self.r = redis
        self.stream = settings.redis_stream_name
        self.group = settings.redis_consumer_group

    # Publishing
    async def publish(self, payload: dict[str, Any]) -> str:
        if self.r is None:
            await _mem_queue.put(payload)
            logger.debug("[MemBus] Published: %s", payload)
            return "mem-id"
        msg_id = await self.r.xadd(self.stream, {"data": json.dumps(payload)})
        logger.debug("[Redis] Published [%s]: %s", msg_id, payload)
        return msg_id

    async def ensure_group(self) -> None:
        if self.r is None:
            return
        try:
            await self.r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def consume(
        self, consumer_name: str, batch: int = 10, block_ms: int = 2000
    ) -> AsyncIterator[tuple[str, dict]]:
        await self.ensure_group()
        if self.r is None:
            # In-memory: just drain the asyncio Queue
            while True:
                try:
                    payload = await asyncio.wait_for(_mem_queue.get(), timeout=2.0)
                    yield "mem-id", payload
                    _mem_queue.task_done()
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break
            return

        # Real Redis Streams path
        while True:
            try:
                # First try to recover any pending entries already assigned to this consumer.
                results = await self.r.xreadgroup(
                    self.group, consumer_name, {self.stream: "0"}, count=batch, block=block_ms,
                )
                if not results:
                    results = await self.r.xreadgroup(
                        self.group, consumer_name, {self.stream: ">"}, count=batch, block=block_ms,
                    )
                if results:
                    for _stream, messages in results:
                        for msg_id, fields in messages:
                            payload = json.loads(fields["data"])
                            yield msg_id, payload
                            await self.r.xack(self.stream, self.group, msg_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Redis consume error: %s", exc)
                await asyncio.sleep(2)

    # Pub/Sub (used by WebSocket telemetry)
    async def broadcast_event(self, channel: str, event: dict[str, Any]) -> None:
        if self.r is None:
            for q in _mem_pubsub.get(channel, []):
                await q.put(event)
            return
        await self.r.publish(channel, json.dumps(event))

    async def subscribe_events(self, channel: str) -> AsyncIterator[dict]:
        if self.r is None:
            q: asyncio.Queue = asyncio.Queue()
            _mem_pubsub.setdefault(channel, []).append(q)
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=1.0)
                        yield event
                    except asyncio.TimeoutError:
                        pass
                    except asyncio.CancelledError:
                        break
            finally:
                _mem_pubsub[channel].remove(q)
            return

        pubsub = self.r.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
