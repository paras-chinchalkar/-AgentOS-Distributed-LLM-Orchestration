"""
message_bus.py — Redis Streams message bus.
All inter-agent communication goes through here.
Uses consumer groups for at-least-once delivery and automatic failover.
Agents never call each other directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

STREAMS = [
    "tasks:inbox",      # raw task submissions
    "tasks:queue",      # scheduler → agents
    "tasks:results",    # agent → aggregator
    "tasks:errors",     # failed tasks awaiting retry
    "tasks:dlq",        # dead-letter queue
    "agents:heartbeat", # agent liveness signals
]
CONSUMER_GROUP = "agents"
ACK_TIMEOUT_MS = 30_000  # 30s — claim pending msgs from crashed consumers


class MessageBus:
    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._url = redis_url
        self._redis: Optional[aioredis.Redis] = None
        self._consumer_id = f"consumer-{uuid.uuid4().hex[:8]}"

    async def connect(self) -> None:
        self._redis = aioredis.from_url(self._url, decode_responses=True)
        await self._ensure_streams()
        logger.info("[message_bus] connected — consumer id: %s", self._consumer_id)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    # ------------------------------------------------------------------ #
    # Publish                                                              #
    # ------------------------------------------------------------------ #

    async def publish(self, stream: str, payload: Dict[str, Any]) -> str:
        """
        Publish a message. Returns the Redis entry ID.
        Payload values are JSON-serialised so they survive the string-only
        Redis wire format.
        """
        fields = {k: json.dumps(v) if not isinstance(v, str) else v
                  for k, v in payload.items()}
        msg_id = await self._redis.xadd(stream, fields)
        logger.debug("[bus] → %s id=%s", stream, msg_id)
        return msg_id

    # ------------------------------------------------------------------ #
    # Consume                                                              #
    # ------------------------------------------------------------------ #

    async def consume(
        self,
        stream: str,
        handler: Callable[[str, Dict[str, Any]], Any],
        *,
        batch_size: int = 1,
    ) -> None:
        """
        Blocking consumer loop.
        - Reads new messages with XREADGROUP.
        - On success, ACKs immediately (at-least-once).
        - On startup, claims any pending messages from timed-out consumers.
        """
        logger.info("[bus] consuming stream=%s consumer=%s", stream, self._consumer_id)
        await self._claim_pending(stream)

        while True:
            try:
                messages = await self._redis.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=self._consumer_id,
                    streams={stream: ">"},
                    count=batch_size,
                    block=1000,   # ms; unblocks every 1s so we can react to shutdown
                )
                for _, entries in (messages or []):
                    for entry_id, raw in entries:
                        payload = {k: self._decode(v) for k, v in raw.items()}
                        try:
                            await handler(entry_id, payload)
                            await self._redis.xack(stream, CONSUMER_GROUP, entry_id)
                        except Exception:
                            logger.exception("[bus] handler error for %s — will retry", entry_id)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[bus] consume error — reconnecting in 2s")
                await asyncio.sleep(2)

    async def consume_many(
        self,
        streams: List[str],
        handler: Callable[[str, str, Dict[str, Any]], Any],
    ) -> None:
        """Fan-out consumer across multiple streams concurrently."""
        tasks = [
            asyncio.create_task(
                self.consume(s, lambda eid, p, _s=s: handler(_s, eid, p))
            )
            for s in streams
        ]
        await asyncio.gather(*tasks)

    # ------------------------------------------------------------------ #
    # Point-to-point (request / reply pattern)                           #
    # ------------------------------------------------------------------ #

    async def request(
        self,
        stream: str,
        payload: Dict[str, Any],
        reply_stream: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Publish to `stream` and block until a matching reply appears in
        `reply_stream` (keyed on correlation_id). Useful for RPC-style calls.
        """
        correlation_id = uuid.uuid4().hex
        reply_stream = reply_stream or f"reply:{correlation_id}"
        payload["correlation_id"] = correlation_id
        payload["reply_to"] = reply_stream

        await self.publish(stream, payload)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            messages = await self._redis.xread({reply_stream: "0"}, count=1, block=500)
            for _, entries in (messages or []):
                for entry_id, raw in entries:
                    await self._redis.delete(reply_stream)
                    return {k: self._decode(v) for k, v in raw.items()}
        return None

    # ------------------------------------------------------------------ #
    # Observability helpers                                               #
    # ------------------------------------------------------------------ #

    async def stream_lengths(self) -> Dict[str, int]:
        pipe = self._redis.pipeline(transaction=False)
        for s in STREAMS:
            pipe.xlen(s)
        counts = await pipe.execute()
        return dict(zip(STREAMS, counts))

    async def pending_counts(self) -> Dict[str, int]:
        result = {}
        for s in STREAMS:
            info = await self._redis.xpending(s, CONSUMER_GROUP)
            result[s] = info.get("pending", 0) if isinstance(info, dict) else 0
        return result

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    async def _ensure_streams(self) -> None:
        for stream in STREAMS:
            try:
                await self._redis.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
                logger.debug("[bus] created consumer group for %s", stream)
            except aioredis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

    async def _claim_pending(self, stream: str) -> None:
        """Reclaim messages pending for longer than ACK_TIMEOUT_MS."""
        try:
            pending = await self._redis.xpending_range(
                stream, CONSUMER_GROUP, min="-", max="+", count=50
            )
            claimable = [
                p["message_id"] for p in (pending or [])
                if p.get("time_since_delivered", 0) > ACK_TIMEOUT_MS
            ]
            if claimable:
                await self._redis.xclaim(
                    stream, CONSUMER_GROUP, self._consumer_id,
                    min_idle_time=ACK_TIMEOUT_MS, message_ids=claimable,
                )
                logger.info("[bus] reclaimed %d pending messages from %s", len(claimable), stream)
        except Exception:
            logger.debug("[bus] pending claim skipped for %s", stream)

    @staticmethod
    def _decode(value: str) -> Any:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
