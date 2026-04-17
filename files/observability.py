"""
observability.py — Structured logging, OpenTelemetry tracing,
and Prometheus metrics for full execution visibility.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, Optional

# OpenTelemetry (install: opentelemetry-sdk opentelemetry-exporter-otlp)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Prometheus (install: prometheus_client)
from prometheus_client import (
    Counter, Gauge, Histogram, start_http_server, CollectorRegistry,
)

logger = logging.getLogger("agent_os")


# ------------------------------------------------------------------ #
# Structured JSON logging                                             #
# ------------------------------------------------------------------ #

class StructuredFormatter(logging.Formatter):
    """Emit newline-delimited JSON log records."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        payload = {
            "ts":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    logging.basicConfig(handlers=[handler], level=level)
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)


# ------------------------------------------------------------------ #
# OpenTelemetry tracing                                              #
# ------------------------------------------------------------------ #

_tracer: Optional[trace.Tracer] = None


def init_tracing(otlp_endpoint: str = "http://localhost:4317", service: str = "agent-os") -> None:
    global _tracer
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service)
    logger.info("[obs] OpenTelemetry tracing → %s", otlp_endpoint)


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        return trace.get_tracer("agent-os")
    return _tracer


@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[trace.Span]:
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, str(v))
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            s.set_status(trace.StatusCode.ERROR, str(exc))
            raise


@asynccontextmanager
async def async_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> AsyncIterator[trace.Span]:
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, str(v))
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            s.set_status(trace.StatusCode.ERROR, str(exc))
            raise


# ------------------------------------------------------------------ #
# Prometheus metrics                                                  #
# ------------------------------------------------------------------ #

REGISTRY = CollectorRegistry()

tasks_dispatched_total = Counter(
    "tasks_dispatched_total", "Total tasks dispatched to agents",
    ["agent_type"], registry=REGISTRY,
)
tasks_completed_total = Counter(
    "tasks_completed_total", "Tasks completed successfully",
    ["agent_type"], registry=REGISTRY,
)
tasks_failed_total = Counter(
    "tasks_failed_total", "Tasks that failed after all retries",
    ["agent_type", "reason"], registry=REGISTRY,
)
tasks_retried_total = Counter(
    "tasks_retried_total", "Task retry attempts",
    ["agent_type"], registry=REGISTRY,
)
active_agents_gauge = Gauge(
    "active_agents", "Number of currently running agents",
    ["agent_type"], registry=REGISTRY,
)
task_duration_seconds = Histogram(
    "task_duration_seconds", "Task execution latency",
    ["agent_type"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
    registry=REGISTRY,
)
llm_tokens_total = Counter(
    "llm_tokens_total", "LLM tokens consumed",
    ["direction", "model"], registry=REGISTRY,
)
stream_lag_gauge = Gauge(
    "stream_lag_messages", "Pending messages in Redis stream",
    ["stream"], registry=REGISTRY,
)


def start_metrics_server(port: int = 9090) -> None:
    start_http_server(port, registry=REGISTRY)
    logger.info("[obs] Prometheus metrics → http://localhost:%d/metrics", port)


# ------------------------------------------------------------------ #
# High-level execution tracker                                       #
# ------------------------------------------------------------------ #

class ExecutionTracker:
    """
    Wraps a single task execution with timing, tracing, and metrics.
    Usage:
        async with ExecutionTracker(task_id, agent_type) as tracker:
            result = await agent.run(...)
            tracker.record_tokens(input_t, output_t, model)
    """

    def __init__(self, task_id: str, agent_type: str) -> None:
        self.task_id = task_id
        self.agent_type = agent_type
        self._start: float = 0.0
        self._span_cm = None
        self._span = None

    async def __aenter__(self) -> "ExecutionTracker":
        self._start = time.perf_counter()
        self._span_cm = async_span(
            f"task.execute",
            {"task.id": self.task_id, "agent.type": self.agent_type},
        )
        self._span = await self._span_cm.__aenter__()
        active_agents_gauge.labels(self.agent_type).inc()
        tasks_dispatched_total.labels(self.agent_type).inc()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        duration = time.perf_counter() - self._start
        task_duration_seconds.labels(self.agent_type).observe(duration)
        active_agents_gauge.labels(self.agent_type).dec()

        if exc_type is None:
            tasks_completed_total.labels(self.agent_type).inc()
            self._span.set_attribute("task.status", "done")
        else:
            tasks_failed_total.labels(self.agent_type, type(exc).__name__).inc()
            self._span.set_attribute("task.status", "failed")

        await self._span_cm.__aexit__(exc_type, exc, tb)
        logger.info(
            "[obs] task %s finished in %.2fs | status=%s",
            self.task_id, duration, "done" if exc_type is None else "failed",
        )

    def record_tokens(self, input_tokens: int, output_tokens: int, model: str) -> None:
        llm_tokens_total.labels("input", model).inc(input_tokens)
        llm_tokens_total.labels("output", model).inc(output_tokens)

    def record_retry(self) -> None:
        tasks_retried_total.labels(self.agent_type).inc()


# ------------------------------------------------------------------ #
# Stream lag monitor (background task)                               #
# ------------------------------------------------------------------ #

async def monitor_stream_lag(message_bus, interval: float = 15.0) -> None:
    """
    Periodically polls Redis Streams and updates the stream_lag_gauge.
    Run as an asyncio background task.
    """
    while True:
        try:
            lengths = await message_bus.stream_lengths()
            for stream, length in lengths.items():
                stream_lag_gauge.labels(stream).set(length)
        except Exception:
            logger.exception("[obs] stream lag monitor error")
        await asyncio.sleep(interval)


# ------------------------------------------------------------------ #
# Convenience initialiser                                            #
# ------------------------------------------------------------------ #

def setup_observability(
    otlp_endpoint: str = "http://localhost:4317",
    prometheus_port: int = 9090,
    log_level: str = "INFO",
) -> None:
    configure_logging(log_level)
    init_tracing(otlp_endpoint)
    start_metrics_server(prometheus_port)
    logger.info("[obs] observability stack initialised")
