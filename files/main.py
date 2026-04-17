"""
main.py — Orchestrator entry point.
Wires all modules together and exposes the REST API.

Start:
    uvicorn ai_agent_os.api.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core.agent_core import AgentConfig, AgentRuntime, GLOBAL_REGISTRY
from ..core.message_bus import MessageBus
from ..core.scheduler import Scheduler, Task
from ..core.state_store import StateStore
from ..observability.observability import setup_observability, ExecutionTracker, monitor_stream_lag
from ..workflows.workflow_graph import run_workflow

logger = logging.getLogger("agent_os.api")

# ------------------------------------------------------------------ #
# Singletons (injected via lifespan)                                 #
# ------------------------------------------------------------------ #

bus: Optional[MessageBus] = None
store: Optional[StateStore] = None
scheduler: Optional[Scheduler] = None


class AgentPool:
    """Minimal dynamic agent pool — spawns workers on demand."""

    def __init__(self) -> None:
        self._free: Dict[str, List[str]] = {}

    def provision(self, agent_type: str, count: int) -> None:
        from uuid import uuid4
        self._free.setdefault(agent_type, [])
        for _ in range(count):
            self._free[agent_type].append(f"{agent_type}-{uuid4().hex[:6]}")

    async def acquire(self, agent_type: str = "worker") -> str:
        pool = self._free.get(agent_type, [])
        if pool:
            return pool.pop(0)
        # Dynamically spawn
        from uuid import uuid4
        new_id = f"{agent_type}-{uuid4().hex[:6]}"
        logger.info("[pool] spawned new agent %s", new_id)
        return new_id


pool = AgentPool()


# ------------------------------------------------------------------ #
# Lifespan                                                           #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bus, store, scheduler

    setup_observability(
        otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://localhost:4317"),
        prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    bus = MessageBus(os.getenv("REDIS_URL", "redis://localhost:6379"))
    await bus.connect()

    store = StateStore(os.getenv("DATABASE_URL", "postgresql://localhost/agent_os"))
    await store.connect()

    pool.provision("worker", 4)
    pool.provision("planner", 1)
    pool.provision("validator", 1)

    scheduler = Scheduler(bus, store, pool)

    dispatch_task = asyncio.create_task(scheduler.run_dispatch_loop())
    lag_task = asyncio.create_task(monitor_stream_lag(bus))

    logger.info("Agent Orchestration OS online")
    yield

    await scheduler.stop()
    dispatch_task.cancel()
    lag_task.cancel()
    await bus.close()
    await store.close()
    logger.info("Agent Orchestration OS shut down")


# ------------------------------------------------------------------ #
# FastAPI app                                                        #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="AI Agent Orchestration OS",
    description="Distributed multi-agent task orchestration runtime",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Request / Response models                                          #
# ------------------------------------------------------------------ #

class SubmitTaskRequest(BaseModel):
    goal: str = Field(..., min_length=5, description="High-level task goal")
    agent_type: str = Field("worker", description="Target agent type")
    priority: int = Field(5, ge=1, le=10)
    max_retries: int = Field(3, ge=0, le=10)


class WorkflowRequest(BaseModel):
    goal: str = Field(..., min_length=5)


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class WorkflowResponse(BaseModel):
    workflow_id: str
    goal: str
    subtask_count: int
    result: Optional[str]
    error: Optional[str]


# ------------------------------------------------------------------ #
# Endpoints                                                          #
# ------------------------------------------------------------------ #

@app.post("/api/v1/task", response_model=TaskResponse, tags=["Tasks"])
async def submit_task(req: SubmitTaskRequest, bg: BackgroundTasks) -> TaskResponse:
    """Submit a single task to the scheduler."""
    task = Task(
        goal=req.goal,
        agent_type=req.agent_type,
        priority=req.priority,
        max_retries=req.max_retries,
    )
    task_id = await scheduler.submit(task)
    logger.info("[api] task submitted: %s", task_id)
    return TaskResponse(task_id=task_id, status="queued", message="Task accepted")


@app.post("/api/v1/workflow", response_model=WorkflowResponse, tags=["Workflows"])
async def submit_workflow(req: WorkflowRequest) -> WorkflowResponse:
    """
    Submit a high-level goal.
    The Planner Agent decomposes it into a DAG; the Scheduler executes it.
    """
    result = await run_workflow(req.goal)
    return WorkflowResponse(**result)


@app.get("/api/v1/task/{task_id}", tags=["Tasks"])
async def get_task_result(task_id: str) -> Dict[str, Any]:
    result = await store.get_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


@app.get("/api/v1/scheduler/snapshot", tags=["Observability"])
async def scheduler_snapshot() -> Dict[str, Any]:
    return scheduler.snapshot()


@app.get("/api/v1/streams", tags=["Observability"])
async def stream_lengths() -> Dict[str, int]:
    return await bus.stream_lengths()


@app.get("/api/v1/stats", tags=["Observability"])
async def db_stats() -> Dict[str, Any]:
    return await store.stats()


@app.get("/health", tags=["System"])
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "agent-orchestration-os"}
