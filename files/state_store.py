"""
state_store.py — Async PostgreSQL state store (asyncpg).
Persists agent checkpoints and task results.
Every write is ACID — no state is ever lost.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import asyncpg

from .agent_core import AgentState, AgentStatus, AgentStep

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agent_checkpoints (
    agent_id            TEXT        NOT NULL,
    task_id             TEXT        NOT NULL,
    checkpoint_version  INTEGER     NOT NULL,
    status              TEXT        NOT NULL,
    goal                TEXT        NOT NULL,
    steps               JSONB       NOT NULL DEFAULT '[]',
    result              TEXT,
    error               TEXT,
    created_at          DOUBLE PRECISION NOT NULL,
    updated_at          DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (agent_id, task_id)
);

CREATE TABLE IF NOT EXISTS task_results (
    task_id     TEXT        PRIMARY KEY,
    agent_id    TEXT        NOT NULL,
    result      TEXT,
    error       TEXT,
    status      TEXT        NOT NULL,
    started_at  DOUBLE PRECISION,
    finished_at DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_status ON agent_checkpoints(status);
CREATE INDEX IF NOT EXISTS idx_results_status      ON task_results(status);
"""


class StateStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=5, max_size=20)
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("[state_store] connected — PostgreSQL pool ready")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    async def save_checkpoint(self, state: AgentState) -> None:
        """Upsert the full agent state. Never deletes; always overwrites."""
        steps_json = json.dumps([
            {
                "step_id": s.step_id,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation,
                "timestamp": s.timestamp,
            }
            for s in state.steps
        ])
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_checkpoints
                    (agent_id, task_id, checkpoint_version, status, goal,
                     steps, result, error, created_at, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,$10)
                ON CONFLICT (agent_id, task_id) DO UPDATE SET
                    checkpoint_version = EXCLUDED.checkpoint_version,
                    status             = EXCLUDED.status,
                    steps              = EXCLUDED.steps,
                    result             = EXCLUDED.result,
                    error              = EXCLUDED.error,
                    updated_at         = EXCLUDED.updated_at
                """,
                state.agent_id, state.task_id, state.checkpoint_version,
                state.status.value, state.goal, steps_json,
                state.result, state.error, state.created_at, state.updated_at,
            )
        logger.debug("[state_store] checkpoint saved — %s v%d", state.agent_id, state.checkpoint_version)

    async def load_checkpoint(self, agent_id: str, task_id: str) -> Optional[AgentState]:
        """Returns the latest checkpoint for an agent+task pair, or None."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agent_checkpoints WHERE agent_id=$1 AND task_id=$2",
                agent_id, task_id,
            )
        if row is None:
            return None

        raw_steps = json.loads(row["steps"])
        steps = [
            AgentStep(
                step_id=s["step_id"],
                thought=s["thought"],
                action=s.get("action"),
                action_input=s.get("action_input"),
                observation=s.get("observation"),
                timestamp=s.get("timestamp", 0.0),
            )
            for s in raw_steps
        ]
        return AgentState(
            agent_id=row["agent_id"],
            task_id=row["task_id"],
            status=AgentStatus(row["status"]),
            goal=row["goal"],
            steps=steps,
            result=row["result"],
            error=row["error"],
            checkpoint_version=row["checkpoint_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------ #
    # Task results                                                         #
    # ------------------------------------------------------------------ #

    async def persist_result(
        self,
        task_id: str,
        agent_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
        started_at: Optional[float] = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO task_results
                    (task_id, agent_id, result, error, status, started_at)
                VALUES ($1,$2,$3,$4,$5,$6)
                ON CONFLICT (task_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    result = EXCLUDED.result,
                    error  = EXCLUDED.error,
                    finished_at = EXTRACT(EPOCH FROM NOW())
                """,
                task_id, agent_id, result, error, status, started_at,
            )

    async def get_result(self, task_id: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM task_results WHERE task_id=$1", task_id
            )
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    # Aggregate queries (used by observability layer)                     #
    # ------------------------------------------------------------------ #

    async def stats(self) -> dict:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT status, COUNT(*) AS n FROM task_results GROUP BY status"
            )
        return {r["status"]: r["n"] for r in rows}
