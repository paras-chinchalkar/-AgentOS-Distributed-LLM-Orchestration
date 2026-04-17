"""
scheduler.py — DAG-aware task scheduler with a priority heap.
Only dispatches tasks whose upstream dependencies are committed.
Integrates with the message bus and agent pool for dynamic spawning.
"""
from __future__ import annotations

import asyncio
import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNABLE = "runnable"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass(order=True)
class SchedulerEntry:
    priority: int                        # lower = higher priority
    enqueued_at: float
    task_id: str = field(compare=False)


@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    agent_type: str = "worker"
    upstream: List[str] = field(default_factory=list)   # task_ids that must be DONE first
    priority: int = 5                                    # 1 (highest) … 10 (lowest)
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    assigned_agent: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None                    # unix ts; None = no deadline


class Scheduler:
    """
    Priority-heap scheduler that is aware of DAG dependency edges.
    A task becomes RUNNABLE only when all tasks in `upstream` are DONE.
    """

    def __init__(self, message_bus, state_store, agent_pool) -> None:
        self.bus = message_bus
        self.store = state_store
        self.pool = agent_pool

        self._tasks: Dict[str, Task] = {}
        self._heap: List[SchedulerEntry] = []
        self._done_ids: Set[str] = set()
        self._lock = asyncio.Lock()
        self._running = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def submit(self, task: Task) -> str:
        async with self._lock:
            self._tasks[task.task_id] = task
            if self._is_runnable(task):
                self._push(task)
                task.status = TaskStatus.RUNNABLE
                logger.info("[scheduler] task %s immediately runnable", task.task_id)
            else:
                logger.info(
                    "[scheduler] task %s waiting on %s", task.task_id, task.upstream
                )
        return task.task_id

    async def mark_done(self, task_id: str, result: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.DONE
                task.result = result
            self._done_ids.add(task_id)
            await self._unblock_dependents(task_id)

    async def mark_failed(self, task_id: str, error: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                backoff = 2.0 ** task.retry_count
                logger.warning(
                    "[scheduler] task %s failed — retry %d/%d in %.1fs",
                    task_id, task.retry_count, task.max_retries, backoff,
                )
                await asyncio.sleep(backoff)
                task.status = TaskStatus.RUNNABLE
                self._push(task)
                await self.bus.publish("tasks:errors", {
                    "task_id": task_id,
                    "error": error,
                    "retry": task.retry_count,
                })
            else:
                task.status = TaskStatus.DEAD_LETTER
                logger.error("[scheduler] task %s dead-lettered after %d retries", task_id, task.retry_count)
                await self.bus.publish("tasks:dlq", {
                    "task_id": task_id,
                    "goal": task.goal,
                    "error": error,
                })

    async def run_dispatch_loop(self) -> None:
        """
        Continuously pops runnable tasks from the heap and
        dispatches them to agents via the message bus.
        """
        self._running = True
        logger.info("[scheduler] dispatch loop started")

        while self._running:
            task = await self._pop_runnable()
            if task is None:
                await asyncio.sleep(0.1)
                continue

            agent_id = await self.pool.acquire(task.agent_type)
            task.status = TaskStatus.RUNNING
            task.assigned_agent = agent_id

            await self.bus.publish("tasks:queue", {
                "task_id": task.task_id,
                "goal": task.goal,
                "agent_id": agent_id,
                "priority": task.priority,
            })
            logger.info("[scheduler] dispatched task %s → agent %s", task.task_id, agent_id)

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total": len(self._tasks),
            "runnable": sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNABLE),
            "running": sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING),
            "done": len(self._done_ids),
            "dead_letter": sum(1 for t in self._tasks.values() if t.status == TaskStatus.DEAD_LETTER),
        }

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _is_runnable(self, task: Task) -> bool:
        return all(dep in self._done_ids for dep in task.upstream)

    def _push(self, task: Task) -> None:
        heapq.heappush(self._heap, SchedulerEntry(
            priority=task.priority,
            enqueued_at=time.time(),
            task_id=task.task_id,
        ))

    async def _pop_runnable(self) -> Optional[Task]:
        async with self._lock:
            while self._heap:
                entry = heapq.heappop(self._heap)
                task = self._tasks.get(entry.task_id)
                if task and task.status == TaskStatus.RUNNABLE:
                    # Deadline check
                    if task.deadline and time.time() > task.deadline:
                        task.status = TaskStatus.DEAD_LETTER
                        logger.warning("[scheduler] task %s expired past deadline", entry.task_id)
                        continue
                    return task
        return None

    async def _unblock_dependents(self, completed_id: str) -> None:
        for task in self._tasks.values():
            if (
                task.status == TaskStatus.PENDING
                and completed_id in task.upstream
                and self._is_runnable(task)
            ):
                task.status = TaskStatus.RUNNABLE
                self._push(task)
                logger.info("[scheduler] unblocked task %s", task.task_id)
