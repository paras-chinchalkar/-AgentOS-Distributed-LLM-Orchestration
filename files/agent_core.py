"""
agent_core.py — Base agent runtime with ReAct reasoning loop,
short-term memory, and PostgreSQL-backed checkpointing.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    FAILED = "failed"
    DONE = "done"


@dataclass
class AgentConfig:
    agent_id: str = field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:8]}")
    agent_type: str = "worker"
    model: str = "claude-opus-4-5"
    max_steps: int = 20
    step_timeout: float = 60.0
    memory_window: int = 10          # last N messages kept in context
    retry_attempts: int = 3
    retry_backoff: float = 2.0


@dataclass
class AgentStep:
    step_id: int
    thought: str
    action: Optional[str]
    action_input: Optional[Dict[str, Any]]
    observation: Optional[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentState:
    agent_id: str
    task_id: str
    status: AgentStatus
    goal: str
    steps: List[AgentStep] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    checkpoint_version: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ToolRegistry:
    """Registry of callable tools available to agents."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict] = {}

    def register(self, name: str, schema: Dict) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._tools[name] = fn
            self._schemas[name] = schema
            return fn
        return decorator

    async def invoke(self, name: str, inputs: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        try:
            result = await self._tools[name](**inputs)
            return str(result)
        except Exception as exc:
            logger.exception("Tool %s raised", name)
            return f"Error: {exc}"

    def anthropic_tool_schemas(self) -> List[Dict]:
        return [
            {"name": name, **schema}
            for name, schema in self._schemas.items()
        ]


GLOBAL_REGISTRY = ToolRegistry()


class AgentRuntime:
    """
    Core agent runtime.
    Runs a ReAct loop: Think → Act → Observe → repeat until done.
    Persists state after every step via state_store.
    """

    def __init__(
        self,
        config: AgentConfig,
        state_store,   # injected — see state_store.py
        message_bus,   # injected — see message_bus.py
        tool_registry: ToolRegistry = GLOBAL_REGISTRY,
    ) -> None:
        self.config = config
        self.store = state_store
        self.bus = message_bus
        self.tools = tool_registry
        self.client = AsyncAnthropic()
        self._state: Optional[AgentState] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def run(self, task_id: str, goal: str) -> AgentState:
        """Entry point — resumes from checkpoint if one exists."""
        self._state = await self.store.load_checkpoint(
            self.config.agent_id, task_id
        ) or AgentState(
            agent_id=self.config.agent_id,
            task_id=task_id,
            status=AgentStatus.RUNNING,
            goal=goal,
        )

        logger.info("[%s] starting — goal: %s", self.config.agent_id, goal[:80])
        await self._publish_status(AgentStatus.RUNNING)

        try:
            await self._reasoning_loop()
        except Exception as exc:
            self._state.status = AgentStatus.FAILED
            self._state.error = str(exc)
            logger.exception("[%s] crashed", self.config.agent_id)
            await self._publish_status(AgentStatus.FAILED)
            await self._checkpoint()
            raise

        await self._checkpoint()
        await self._publish_status(self._state.status)
        return self._state

    # ------------------------------------------------------------------ #
    # Internal reasoning loop                                              #
    # ------------------------------------------------------------------ #

    async def _reasoning_loop(self) -> None:
        step_n = len(self._state.steps)

        while step_n < self.config.max_steps:
            messages = self._build_messages()
            try:
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.config.model,
                        max_tokens=2048,
                        system=self._system_prompt(),
                        tools=self.tools.anthropic_tool_schemas(),
                        messages=messages,
                    ),
                    timeout=self.config.step_timeout,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(f"LLM call timed out after {self.config.step_timeout}s")

            step = await self._process_response(response, step_n)
            self._state.steps.append(step)
            step_n += 1

            await self._checkpoint()

            if response.stop_reason == "end_turn" and step.action is None:
                # Agent declared it is done
                self._state.result = step.thought
                self._state.status = AgentStatus.DONE
                return

        # Hit step limit without finishing
        self._state.status = AgentStatus.FAILED
        self._state.error = f"Exceeded max_steps={self.config.max_steps}"

    async def _process_response(self, response, step_n: int) -> AgentStep:
        thought = ""
        action = None
        action_input = None
        observation = None

        for block in response.content:
            if block.type == "text":
                thought = block.text
            elif block.type == "tool_use":
                action = block.name
                action_input = block.input
                observation = await asyncio.wait_for(
                    self.tools.invoke(block.name, block.input),
                    timeout=self.config.step_timeout,
                )

        return AgentStep(
            step_id=step_n,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _system_prompt(self) -> str:
        return (
            "You are an autonomous AI agent operating inside a distributed orchestration system.\n"
            "Your goal is to complete assigned tasks using available tools.\n"
            "Think step-by-step. Use tools when you need external information or actions.\n"
            "When you have fully completed the goal, state the final answer and stop.\n"
            f"Agent ID: {self.config.agent_id} | Task ID: {self._state.task_id}"
        )

    def _build_messages(self) -> List[Dict]:
        messages: List[Dict] = [{"role": "user", "content": f"Goal: {self._state.goal}"}]
        recent = self._state.steps[-self.config.memory_window:]
        for step in recent:
            if step.thought:
                messages.append({"role": "assistant", "content": step.thought})
            if step.action and step.observation:
                messages.append({
                    "role": "user",
                    "content": f"Tool '{step.action}' returned: {step.observation}",
                })
        return messages

    async def _checkpoint(self) -> None:
        self._state.checkpoint_version += 1
        self._state.updated_at = time.time()
        await self.store.save_checkpoint(self._state)

    async def _publish_status(self, status: AgentStatus) -> None:
        await self.bus.publish(
            stream="agents:heartbeat",
            payload={
                "agent_id": self.config.agent_id,
                "task_id": self._state.task_id if self._state else None,
                "status": status.value,
                "timestamp": time.time(),
            },
        )
