"""
Agent Lifecycle Manager
Coordinates agent registration, status transitions, retry logic, and topology storage.
"""
import json
import logging
import uuid
from datetime import datetime

from sqlalchemy import select

from src.core.db import AgentRecord, WorkflowRun, AgentStatus, get_session
from src.core.message_bus import MessageBus

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
DEFAULT_TOPOLOGY = [["researcher"], ["critic"], ["summarizer"]]


class LifecycleManager:
    def __init__(self, bus: MessageBus):
        self.bus = bus

    # ── Workflow helpers ──────────────────────────────────────────────────
    async def create_workflow(
        self,
        name: str,
        goal: str,
        topology: list[list[str]] | None = None,
    ) -> str:
        resolved_topology = topology or DEFAULT_TOPOLOGY
        async with get_session() as session:
            run = WorkflowRun(
                id=str(uuid.uuid4()),
                name=name,
                goal=goal,
                status=AgentStatus.PENDING,
                topology=json.dumps(resolved_topology),
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            run_id = run.id

        await self.bus.publish({
            "type": "workflow",
            "workflow_id": run_id,
            "name": name,
            "goal": goal,
            "topology": json.dumps(resolved_topology),
        })
        await self.bus.broadcast_event("agent_events", {
            "event": "workflow_created",
            "workflow_id": run_id,
            "name": name,
            "goal": goal,
            "status": AgentStatus.PENDING,
            "topology": resolved_topology,
        })
        logger.info("Created workflow %s (%s) with topology %s", run_id, name, resolved_topology)
        return run_id

    async def get_workflow(self, workflow_id: str) -> WorkflowRun | None:
        async with get_session() as session:
            result = await session.execute(
                select(WorkflowRun).where(WorkflowRun.id == workflow_id)
            )
            return result.scalar_one_or_none()

    async def list_workflows(self) -> list[WorkflowRun]:
        async with get_session() as session:
            result = await session.execute(
                select(WorkflowRun).order_by(WorkflowRun.created_at.desc())
            )
            return list(result.scalars().all())

    async def retry_workflow(self, workflow_id: str) -> bool:
        """Re-dispatch a failed (or retrying) workflow with its original topology."""
        wf = await self.get_workflow(workflow_id)
        if not wf:
            return False
        if wf.status not in (AgentStatus.FAILED, AgentStatus.RETRYING):
            return False

        topology = wf.get_topology() or DEFAULT_TOPOLOGY

        # Reset workflow status
        await self.update_workflow_status(workflow_id, AgentStatus.PENDING)

        # Re-publish to message bus
        await self.bus.publish({
            "type": "workflow",
            "workflow_id": workflow_id,
            "name": wf.name,
            "goal": wf.goal,
            "topology": json.dumps(topology),
            "is_retry": True,
        })
        await self.bus.broadcast_event("agent_events", {
            "event": "workflow_retried",
            "workflow_id": workflow_id,
            "name": wf.name,
            "topology": topology,
        })
        logger.info("Retrying workflow %s", workflow_id)
        return True

    # ── Agent helpers ─────────────────────────────────────────────────────
    async def spawn_agent(self, workflow_id: str, agent_name: str, role: str) -> AgentRecord:
        async with get_session() as session:
            agent = AgentRecord(
                id=str(uuid.uuid4()),
                workflow_id=workflow_id,
                agent_name=agent_name,
                role=role,
                status=AgentStatus.PENDING,
            )
            session.add(agent)
            await session.commit()
            await session.refresh(agent)
            agent_id = agent.id

        await self.bus.broadcast_event("agent_events", {
            "event": "agent_spawned",
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "role": role,
            "status": AgentStatus.PENDING,
        })
        return agent

    async def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        output: str | None = None,
        error: str | None = None,
        increment_retry: bool = False,
    ) -> None:
        async with get_session() as session:
            result = await session.execute(select(AgentRecord).where(AgentRecord.id == agent_id))
            agent = result.scalar_one_or_none()
            if not agent:
                return
            agent.status = status
            if status == AgentStatus.RUNNING:
                agent.started_at = datetime.utcnow()
            if status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
                agent.finished_at = datetime.utcnow()
            if increment_retry:
                agent.retry_count = (agent.retry_count or 0) + 1
            if output is not None:
                agent.output = output
            if error:
                agent.error = error
            await session.commit()

        await self.bus.broadcast_event("agent_events", {
            "event": "agent_status_change",
            "agent_id": agent_id,
            "status": status,
            "output": output,
            "error": error,
        })

    async def update_workflow_status(
        self,
        workflow_id: str,
        status: AgentStatus,
        final_output: str | None = None,
    ) -> None:
        async with get_session() as session:
            result = await session.execute(select(WorkflowRun).where(WorkflowRun.id == workflow_id))
            wf = result.scalar_one_or_none()
            if not wf:
                return
            wf.status = status
            wf.updated_at = datetime.utcnow()
            if final_output:
                wf.final_output = final_output
            await session.commit()

        await self.bus.broadcast_event("agent_events", {
            "event": "workflow_status_change",
            "workflow_id": workflow_id,
            "status": status,
            "final_output": final_output,
        })

    async def list_agents(self, workflow_id: str) -> list[AgentRecord]:
        async with get_session() as session:
            result = await session.execute(
                select(AgentRecord).where(AgentRecord.workflow_id == workflow_id)
            )
            return list(result.scalars().all())
