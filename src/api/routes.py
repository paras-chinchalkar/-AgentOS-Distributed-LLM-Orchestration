"""
REST API routes for the Agent Orchestration Runtime.
"""
import uuid
from typing import Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.core.db import AgentStatus
from src.runtime.lifecycle import LifecycleManager

router = APIRouter(prefix="/api/v1", tags=["orchestration"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class CreateWorkflowRequest(BaseModel):
    name: str = Field(..., example="Market Analysis Run")
    goal: str = Field(..., example="Research the current state of AI chip manufacturing.")
    topology: list[list[str]] | None = Field(
        default=None,
        example=[["researcher"], ["critic", "fact_checker"], ["aggregator"], ["summarizer"]],
        description=(
            "Optional DAG topology as a list of parallel stages. "
            "Each stage is a list of agent names that run concurrently. "
            "Defaults to the classic sequential pipeline."
        ),
    )


class AgentOut(BaseModel):
    id: str
    agent_name: str
    role: str
    status: str
    output: str | None
    error: str | None
    retry_count: int

    class Config:
        from_attributes = True


class WorkflowOut(BaseModel):
    id: str
    name: str
    goal: str
    status: str
    final_output: str | None
    topology: str | None
    created_at: Any
    updated_at: Any

    class Config:
        from_attributes = True


class AgentPluginOut(BaseModel):
    name: str
    label: str
    emoji: str
    role: str
    output_key: str


# ── Dependency injection helper ───────────────────────────────────────────────
def get_manager() -> LifecycleManager:
    from src.api.main import lifecycle_manager
    if lifecycle_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Runtime not ready — lifecycle manager is not initialised yet. Check server logs.",
        )
    return lifecycle_manager


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/workflows", response_model=dict, status_code=202)
async def create_workflow(
    body: CreateWorkflowRequest,
    manager: LifecycleManager = Depends(get_manager),
):
    """Spawn a new multi-agent workflow with an optional dynamic topology."""
    workflow_id = await manager.create_workflow(
        name=body.name,
        goal=body.goal,
        topology=body.topology,
    )
    return {"workflow_id": workflow_id, "status": AgentStatus.PENDING}


@router.get("/workflows", response_model=list[WorkflowOut])
async def list_workflows(manager: LifecycleManager = Depends(get_manager)):
    """List all workflow runs."""
    workflows = await manager.list_workflows()
    return [WorkflowOut.model_validate(w) for w in workflows]


@router.get("/workflows/{workflow_id}", response_model=WorkflowOut)
async def get_workflow(workflow_id: str, manager: LifecycleManager = Depends(get_manager)):
    """Get details of a specific workflow."""
    wf = await manager.get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return WorkflowOut.model_validate(wf)


@router.get("/workflows/{workflow_id}/agents", response_model=list[AgentOut])
async def list_workflow_agents(workflow_id: str, manager: LifecycleManager = Depends(get_manager)):
    """List all agent records for a given workflow."""
    agents = await manager.list_agents(workflow_id)
    return [AgentOut.model_validate(a) for a in agents]


@router.post("/workflows/{workflow_id}/retry", response_model=dict)
async def retry_workflow(workflow_id: str, manager: LifecycleManager = Depends(get_manager)):
    """Re-dispatch a failed workflow using its original topology."""
    ok = await manager.retry_workflow(workflow_id)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="Workflow cannot be retried — it must be in 'failed' or 'retrying' status.",
        )
    return {"workflow_id": workflow_id, "status": "retrying"}


@router.get("/agents", response_model=list[AgentPluginOut])
async def list_agents():
    """List all registered agent plugins."""
    from src.runtime.agents import get_agents_list
    return get_agents_list()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "agent-orchestration-runtime"}
