"""
workflow_graph.py — LangGraph-based DAG workflow compiler.
Converts a high-level goal into a typed StateGraph with
dependency edges, conditional routing, and acyclicity enforcement.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from .scheduler import Task

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Shared workflow state schema (passed between all graph nodes)       #
# ------------------------------------------------------------------ #

class WorkflowState(TypedDict):
    goal: str
    workflow_id: str
    subtasks: List[Dict[str, Any]]     # decomposed task specs
    task_results: Dict[str, Any]       # task_id → result
    final_result: Optional[str]
    error: Optional[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ------------------------------------------------------------------ #
# Node implementations                                                #
# ------------------------------------------------------------------ #

async def planner_node(state: WorkflowState) -> WorkflowState:
    """
    Decomposes the top-level goal into a list of subtask specs.
    In production this calls the Planner Agent via the message bus.
    The subtask list forms the node set of the execution DAG.
    """
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()

    prompt = f"""
You are a task planner for a distributed AI agent system.
Decompose the following goal into 3-6 concrete subtasks.

Goal: {state['goal']}

Return a JSON array of subtask objects, each with:
- "task_id": unique string id
- "name": short name
- "description": what the agent should do
- "agent_type": one of ["planner","worker","researcher","validator","writer","aggregator"]
- "upstream": list of task_ids that must complete before this task starts
- "priority": integer 1-10 (lower = higher priority)

Ensure the dependency graph is acyclic (DAG).
Return ONLY the JSON array, no explanation.
"""
    response = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    import json
    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    subtasks = json.loads(raw.strip())

    # Validate DAG acyclicity before accepting
    _assert_acyclic(subtasks)

    logger.info("[workflow] planned %d subtasks for workflow %s", len(subtasks), state["workflow_id"])
    return {**state, "subtasks": subtasks}


async def dispatch_node(state: WorkflowState) -> WorkflowState:
    """
    Converts subtask specs into Task objects and submits them to
    the scheduler. Dependency edges are preserved.
    """
    # In production: inject scheduler via closure or context
    logger.info("[workflow] dispatch_node — %d subtasks ready to submit", len(state["subtasks"]))
    return state


async def aggregator_node(state: WorkflowState) -> WorkflowState:
    """
    Waits for all subtask results, then synthesises the final answer.
    """
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()

    results_block = "\n".join(
        f"- {tid}: {res}" for tid, res in state["task_results"].items()
    )
    response = await client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Original goal: {state['goal']}\n\n"
                f"Subtask results:\n{results_block}\n\n"
                "Synthesise a single coherent final answer from these results."
            ),
        }],
    )
    return {**state, "final_result": response.content[0].text}


async def error_handler_node(state: WorkflowState) -> WorkflowState:
    logger.error("[workflow] error in workflow %s: %s", state["workflow_id"], state.get("error"))
    return state


# ------------------------------------------------------------------ #
# Conditional routing                                                 #
# ------------------------------------------------------------------ #

def route_after_plan(state: WorkflowState) -> str:
    if not state.get("subtasks"):
        return "error_handler"
    return "dispatch"


def route_after_dispatch(state: WorkflowState) -> str:
    pending = [t for t in state["subtasks"] if t["task_id"] not in state["task_results"]]
    if pending:
        return "dispatch"        # loop until all results are in
    return "aggregator"


# ------------------------------------------------------------------ #
# Graph builder                                                       #
# ------------------------------------------------------------------ #

def build_workflow_graph() -> StateGraph:
    """
    Compiles the LangGraph StateGraph.
    The graph is acyclic by construction — the only back-edge is
    the dispatch loop which terminates once all subtasks are done.
    """
    graph = StateGraph(WorkflowState)

    graph.add_node("planner",       planner_node)
    graph.add_node("dispatch",      dispatch_node)
    graph.add_node("aggregator",    aggregator_node)
    graph.add_node("error_handler", error_handler_node)

    graph.add_edge(START, "planner")

    graph.add_conditional_edges("planner", route_after_plan, {
        "dispatch":      "dispatch",
        "error_handler": "error_handler",
    })
    graph.add_conditional_edges("dispatch", route_after_dispatch, {
        "dispatch":   "dispatch",
        "aggregator": "aggregator",
    })
    graph.add_edge("aggregator",    END)
    graph.add_edge("error_handler", END)

    return graph.compile()


# ------------------------------------------------------------------ #
# Acyclicity validation (Kahn's algorithm)                           #
# ------------------------------------------------------------------ #

def _assert_acyclic(subtasks: List[Dict]) -> None:
    """
    Raises ValueError if the subtask dependency graph contains a cycle.
    O(V + E).
    """
    ids = {t["task_id"] for t in subtasks}
    in_degree: Dict[str, int] = {t["task_id"]: 0 for t in subtasks}
    adj: Dict[str, List[str]] = {t["task_id"]: [] for t in subtasks}

    for t in subtasks:
        for dep in t.get("upstream", []):
            if dep not in ids:
                raise ValueError(f"Unknown upstream dependency: {dep}")
            adj[dep].append(t["task_id"])
            in_degree[t["task_id"]] += 1

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for neighbour in adj[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if visited != len(subtasks):
        raise ValueError("Cycle detected in subtask dependency graph — not a valid DAG")


# ------------------------------------------------------------------ #
# Runtime entry point                                                 #
# ------------------------------------------------------------------ #

async def run_workflow(goal: str) -> Dict[str, Any]:
    graph = build_workflow_graph()
    workflow_id = str(uuid.uuid4())

    initial_state: WorkflowState = {
        "goal": goal,
        "workflow_id": workflow_id,
        "subtasks": [],
        "task_results": {},
        "final_result": None,
        "error": None,
        "messages": [],
    }

    logger.info("[workflow] starting %s: %s", workflow_id, goal[:80])
    final_state = await graph.ainvoke(initial_state)

    return {
        "workflow_id": workflow_id,
        "goal": goal,
        "subtask_count": len(final_state.get("subtasks", [])),
        "result": final_state.get("final_result"),
        "error": final_state.get("error"),
    }
