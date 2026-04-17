"""
Dynamic Multi-Agent LangGraph Builder.

Topology format — a list of stages, where each stage is a list of agent names.
Agents within the same stage run concurrently (ThreadPoolExecutor).

Examples:
  Sequential:  [["researcher"], ["critic"], ["summarizer"]]
  Branching:   [["researcher"], ["critic", "fact_checker"], ["aggregator"]]
"""
import concurrent.futures
import hashlib
import json
import logging
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

DEFAULT_TOPOLOGY: list[list[str]] = [["researcher"], ["critic"], ["summarizer"]]

# ── Shared State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    goal: str
    research: str
    critique: str
    fact_check: str
    aggregated: str
    summary: str
    logs: Annotated[list[str], operator.add]


def _empty_state(goal: str) -> AgentState:
    return {
        "goal": goal,
        "research": "",
        "critique": "",
        "fact_check": "",
        "aggregated": "",
        "summary": "",
        "logs": [],
    }


# ── Graph cache (topology_hash → compiled graph) ──────────────────────────────
_graph_cache: dict[str, object] = {}


def _topology_hash(topology: list[list[str]]) -> str:
    return hashlib.md5(json.dumps(topology, sort_keys=False).encode()).hexdigest()[:12]


def _make_parallel_node(agent_names: list[str]):
    """
    Returns a sync LangGraph node function that runs named agents concurrently.
    The import of AGENT_REGISTRY is deferred to execution time to avoid
    circular-import / empty-registry problems at module load.
    """
    def parallel_node(state: dict) -> dict:
        # Lazy import so registry is definitely populated by the time nodes run
        from src.runtime.agents import AGENT_REGISTRY

        plugins = [AGENT_REGISTRY[n] for n in agent_names if n in AGENT_REGISTRY]
        if not plugins:
            logger.warning("No registered plugins found for agents: %s", agent_names)
            return {"logs": [f"[Warning] No plugins found for {agent_names}"]}

        if len(plugins) == 1:
            # Single agent — call directly; return only the diff (LangGraph merges)
            result = plugins[0]["node_fn"](state)
            # Return only the keys that changed (safe approach: return full result)
            return result

        # Multiple agents — run concurrently in a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(plugins)) as pool:
            futures = [pool.submit(p["node_fn"], state) for p in plugins]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Merge results: accumulate all new values; logs are collected separately
        # (LangGraph's operator.add reducer will append them to the existing list)
        merged: dict = {}
        all_new_logs: list[str] = []
        for r in results:
            for k, v in r.items():
                if k == "logs":
                    all_new_logs.extend(v if isinstance(v, list) else [v])
                elif k != "goal":          # never overwrite the goal
                    merged[k] = v         # last writer wins per key
        merged["logs"] = all_new_logs     # LangGraph reducer appends this
        return merged

    parallel_node.__name__ = f"stage_{'__'.join(agent_names)}"
    return parallel_node


def build_dynamic_graph(topology: list[list[str]]) -> object:
    """
    Build (or retrieve from cache) a compiled LangGraph for the given topology.
    Each element of topology is a list of agent names run in parallel within a stage.
    """
    key = _topology_hash(topology)
    if key in _graph_cache:
        logger.debug("Graph cache hit for topology %s", topology)
        return _graph_cache[key]

    logger.info("Building new graph for topology: %s", topology)
    g = StateGraph(AgentState)

    stage_node_names: list[str] = []
    for i, stage in enumerate(topology):
        if not stage:
            continue
        node_name = f"stage_{i}"
        node_fn = _make_parallel_node(stage)
        g.add_node(node_name, node_fn)
        stage_node_names.append(node_name)

    if not stage_node_names:
        raise ValueError("Topology produced no nodes — check agent names are registered.")

    g.set_entry_point(stage_node_names[0])
    for a, b in zip(stage_node_names, stage_node_names[1:]):
        g.add_edge(a, b)
    g.add_edge(stage_node_names[-1], END)

    compiled = g.compile()
    _graph_cache[key] = compiled
    logger.info("Graph compiled and cached (key=%s, stages=%d)", key, len(stage_node_names))
    return compiled


# Keep for backward compatibility (built lazily on first call to avoid import-time issues)
COMPILED_GRAPH = None  # populated on first use via build_dynamic_graph(DEFAULT_TOPOLOGY)
