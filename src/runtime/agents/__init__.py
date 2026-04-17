"""
Agent Plugin Registry — scans the agents/ directory and builds AGENT_REGISTRY.

Each agent module must export:
    METADATA: dict   — name, label, emoji, role, output_key
    node_fn(state)   — synchronous LangGraph node function
"""
import importlib
import logging
from pathlib import Path
from typing import Callable, TypedDict

logger = logging.getLogger(__name__)


class AgentPlugin(TypedDict):
    name: str        # snake_case key, e.g. "fact_checker"
    label: str       # Display name, e.g. "Fact Checker"
    emoji: str       # e.g. "🔎"
    role: str        # e.g. "Verification Specialist"
    output_key: str  # key written into AgentState
    node_fn: Callable


# ── Auto-discover all agent modules ──────────────────────────────────────────
AGENT_REGISTRY: dict[str, AgentPlugin] = {}

_here = Path(__file__).parent
_skip = {"__init__"}

for _path in sorted(_here.glob("*.py")):
    _mod_name = _path.stem
    if _mod_name in _skip:
        continue
    try:
        _mod = importlib.import_module(f"src.runtime.agents.{_mod_name}")
        if hasattr(_mod, "METADATA") and hasattr(_mod, "node_fn"):
            meta: dict = _mod.METADATA
            AGENT_REGISTRY[meta["name"]] = AgentPlugin(
                name=meta["name"],
                label=meta["label"],
                emoji=meta["emoji"],
                role=meta["role"],
                output_key=meta["output_key"],
                node_fn=_mod.node_fn,
            )
            logger.info("Registered agent plugin: %s", meta["name"])
        else:
            logger.warning("Agent module %s missing METADATA or node_fn — skipped.", _mod_name)
    except Exception as exc:
        logger.error("Failed to load agent module %s: %s", _mod_name, exc)

logger.info("Agent registry loaded: %s", list(AGENT_REGISTRY.keys()))


def get_agents_list() -> list[dict]:
    """Return agent metadata (without node_fn) for API serialisation."""
    return [
        {k: v for k, v in plugin.items() if k != "node_fn"}
        for plugin in AGENT_REGISTRY.values()
    ]
