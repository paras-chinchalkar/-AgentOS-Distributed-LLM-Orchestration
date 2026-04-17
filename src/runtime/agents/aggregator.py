"""
Aggregator Agent — merges outputs from multiple parallel agents into a unified analysis.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from src.runtime.agents._base import get_llm, get_state_value

METADATA = {
    "name": "aggregator",
    "label": "Aggregator",
    "emoji": "🔗",
    "role": "Output Aggregator",
    "output_key": "aggregated",
}


def node_fn(state: dict) -> dict:
    llm = get_llm()
    goal = get_state_value(state, "goal", "")

    parts: list[str] = []
    research = get_state_value(state, "research", "")
    critique = get_state_value(state, "critique", "")
    fact_check = get_state_value(state, "fact_check", "")

    if research:
        parts.append(f"## Research Findings\n{research}")
    if critique:
        parts.append(f"## Critical Analysis\n{critique}")
    if fact_check:
        parts.append(f"## Fact-Check Results\n{fact_check}")

    if not parts:
        return {**state, "aggregated": "No outputs to aggregate.", "logs": ["[Aggregator] No inputs found."]}

    combined = "\n\n---\n\n".join(parts)
    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert analyst who synthesizes multiple analytical perspectives "
            "into a coherent, unified analysis. Preserve important details from each perspective "
            "while eliminating redundancy."
        )),
        HumanMessage(content=(
            f"Goal: {goal}\n\n"
            f"Multiple expert analyses to merge:\n\n{combined}\n\n"
            "Produce a unified synthesis that preserves the key insights from each analysis, "
            "resolves any contradictions, and presents a clear, integrated perspective."
        )),
    ])
    return {**state, "aggregated": response.content, "logs": [f"[Aggregator] Aggregation complete ({len(response.content)} chars)."]}
