"""
Researcher Agent — gathers background information and key facts on the goal.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from src.runtime.agents._base import get_llm, get_state_value

METADATA = {
    "name": "researcher",
    "label": "Researcher",
    "emoji": "🔍",
    "role": "Research Analyst",
    "output_key": "research",
}


def node_fn(state: dict) -> dict:
    llm = get_llm()
    goal = get_state_value(state, "goal", "")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a world-class research analyst. "
            "Gather facts, data points, and key insights with depth and precision."
        )),
        HumanMessage(content=(
            f"Research topic: {goal}\n\n"
            "Provide a detailed research report (3–5 paragraphs) covering key facts, "
            "recent developments, statistics, and expert perspectives."
        )),
    ])
    return {**state, "research": response.content, "logs": [f"[Researcher] Research complete ({len(response.content)} chars)."]}
