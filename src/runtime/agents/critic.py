"""
Critic Agent — identifies weaknesses, biases, and gaps in the research.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from src.runtime.agents._base import get_llm, get_state_value

METADATA = {
    "name": "critic",
    "label": "Critic",
    "emoji": "🧐",
    "role": "Critical Analyst",
    "output_key": "critique",
}


def node_fn(state: dict) -> dict:
    llm = get_llm()
    goal = get_state_value(state, "goal", "")
    research = get_state_value(state, "research", "(no research provided)")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a sharp analytical critic. "
            "Identify weaknesses, logical gaps, biases, and unstated assumptions."
        )),
        HumanMessage(content=(
            f"Goal: {goal}\n\nResearch:\n{research}\n\n"
            "Provide a thorough critical analysis, highlight at least 3 specific weaknesses "
            "or blind spots, and suggest what additional research would strengthen the findings."
        )),
    ])
    return {**state, "critique": response.content, "logs": [f"[Critic] Critique complete ({len(response.content)} chars)."]}
