"""
Summarizer Agent — synthesises research and critique into an executive summary.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from src.runtime.agents._base import get_llm, get_state_value

METADATA = {
    "name": "summarizer",
    "label": "Summarizer",
    "emoji": "📝",
    "role": "Executive Summarizer",
    "output_key": "summary",
}


def node_fn(state: dict) -> dict:
    llm = get_llm()
    goal = get_state_value(state, "goal", "")
    research = get_state_value(state, "research", "")
    critique = get_state_value(state, "critique", "")
    fact_check = get_state_value(state, "fact_check", "")
    aggregated = get_state_value(state, "aggregated", "")

    # Use aggregated output if available (from parallel stages), else raw research+critique
    context = aggregated or f"Research:\n{research}\n\nCritique:\n{critique}"
    if fact_check:
        context += f"\n\nFact-Check:\n{fact_check}"

    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert at synthesising complex multi-source information "
            "into concise, actionable executive summaries."
        )),
        HumanMessage(content=(
            f"Goal: {goal}\n\n{context}\n\n"
            "Write a concise executive summary (2–3 paragraphs) that integrates all perspectives, "
            "addresses the key weaknesses identified, and provides clear actionable conclusions."
        )),
    ])
    return {**state, "summary": response.content, "logs": [f"[Summarizer] Summary complete ({len(response.content)} chars)."]}
