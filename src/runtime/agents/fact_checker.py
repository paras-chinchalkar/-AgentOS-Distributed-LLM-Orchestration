"""
FactChecker Agent — verifies claims in the research and flags unverified statements.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from src.runtime.agents._base import get_llm, get_state_value

METADATA = {
    "name": "fact_checker",
    "label": "Fact Checker",
    "emoji": "🔎",
    "role": "Verification Specialist",
    "output_key": "fact_check",
}


def node_fn(state: dict) -> dict:
    llm = get_llm()
    goal = get_state_value(state, "goal", "")
    research = get_state_value(state, "research", "(no research provided)")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a meticulous fact-checker and verification specialist. "
            "Your job is to scrutinize claims, identify what can be independently verified, "
            "flag speculative statements, and rate the overall reliability of the information."
        )),
        HumanMessage(content=(
            f"Goal: {goal}\n\nResearch to verify:\n{research}\n\n"
            "1. List the top 5 key claims made.\n"
            "2. For each claim, indicate: [VERIFIED] [UNVERIFIED] [SPECULATIVE] with a brief reason.\n"
            "3. Give an overall reliability score (1-10) for the research.\n"
            "4. List any critical missing facts."
        )),
    ])
    return {**state, "fact_check": response.content, "logs": [f"[FactChecker] Verification complete ({len(response.content)} chars)."]}
