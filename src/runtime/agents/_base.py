"""
Shared helpers for all agent plugins.
"""
from langchain_groq import ChatGroq
from src.core.config import settings


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.4,
    )


def get_state_value(state: dict, key: str, default: str = "") -> str:
    """Safely retrieve a value from agent state."""
    val = state.get(key, default)
    return val if isinstance(val, str) else default
