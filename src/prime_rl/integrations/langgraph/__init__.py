"""
LangGraph environment integrations.

This module provides adapters for LangGraph-based environments, allowing
LangGraph state machines to be used as RL environments.
"""

from prime_rl.integrations.langgraph.adapter import LangGraphAdapter

__all__ = [
    "LangGraphAdapter",
]

