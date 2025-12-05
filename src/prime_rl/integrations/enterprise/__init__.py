"""
Enterprise integrations for PRIME-RL.

This module provides adapters for enterprise SaaS tools and systems,
enabling RL training on real-world enterprise workflows.
"""

from prime_rl.integrations.enterprise.epic import EpicAdapter
from prime_rl.integrations.enterprise.slack import SlackAdapter

__all__ = [
    "EpicAdapter",
    "SlackAdapter",
]

