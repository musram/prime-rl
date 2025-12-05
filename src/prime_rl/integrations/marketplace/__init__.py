"""
Data marketplace integrations for PRIME-RL.

This module provides clients for data marketplaces (Surge AI, Mercor)
to ingest labeled data and expert demonstrations as Interaction Traces.
"""

from prime_rl.integrations.marketplace.surge import SurgeAIClient
from prime_rl.integrations.marketplace.mercor import MercorClient

__all__ = [
    "SurgeAIClient",
    "MercorClient",
]

