"""
Prime Intellect integrations for PRIME-RL.

This module provides adapters for Prime Intellect Dashboard environments
and Verifier APIs, enabling integration with the Prime Intellect ecosystem.
"""

from prime_rl.integrations.prime_intellect.adapter import PrimeIntellectEnvAdapter
from prime_rl.integrations.prime_intellect.client import PrimeIntellectVerifierClient

__all__ = [
    "PrimeIntellectEnvAdapter",
    "PrimeIntellectVerifierClient",
]

