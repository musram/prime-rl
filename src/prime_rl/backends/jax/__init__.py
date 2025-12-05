"""
JAX backend for PRIME-RL offline RL engine.

This package implements the JAX-based offline RL engine as specified in the PRD (§2.1, §3.3).
It provides high-throughput batch processing and TPU/GPU scaling for offline RL algorithms.
"""

from prime_rl.backends.jax.trainer import JaxTrainer
from prime_rl.backends.jax.algorithms.dpo import JaxDPO

__all__ = [
    "JaxTrainer",
    "JaxDPO",
]

