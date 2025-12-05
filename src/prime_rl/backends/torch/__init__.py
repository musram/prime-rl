"""
PyTorch backend for PRIME-RL online RL engine.

This package implements the PyTorch-based online RL engine as specified in the PRD (§2.2, §3.4).
It provides online RL training with environment interaction and scalable orchestration.
"""

from prime_rl.backends.torch.trainer import TorchTrainer
from prime_rl.backends.torch.algorithms.ppo import TorchPPO
from prime_rl.backends.torch.algorithms.grpo import TorchGRPO

__all__ = [
    "TorchTrainer",
    "TorchPPO",
    "TorchGRPO",
]

