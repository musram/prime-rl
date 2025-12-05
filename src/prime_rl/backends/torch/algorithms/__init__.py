"""
PyTorch algorithm implementations for online RL.
"""

from prime_rl.backends.torch.algorithms.ppo import TorchPPO
from prime_rl.backends.torch.algorithms.grpo import TorchGRPO

__all__ = [
    "TorchPPO",
    "TorchGRPO",
]

