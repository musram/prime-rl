"""
PyTorch trainer for online RL.

This module implements the TorchTrainer class that manages PyTorch-based online RL
training loops with environment interaction and scalable orchestration.
"""

from typing import Any, Dict, Optional
import torch

from prime_rl.core.trainer import Trainer
from prime_rl.core.algorithms import TrainingMetrics


class TorchTrainer(Trainer):
    """
    PyTorch-based trainer for online RL.
    
    Manages the training loop with PyTorch, integrating with EnvironmentAdapter
    instances and remote VerifierClient for reward signals.
    
    This implements the Trainer interface for the PyTorch backend.
    """

    def __init__(
        self,
        algorithm: Any,  # TorchPPO, TorchGRPO, or other TorchAlgorithm
        config: Dict[str, Any],
    ):
        """
        Initialize PyTorch trainer.
        
        Args:
            algorithm: PyTorch algorithm instance (e.g., TorchPPO, TorchGRPO)
            config: Training configuration dictionary
        """
        self.algorithm = algorithm
        self.config = config
        self.step = 0
        self.model = None
        self.optimizer = None

    def run_training_loop(self) -> None:
        """
        Run the main PyTorch training loop.
        
        This integrates with:
        - EnvironmentAdapter instances for environment interaction
        - VerifierClient for remote reward signals
        - Orchestrator for distributed rollout collection
        
        This is a stub implementation for Phase 2. Full implementation will:
        - Initialize model and optimizer
        - Collect rollouts from environments
        - Compute advantages
        - Call algorithm.train_step()
        - Handle checkpointing and logging
        """
        # Stub: Initialize model/optimizer would go here
        print(f"TorchTrainer: Training loop started (stub implementation)")
        print(f"Algorithm: {type(self.algorithm).__name__}")
        print(f"Config: {self.config}")

    def checkpoint(self, step: int, metrics: Dict[str, Any]) -> None:
        """Save a checkpoint (stub implementation)."""
        print(f"TorchTrainer: Checkpoint saved at step {step}")

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log training metrics (stub implementation)."""
        print(f"TorchTrainer: Step {step}, Metrics: {metrics}")

    def evaluate(self, step: int) -> Optional[Dict[str, Any]]:
        """Run evaluation (stub implementation)."""
        return None

