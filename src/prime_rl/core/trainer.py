"""
Trainer interface for RL training loops.

This module defines the abstract Trainer interface that all backend-specific trainers
must implement. It provides hooks for logging, checkpointing, and evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol


class Trainer(ABC):
    """
    Abstract base class for RL trainers.
    
    All backend-specific trainers (JaxTrainer, TorchTrainer) must implement this interface.
    This ensures a consistent API across backends while allowing backend-specific optimizations.
    """

    @abstractmethod
    def run_training_loop(self) -> None:
        """
        Run the main training loop.
        
        This method should:
        - Load/initialize the model and optimizer
        - Iterate over training batches
        - Call algorithm.train_step() for each batch
        - Handle checkpointing, logging, and evaluation hooks
        - Clean up resources on completion
        """
        pass

    @abstractmethod
    def checkpoint(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Save a checkpoint at the given step.
        
        Args:
            step: Current training step
            metrics: Current training metrics
        """
        pass

    @abstractmethod
    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Log training metrics (e.g., to W&B, TensorBoard, console).
        
        Args:
            step: Current training step
            metrics: Metrics to log
        """
        pass

    @abstractmethod
    def evaluate(self, step: int) -> Optional[Dict[str, Any]]:
        """
        Run evaluation on the current model.
        
        Args:
            step: Current training step
            
        Returns:
            Evaluation metrics dictionary, or None if evaluation is not configured
        """
        pass


class CheckpointHook(Protocol):
    """Protocol for checkpoint hooks that can be registered with trainers."""
    
    def __call__(self, step: int, metrics: Dict[str, Any], trainer: Trainer) -> None:
        """Called when a checkpoint should be saved."""
        ...


class LoggingHook(Protocol):
    """Protocol for logging hooks that can be registered with trainers."""
    
    def __call__(self, step: int, metrics: Dict[str, Any], trainer: Trainer) -> None:
        """Called when metrics should be logged."""
        ...


class EvaluationHook(Protocol):
    """Protocol for evaluation hooks that can be registered with trainers."""
    
    def __call__(self, step: int, trainer: Trainer) -> Optional[Dict[str, Any]]:
        """Called when evaluation should run. Returns evaluation metrics or None."""
        ...

