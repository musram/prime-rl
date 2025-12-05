"""
Shared abstractions for RL algorithms and data structures.

This module provides the interface layer that decouples the core logic from specific backends (PyTorch/JAX).
It implements the core abstractions defined in the PRD (§3.1).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Generic, TypeVar

T = TypeVar("T")

# Type aliases for clarity
State = TypeVar("State")
Batch = TypeVar("Batch")
LossDict = Dict[str, Any]


@dataclass
class UniversalRollout:
    """
    Framework-agnostic rollout representation.
    
    Can be converted to PyTorch tensors or JAX arrays by backend-specific adapters.
    Supports both time-major and batch-major layouts for efficient processing.
    
    Attributes:
        prompts: List of prompt strings (one per trajectory)
        completions: List of completion strings (one per trajectory)
        rewards: List of reward values (one per trajectory)
        observations: Optional list of observations (for multi-step trajectories)
        actions: Optional list of actions (for multi-step trajectories)
        dones: Optional list of done flags (for multi-step trajectories)
        metadata: Optional metadata dictionary (logprobs, advantages, trace_ids, etc.)
    """
    prompts: List[str]
    completions: List[str]
    rewards: List[float]
    # Optional fields for multi-step trajectories
    observations: Optional[List[Any]] = None
    actions: Optional[List[Any]] = None
    dones: Optional[List[bool]] = None
    # Optional metadata for debugging or advanced algorithms (e.g. logprobs, advantages, trace_ids)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rollout to dictionary representation."""
        result = {
            "prompts": self.prompts,
            "completions": self.completions,
            "rewards": self.rewards,
            "metadata": self.metadata,
        }
        if self.observations is not None:
            result["observations"] = self.observations
        if self.actions is not None:
            result["actions"] = self.actions
        if self.dones is not None:
            result["dones"] = self.dones
        return result


@dataclass
class TrainingMetrics:
    """
    Common training metrics that every backend must report.
    
    Attributes:
        loss: Total loss value
        grad_norm: Gradient norm (for monitoring training stability)
        learning_rate: Current learning rate
        step: Current training step number
        framework_specific: Backend-specific metrics (e.g. "cuda_memory" for PyTorch, 
                           "compilation_time" for JAX)
    """
    loss: float
    grad_norm: float
    learning_rate: float
    step: int
    # Backend-specific metrics (e.g. "cuda_memory" for PyTorch, "compilation_time" for JAX)
    framework_specific: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TrainingMetrics to dictionary."""
        return {
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "step": self.step,
            **self.framework_specific,
        }


class RLAlgorithm(ABC, Generic[State, Batch]):
    """
    Base interface for RL algorithms (backend-agnostic).
    
    Concrete implementations will handle the math using their respective framework (PyTorch/JAX).
    This interface follows the PRD specification (§3.1).
    """

    @abstractmethod
    def init_state(self, rng: Any, config: Any) -> State:
        """
        Initialize algorithm state (e.g., model parameters, optimizer state).
        
        Args:
            rng: Random number generator/key (framework-specific)
            config: Algorithm configuration
            
        Returns:
            Initial algorithm state
        """
        pass

    @abstractmethod
    def train_step(self, state: State, batch: Batch) -> Tuple[State, TrainingMetrics]:
        """
        Perform a single training step.
        
        Args:
            state: Current algorithm state
            batch: Training batch (framework-specific format)
            
        Returns:
            Tuple of (updated_state, metrics)
        """
        pass

    @abstractmethod
    def compute_loss(self, rollouts: Any, advantages: Optional[Any] = None) -> LossDict:
        """
        Compute loss given rollouts and optional advantages.
        
        Args:
            rollouts: Rollout data (framework-specific format)
            advantages: Optional advantage estimates
            
        Returns:
            Dictionary of loss components (e.g., {"total": ..., "policy": ..., "value": ...})
        """
        pass


class OfflineRLAlgorithm(RLAlgorithm[State, Batch], Generic[State, Batch]):
    """
    Specialized interface for Offline RL (DPO, CQL, etc.) where data comes from a dataset.
    
    Offline RL algorithms learn from static interaction traces rather than live environment
    interaction. This interface extends RLAlgorithm with dataset processing capabilities.
    """
    
    @abstractmethod
    def process_dataset(self, dataset: Any) -> Any:
        """
        Pre-process a raw dataset (e.g., interaction traces) into a format suitable for training.
        
        Args:
            dataset: Raw dataset (e.g., list of InteractionTrace objects)
            
        Returns:
            Processed dataset in framework-specific format
        """
        pass

