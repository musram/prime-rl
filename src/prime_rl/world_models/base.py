"""
World Model core abstractions.

This module defines the base interface for world models and a registry mechanism
for registering concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from pathlib import Path
from pydantic import BaseModel, Field

from loguru import logger


class WorldModelConfig(BaseModel):
    """
    Configuration for a world model.
    
    Attributes:
        algorithm: Algorithm name (e.g., "crm_mlp", "text_transformer")
        state_representation: State representation type (e.g., "crm_structured", "text")
        checkpoint_path: Optional path to checkpoint for loading
        algorithm_config: Algorithm-specific configuration dictionary
    """
    algorithm: str = Field(..., description="World model algorithm name")
    state_representation: str = Field(..., description="State representation type")
    checkpoint_path: Optional[Path] = Field(default=None, description="Path to checkpoint")
    algorithm_config: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific config")


class WorldModel(ABC):
    """
    Abstract base class for world models.
    
    A world model predicts the next state, reward, and done flag given
    the current state and action. It can be trained from InteractionTrace data
    and used to simulate environment dynamics.
    """
    
    def __init__(self, config: WorldModelConfig):
        """
        Initialize world model.
        
        Args:
            config: WorldModelConfig instance
        """
        self.config = config
    
    @abstractmethod
    def predict_next(
        self,
        state_batch: Any,
        action_batch: Any,
    ) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        """
        Predict next state, reward, and done flag.
        
        Args:
            state_batch: Batch of current states (format depends on state_representation)
            action_batch: Batch of actions (format depends on action encoding)
            
        Returns:
            Tuple of (next_state_batch, reward_batch, done_batch, info_dict)
            - next_state_batch: Predicted next states
            - reward_batch: Predicted rewards
            - done_batch: Predicted done flags
            - info_dict: Additional information (e.g., confidence, auxiliary predictions)
        """
        pass
    
    @abstractmethod
    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform one training step.
        
        Args:
            batch: Training batch dictionary with keys like:
                - state: Current states
                - action: Actions
                - next_state: Next states (targets)
                - reward: Rewards (targets)
                - done: Done flags (targets)
                
        Returns:
            Dictionary of training metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save world model to disk.
        
        Args:
            path: Directory path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load world model from disk.
        
        Args:
            path: Directory path to load model from
        """
        pass
    
    @abstractmethod
    def sample_initial_state(self, num_samples: int = 1) -> Any:
        """
        Sample initial states from the world model's learned distribution.
        
        Args:
            num_samples: Number of states to sample
            
        Returns:
            Batch of initial states
        """
        pass


# Registry for world model implementations
WORLD_MODEL_REGISTRY: Dict[str, Type[WorldModel]] = {}


def register_world_model(name: str, model_class: Type[WorldModel]) -> None:
    """
    Register a world model implementation.
    
    Args:
        name: Algorithm name (e.g., "crm_mlp")
        model_class: WorldModel subclass
    """
    WORLD_MODEL_REGISTRY[name] = model_class
    logger.debug(f"Registered world model: {name} -> {model_class.__name__}")


def create_world_model(config: WorldModelConfig) -> WorldModel:
    """
    Create a world model instance from configuration.
    
    Args:
        config: WorldModelConfig instance
        
    Returns:
        WorldModel instance
        
    Raises:
        ValueError: If algorithm is not registered
    """
    if config.algorithm not in WORLD_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown world model algorithm: {config.algorithm}. "
            f"Available algorithms: {list(WORLD_MODEL_REGISTRY.keys())}"
        )
    
    model_class = WORLD_MODEL_REGISTRY[config.algorithm]
    model = model_class(config)
    
    # Load checkpoint if provided
    if config.checkpoint_path and config.checkpoint_path.exists():
        model.load(config.checkpoint_path)
        logger.info(f"Loaded world model checkpoint from {config.checkpoint_path}")
    
    return model

