"""
World Model Environment Adapter.

Wraps any WorldModel implementation into an EnvironmentAdapter, allowing
world models to be used seamlessly with existing PRIME-RL infrastructure.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import random

from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.world_models.base import WorldModel, WorldModelConfig, create_world_model
from prime_rl.world_models.data import (
    StateEncoder,
    ActionEncoder,
    CRMStateEncoder,
    TextObsEncoder,
    DictActionEncoder,
)
from loguru import logger


def get_encoder_for_state_representation(state_representation: str) -> StateEncoder:
    """
    Get appropriate state encoder for state representation type.
    
    Args:
        state_representation: State representation type (e.g., "crm_structured", "text")
        
    Returns:
        StateEncoder instance
    """
    if state_representation == "crm_structured":
        return CRMStateEncoder()
    elif state_representation == "text":
        return TextObsEncoder()
    else:
        raise ValueError(f"Unknown state representation: {state_representation}")


def get_encoder_for_action(action_type: str = "dict") -> ActionEncoder:
    """
    Get appropriate action encoder.
    
    Args:
        action_type: Action type (default: "dict")
        
    Returns:
        ActionEncoder instance
    """
    if action_type == "dict":
        return DictActionEncoder()
    else:
        raise ValueError(f"Unknown action type: {action_type}")


class WorldModelEnvAdapter(EnvironmentAdapter):
    """
    Environment adapter that wraps a WorldModel.
    
    This allows any world model to be used as an environment through the
    standard EnvironmentAdapter interface, enabling:
    - Pure simulation rollouts
    - Mixed real/simulated training (Dyna-style)
    - Sim-to-real gap analysis
    
    Example:
        ```python
        config = WorldModelConfig(
            algorithm="crm_mlp",
            state_representation="crm_structured",
            checkpoint_path=Path("checkpoints/crm_world_model"),
        )
        world_model = create_world_model(config)
        env = WorldModelEnvAdapter(world_model, config)
        
        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step({"action_type": "reply_to_ticket"})
        ```
    """
    
    def __init__(
        self,
        world_model: Optional[WorldModel] = None,
        config: Optional[WorldModelConfig] = None,
        initial_states: Optional[List[Any]] = None,
    ):
        """
        Initialize WorldModelEnvAdapter.
        
        Args:
            world_model: WorldModel instance (created from config if not provided)
            config: WorldModelConfig instance (required if world_model not provided)
            initial_states: Optional list of initial states for sampling (from training data)
        """
        if world_model is None:
            if config is None:
                raise ValueError("Either world_model or config must be provided")
            world_model = create_world_model(config)
        
        if config is None:
            config = world_model.config
        
        self.world_model = world_model
        self.config = config
        
        # Initialize encoders
        self.state_encoder = get_encoder_for_state_representation(config.state_representation)
        self.action_encoder = get_encoder_for_action("dict")
        
        # Initial state distribution (for reset)
        self.initial_states = initial_states or []
        
        # Current state
        self._current_state: Optional[Any] = None
        self._episode_started = False
        self._step_count = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the world model environment.
        
        Args:
            seed: Optional random seed
            options: Optional reset options (e.g., initial_state)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        if seed is not None:
            random.seed(seed)
        
        # Sample initial state
        if options and "initial_state" in options:
            initial_state = options["initial_state"]
        elif self.initial_states:
            initial_state = random.choice(self.initial_states)
        else:
            # Sample from world model
            sampled = self.world_model.sample_initial_state(num_samples=1)
            initial_state = sampled[0] if sampled else {}
        
        self._current_state = initial_state
        self._episode_started = True
        self._step_count = 0
        
        # Render observation from state
        observation = self._render_state(initial_state)
        
        info = {
            "world_model": self.config.algorithm,
            "state_representation": self.config.state_representation,
            "initial_state": initial_state,
        }
        
        return observation, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the world model environment.
        
        Args:
            action: Agent action
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict)
        """
        if not self._episode_started:
            raise RuntimeError("Environment must be reset before stepping")
        
        if self._current_state is None:
            raise RuntimeError("No current state")
        
        # Predict next state, reward, and done
        next_states, rewards, dones, info = self.world_model.predict_next(
            state_batch=[self._current_state],
            action_batch=[action],
        )
        
        # Update state
        self._current_state = next_states[0]
        reward = rewards[0]
        terminated = dones[0]
        truncated = False
        
        self._step_count += 1
        
        # Render observation from next state
        observation = self._render_state(self._current_state)
        
        info_dict = {
            "step": self._step_count,
            "world_model": self.config.algorithm,
            **info,
        }
        
        return observation, reward, terminated, truncated, info_dict
    
    def _render_state(self, state: Any) -> Any:
        """
        Render state as observation.
        
        For structured states, returns a dict. For text states, returns text.
        
        Args:
            state: State in internal format
            
        Returns:
            Observation (rendered state)
        """
        if self.config.state_representation == "crm_structured":
            # Return state dict directly (or format as observation)
            if isinstance(state, dict):
                return state
            else:
                # Decode if encoded
                return self.state_encoder.decode(state)
        elif self.config.state_representation == "text":
            # Return text observation
            if isinstance(state, str):
                return state
            else:
                return self.state_encoder.decode(state)
        else:
            # Default: return state as-is
            return state
    
    def render(self) -> Optional[Any]:
        """
        Render the current state of the environment.
        
        Returns:
            Current state/observation
        """
        if self._current_state is not None:
            return self._render_state(self._current_state)
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        self._current_state = None
        self._episode_started = False
        self._step_count = 0
    
    @property
    def observation_space(self) -> Any:
        """
        Get the observation space specification.
        
        Returns:
            None (world models don't have formal spaces)
        """
        return None
    
    @property
    def action_space(self) -> Any:
        """
        Get the action space specification.
        
        Returns:
            List of available action types (if applicable)
        """
        return None
    
    def set_initial_states(self, states: List[Any]) -> None:
        """
        Set initial states for sampling during reset.
        
        Args:
            states: List of initial states (from training data)
        """
        self.initial_states = states
        logger.info(f"Set {len(states)} initial states for world model environment")

