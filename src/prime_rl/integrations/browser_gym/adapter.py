"""
Browser Gym environment adapter.

This module implements BrowserGymAdapter, which wraps Gymnasium environments
(including browsergym) and maps observations/actions to UniversalRollout format.
"""

from typing import Any, Dict, Optional, Tuple, Callable
import json

try:
    import gymnasium as gym
    from gymnasium import Env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import Env
        GYMNASIUM_AVAILABLE = False
    except ImportError:
        GYMNASIUM_AVAILABLE = False
        gym = None  # type: ignore
        Env = None  # type: ignore

from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.core.algorithms import UniversalRollout
from loguru import logger


class BrowserGymAdapter(EnvironmentAdapter):
    """
    Adapter for Gymnasium/Gym environments, specifically targeting browser environments.
    
    Wraps a standard Gym/Gymnasium environment and provides a consistent interface
    for PRIME-RL. Maps observations (e.g., DOM snapshot, accessibility tree) to
    the UniversalRollout format.
    
    This adapter handles both single-step and multi-step trajectories, converting
    Gym observations and actions into the UniversalRollout representation.
    
    Example:
        ```python
        import gymnasium as gym
        
        env = gym.make("BrowserEnv-v0")
        adapter = BrowserGymAdapter(env)
        
        obs, info = adapter.reset()
        obs, reward, terminated, truncated, info = adapter.step("click(button)")
        
        rollout = adapter.get_rollout()
        ```
    """
    
    def __init__(
        self,
        env: Any,  # gym.Env or gymnasium.Env
        observation_to_string: Optional[Callable[[Any], str]] = None,
        action_to_string: Optional[Callable[[Any], str]] = None,
    ):
        """
        Initialize Browser Gym adapter.
        
        Args:
            env: Gymnasium or Gym environment instance
            observation_to_string: Optional function to convert observation to string.
                Default: uses str() or JSON serialization
            action_to_string: Optional function to convert action to string.
                Default: uses str() or JSON serialization
        """
        if not GYMNASIUM_AVAILABLE and gym is None:
            raise ImportError(
                "Gymnasium or Gym required for BrowserGymAdapter. Install with: pip install gymnasium"
            )
        
        self.env = env
        self.observation_to_string = observation_to_string or self._default_observation_to_string
        self.action_to_string = action_to_string or self._default_action_to_string
        
        # Track trajectory for rollout conversion
        self._current_trajectory: Dict[str, list] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        self._episode_started = False
        self._initial_observation: Any = None
    
    def _default_observation_to_string(self, obs: Any) -> str:
        """Default observation to string conversion."""
        if isinstance(obs, str):
            return obs
        elif isinstance(obs, dict):
            # Try to extract common browser observation fields
            if "dom" in obs:
                return str(obs["dom"])
            elif "accessibility_tree" in obs:
                return str(obs["accessibility_tree"])
            elif "screenshot" in obs:
                return f"Screenshot: {type(obs['screenshot'])}"
            else:
                return json.dumps(obs)
        else:
            return str(obs)
    
    def _default_action_to_string(self, action: Any) -> str:
        """Default action to string conversion."""
        if isinstance(action, str):
            return action
        elif isinstance(action, dict):
            return json.dumps(action)
        else:
            return str(action)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            options: Optional dictionary of reset options (environment-specific)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        # Reset trajectory tracking
        self._current_trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        # Reset environment
        if GYMNASIUM_AVAILABLE:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            # Legacy Gym API
            obs = self.env.reset()
            info = {}
        
        self._initial_observation = obs
        self._current_trajectory["observations"].append(obs)
        self._episode_started = True
        
        return obs, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (format depends on environment)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict)
        """
        if not self._episode_started:
            raise RuntimeError("Environment must be reset before stepping")
        
        # Step environment
        if GYMNASIUM_AVAILABLE:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            # Legacy Gym API (no truncated flag)
            obs, reward, done, info = self.env.step(action)
            terminated = done
            truncated = False
        
        # Track trajectory
        self._current_trajectory["actions"].append(action)
        self._current_trajectory["observations"].append(obs)
        self._current_trajectory["rewards"].append(reward)
        self._current_trajectory["dones"].append(terminated or truncated)
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[Any]:
        """
        Render the current state of the environment (optional).
        
        Returns:
            Rendered output (e.g., image, string, None if rendering not supported)
        """
        try:
            return self.env.render()
        except Exception as e:
            logger.warning(f"Rendering failed: {e}")
            return None
    
    def close(self) -> None:
        """
        Clean up environment resources.
        """
        try:
            self.env.close()
        except Exception as e:
            logger.warning(f"Error closing environment: {e}")
        finally:
            self._episode_started = False
            self._current_trajectory = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": [],
            }
    
    @property
    def observation_space(self) -> Any:
        """
        Get the observation space specification.
        
        Returns:
            Space specification (format depends on environment library)
        """
        return self.env.observation_space
    
    @property
    def action_space(self) -> Any:
        """
        Get the action space specification.
        
        Returns:
            Space specification (format depends on environment library)
        """
        return self.env.action_space
    
    def get_rollout(self) -> UniversalRollout:
        """
        Convert current trajectory to UniversalRollout format.
        
        This method extracts the accumulated trajectory data and converts it
        to a UniversalRollout object suitable for offline RL training.
        
        Returns:
            UniversalRollout object with current trajectory data
        """
        if not self._episode_started or len(self._current_trajectory["observations"]) == 0:
            raise RuntimeError("No trajectory data available. Reset and step the environment first.")
        
        # Extract prompt (first observation)
        prompt = self.observation_to_string(self._initial_observation)
        
        # Extract completion (all actions concatenated)
        actions_str = [self.action_to_string(act) for act in self._current_trajectory["actions"]]
        completion = " ".join(actions_str) if len(actions_str) > 1 else (actions_str[0] if actions_str else "")
        
        # Extract reward (sum of all rewards)
        total_reward = sum(self._current_trajectory["rewards"])
        
        # Build metadata
        metadata = {
            "num_steps": len(self._current_trajectory["actions"]),
            "episode_length": len(self._current_trajectory["rewards"]),
        }
        
        # For multi-step trajectories, include full trajectory data
        observations = None
        actions = None
        dones = None
        
        if len(self._current_trajectory["observations"]) > 1:
            observations = [self.observation_to_string(obs) for obs in self._current_trajectory["observations"]]
            actions = [self.action_to_string(act) for act in self._current_trajectory["actions"]]
            dones = self._current_trajectory["dones"]
        
        return UniversalRollout(
            prompts=[prompt],
            completions=[completion],
            rewards=[total_reward],
            observations=observations,
            actions=actions,
            dones=dones,
            metadata=metadata,
        )

