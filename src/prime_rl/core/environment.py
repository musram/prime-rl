"""
Environment adapter interface for RL environments.

This module defines the EnvironmentAdapter abstract base class that provides a stable
interface over concrete environments (browser envs, internal tools, simulators).
This decouples engines and algorithms from specific UI/SDK implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class EnvironmentAdapter(ABC):
    """
    Abstract base class for environment adapters.
    
    Provides a standardized interface for RL environments, allowing algorithms and
    orchestrators to work with any environment implementation. Concrete adapters
    wrap specific environment libraries (e.g., verifiers, gymnasium, custom simulators).
    
    This interface follows the PRD specification (§3.1, Environment & Verifier Contracts).
    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            options: Optional dictionary of reset options (environment-specific)
            
        Returns:
            Tuple of (observation, info_dict)
            - observation: Initial observation (format depends on environment)
            - info_dict: Additional information (e.g., available actions, metadata)
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (format depends on environment)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict)
            - observation: Next observation
            - reward: Reward signal
            - terminated: Whether episode terminated (task completed/failed)
            - truncated: Whether episode was truncated (time limit, etc.)
            - info_dict: Additional information (e.g., metrics, debug info)
        """
        pass

    @abstractmethod
    def render(self) -> Optional[Any]:
        """
        Render the current state of the environment (optional).
        
        Returns:
            Rendered output (e.g., image, string, None if rendering not supported)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up environment resources.
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """
        Get the observation space specification.
        
        Returns:
            Space specification (format depends on environment library)
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """
        Get the action space specification.
        
        Returns:
            Space specification (format depends on environment library)
        """
        pass


class AsyncEnvironmentAdapter(EnvironmentAdapter):
    """
    Extended interface for asynchronous environments.
    
    Some environments support async stepping for better throughput in distributed settings.
    """

    @abstractmethod
    async def reset_async(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Async version of reset()."""
        pass

    @abstractmethod
    async def step_async(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Async version of step()."""
        pass

