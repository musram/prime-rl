"""
Prime Intellect Environment Adapter.

This module implements PrimeIntellectEnvAdapter, which connects to Prime Intellect
Dashboard environments via HTTP API, enabling PRIME-RL to use Prime Intellect
environments as RL training environments.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    try:
        import requests
        HTTPX_AVAILABLE = False
        httpx = None  # type: ignore
    except ImportError:
        HTTPX_AVAILABLE = False
        httpx = None  # type: ignore
        requests = None  # type: ignore

from prime_rl.core.environment import EnvironmentAdapter
from loguru import logger


class PrimeIntellectEnvAdapter(EnvironmentAdapter):
    """
    Environment adapter for Prime Intellect Dashboard environments.
    
    Connects to Prime Intellect environment APIs to provide a standard
    EnvironmentAdapter interface for RL training.
    
    This adapter follows the specification in `docs/verifiers_and_environments_integration.md`.
    
    Example:
        ```python
        adapter = PrimeIntellectEnvAdapter(
            environment_id="browser-gym-v1",
            endpoint="https://api.primeintellect.ai/environments",
            api_key="your-api-key",
        )
        
        obs, info = adapter.reset()
        obs, reward, terminated, truncated, info = adapter.step("click(button)")
        ```
    """
    
    def __init__(
        self,
        environment_id: str,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize Prime Intellect environment adapter.
        
        Args:
            environment_id: Environment identifier (e.g., "browser-gym-v1")
            endpoint: Base endpoint URL (e.g., "https://api.primeintellect.ai/environments")
            api_key: Optional API key (can also be set via env var if api_key starts with "env:")
            timeout: Request timeout in seconds
        """
        if not HTTPX_AVAILABLE and requests is None:
            raise ImportError(
                "HTTP client required for PrimeIntellectEnvAdapter. Install with: pip install httpx or requests"
            )
        
        self.environment_id = environment_id
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        
        # Handle API key (support "env:VAR_NAME" syntax)
        if api_key and api_key.startswith("env:"):
            env_var = api_key[4:]
            self.api_key = os.getenv(env_var)
            if not self.api_key:
                raise ValueError(f"Environment variable {env_var} not set")
        else:
            self.api_key = api_key or os.getenv("PRIME_INTELLECT_API_KEY")
        
        # Setup HTTP client
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        if HTTPX_AVAILABLE:
            self._client = httpx.Client(
                base_url=self.endpoint,
                headers=headers,
                timeout=self.timeout,
            )
        else:
            # Use requests
            self._client = requests.Session()
            self._client.headers.update(headers)
            self._client.timeout = self.timeout
        
        # Track current state
        self._current_observation: Optional[Any] = None
        self._episode_id: Optional[str] = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            options: Optional dictionary of reset options
            
        Returns:
            Tuple of (observation, info_dict)
            
        Raises:
            RuntimeError: If reset fails
        """
        payload = {
            "env_id": self.environment_id,
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        if options:
            payload.update(options)
        
        try:
            if HTTPX_AVAILABLE:
                response = self._client.post("/reset", json=payload)
                response.raise_for_status()
                data = response.json()
            else:
                response = self._client.post(f"{self.endpoint}/reset", json=payload)
                response.raise_for_status()
                data = response.json()
            
            # Extract observation and info
            observation = data.get("observation", data.get("obs"))
            info = data.get("info", {})
            self._episode_id = data.get("episode_id")
            
            if observation is None:
                raise ValueError("Response missing observation field")
            
            self._current_observation = observation
            
            logger.debug(f"Environment reset: episode_id={self._episode_id}")
            
            return observation, info
        
        except Exception as e:
            if HTTPX_AVAILABLE and isinstance(e, httpx.HTTPError):
                raise RuntimeError(f"Failed to reset environment: {e}") from e
            elif not HTTPX_AVAILABLE and isinstance(e, requests.exceptions.RequestException):
                raise RuntimeError(f"Failed to reset environment: {e}") from e
            else:
                raise RuntimeError(f"Failed to reset environment: {e}") from e
    
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
            
        Raises:
            RuntimeError: If step fails
        """
        if self._current_observation is None:
            raise RuntimeError("Environment must be reset before stepping")
        
        payload = {
            "action": action,
        }
        
        if self._episode_id:
            payload["episode_id"] = self._episode_id
        
        try:
            if HTTPX_AVAILABLE:
                response = self._client.post("/step", json=payload)
                response.raise_for_status()
                data = response.json()
            else:
                response = self._client.post(f"{self.endpoint}/step", json=payload)
                response.raise_for_status()
                data = response.json()
            
            # Extract step results
            observation = data.get("observation", data.get("obs"))
            reward = float(data.get("reward", 0.0))
            terminated = bool(data.get("terminated", data.get("done", False)))
            truncated = bool(data.get("truncated", False))
            info = data.get("info", {})
            
            if observation is None:
                raise ValueError("Response missing observation field")
            
            self._current_observation = observation
            
            logger.debug(f"Environment step: reward={reward}, terminated={terminated}, truncated={truncated}")
            
            return observation, reward, terminated, truncated, info
        
        except Exception as e:
            if HTTPX_AVAILABLE and isinstance(e, httpx.HTTPError):
                raise RuntimeError(f"Failed to step environment: {e}") from e
            elif not HTTPX_AVAILABLE and isinstance(e, requests.exceptions.RequestException):
                raise RuntimeError(f"Failed to step environment: {e}") from e
            else:
                raise RuntimeError(f"Failed to step environment: {e}") from e
    
    def render(self) -> Optional[Any]:
        """
        Render the current state of the environment.
        
        Returns:
            Current observation or None if rendering not supported
        """
        return self._current_observation
    
    def close(self) -> None:
        """
        Clean up environment resources.
        """
        if HTTPX_AVAILABLE:
            self._client.close()
        else:
            self._client.close()
        
        self._current_observation = None
        self._episode_id = None
    
    @property
    def observation_space(self) -> Any:
        """
        Get the observation space specification.
        
        Returns:
            None (Prime Intellect API doesn't expose explicit spaces)
        """
        return None
    
    @property
    def action_space(self) -> Any:
        """
        Get the action space specification.
        
        Returns:
            None (Prime Intellect API doesn't expose explicit spaces)
        """
        return None

