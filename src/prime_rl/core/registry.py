"""
Registry for environment adapters and verifier clients.

This module provides a factory pattern for creating EnvironmentAdapter and
VerifierClient instances from configuration strings, enabling dynamic
instantiation based on TOML/YAML config files.
"""

from typing import Dict, Type, Any, Optional
from loguru import logger

from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.core.verifier import VerifierClient


# Registry for environment adapters
_ENVIRONMENT_ADAPTER_REGISTRY: Dict[str, Type[EnvironmentAdapter]] = {}

# Registry for verifier clients
_VERIFIER_CLIENT_REGISTRY: Dict[str, Type[VerifierClient]] = {}


def register_environment_adapter(name: str, adapter_class: Type[EnvironmentAdapter]) -> None:
    """
    Register an environment adapter class.
    
    Args:
        name: Configuration name (e.g., "prime_intellect")
        adapter_class: EnvironmentAdapter subclass
    """
    _ENVIRONMENT_ADAPTER_REGISTRY[name] = adapter_class
    logger.debug(f"Registered environment adapter: {name} -> {adapter_class.__name__}")


def register_verifier_client(name: str, client_class: Type[VerifierClient]) -> None:
    """
    Register a verifier client class.
    
    Args:
        name: Configuration name (e.g., "prime_intellect_rar")
        client_class: VerifierClient subclass
    """
    _VERIFIER_CLIENT_REGISTRY[name] = client_class
    logger.debug(f"Registered verifier client: {name} -> {client_class.__name__}")


def create_environment_adapter(adapter_type: str, **kwargs) -> EnvironmentAdapter:
    """
    Create an environment adapter instance from configuration.
    
    Args:
        adapter_type: Adapter type name (e.g., "prime_intellect")
        **kwargs: Configuration arguments for the adapter
        
    Returns:
        EnvironmentAdapter instance
        
    Raises:
        ValueError: If adapter type is not registered
    """
    if adapter_type not in _ENVIRONMENT_ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown environment adapter type: {adapter_type}. "
            f"Available types: {list(_ENVIRONMENT_ADAPTER_REGISTRY.keys())}"
        )
    
    adapter_class = _ENVIRONMENT_ADAPTER_REGISTRY[adapter_type]
    return adapter_class(**kwargs)


def create_verifier_client(client_type: str, **kwargs) -> VerifierClient:
    """
    Create a verifier client instance from configuration.
    
    Args:
        client_type: Client type name (e.g., "prime_intellect_rar")
        **kwargs: Configuration arguments for the client
        
    Returns:
        VerifierClient instance
        
    Raises:
        ValueError: If client type is not registered
    """
    if client_type not in _VERIFIER_CLIENT_REGISTRY:
        raise ValueError(
            f"Unknown verifier client type: {client_type}. "
            f"Available types: {list(_VERIFIER_CLIENT_REGISTRY.keys())}"
        )
    
    client_class = _VERIFIER_CLIENT_REGISTRY[client_type]
    return client_class(**kwargs)


def get_registered_adapters() -> Dict[str, str]:
    """Get dictionary of registered adapter types and their class names."""
    return {name: cls.__name__ for name, cls in _ENVIRONMENT_ADAPTER_REGISTRY.items()}


def get_registered_verifiers() -> Dict[str, str]:
    """Get dictionary of registered verifier types and their class names."""
    return {name: cls.__name__ for name, cls in _VERIFIER_CLIENT_REGISTRY.items()}


# Auto-register Prime Intellect integrations
try:
    from prime_rl.integrations.prime_intellect import (
        PrimeIntellectEnvAdapter,
        PrimeIntellectVerifierClient,
    )
    
    register_environment_adapter("prime_intellect", PrimeIntellectEnvAdapter)
    register_verifier_client("prime_intellect_rar", PrimeIntellectVerifierClient)
    register_verifier_client("prime_intellect", PrimeIntellectVerifierClient)  # Alias
except ImportError:
    # Prime Intellect integrations not available
    pass

# Auto-register other integrations
try:
    from prime_rl.integrations.browser_gym import BrowserGymAdapter
    register_environment_adapter("browser_gym", BrowserGymAdapter)
except ImportError:
    pass

try:
    from prime_rl.integrations.remote.http_verifier import HttpVerifierClient
    register_verifier_client("http", HttpVerifierClient)
except ImportError:
    pass

