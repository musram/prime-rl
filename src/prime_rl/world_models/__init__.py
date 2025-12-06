"""
World Model framework for PRIME-RL.

This module provides a generic, pluggable framework for learning and using
environment models that can be used through the standard EnvironmentAdapter interface.
"""

from prime_rl.world_models.base import (
    WorldModel,
    WorldModelConfig,
    WORLD_MODEL_REGISTRY,
    create_world_model,
    register_world_model,
)
from prime_rl.world_models.env_adapter import WorldModelEnvAdapter

# Auto-import concrete implementations to register them
try:
    from prime_rl.world_models import crm_mlp  # noqa: F401
except ImportError:
    pass

try:
    from prime_rl.world_models import text_transformer  # noqa: F401
except ImportError:
    pass

__all__ = [
    "WorldModel",
    "WorldModelConfig",
    "WORLD_MODEL_REGISTRY",
    "create_world_model",
    "register_world_model",
    "WorldModelEnvAdapter",
]

