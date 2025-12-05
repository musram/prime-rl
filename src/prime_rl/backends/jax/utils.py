"""
Utility functions for JAX backend.

This module provides helper functions for multi-device training, mixed precision,
and other JAX-specific utilities.
"""

from typing import Any, Dict, Optional
import functools

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    lax = None  # type: ignore


def get_dtype(dtype_str: str) -> Any:
    """
    Convert dtype string to JAX dtype.
    
    Args:
        dtype_str: "float32", "float16", or "bfloat16"
        
    Returns:
        JAX dtype object
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required")
    
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Must be one of {list(dtype_map.keys())}")
    
    return dtype_map[dtype_str]


def get_num_devices() -> int:
    """Get the number of available JAX devices."""
    if not JAX_AVAILABLE:
        return 1
    return len(jax.devices())


def shard_batch(batch: Dict[str, Any], num_devices: int) -> Dict[str, Any]:
    """
    Shard a batch across multiple devices.
    
    Args:
        batch: Batch dictionary with JAX arrays
        num_devices: Number of devices to shard across
        
    Returns:
        Sharded batch dictionary
    """
    if num_devices == 1:
        return batch
    
    sharded = {}
    for key, value in batch.items():
        if isinstance(value, jnp.ndarray):
            # Reshape to [num_devices, batch_per_device, ...]
            batch_size = value.shape[0]
            batch_per_device = batch_size // num_devices
            if batch_per_device > 0:
                sharded_shape = (num_devices, batch_per_device) + value.shape[1:]
                sharded[key] = value[:num_devices * batch_per_device].reshape(sharded_shape)
            else:
                # Not enough samples for sharding
                sharded[key] = value
        else:
            sharded[key] = value
    
    return sharded


def aggregate_gradients(grads: Any) -> Any:
    """
    Aggregate gradients across devices using pmean.
    
    Args:
        grads: Gradient tree structure
        
    Returns:
        Aggregated gradients
    """
    if not JAX_AVAILABLE:
        return grads
    
    def _aggregate(x):
        if isinstance(x, jnp.ndarray):
            return lax.pmean(x, axis_name="devices")
        return x
    
    return jax.tree_map(_aggregate, grads)


def aggregate_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate metrics across devices using pmean.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        Aggregated metrics dictionary
    """
    if not JAX_AVAILABLE:
        return metrics
    
    aggregated = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Convert to array for pmean
            value_array = jnp.array(value)
            aggregated[key] = float(lax.pmean(value_array, axis_name="devices"))
        elif isinstance(value, jnp.ndarray):
            aggregated[key] = float(lax.pmean(value, axis_name="devices"))
        else:
            aggregated[key] = value
    
    return aggregated

