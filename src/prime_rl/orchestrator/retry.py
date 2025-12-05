"""
Retry and backoff utilities for orchestrator.

This module provides retry/backoff semantics for environment and verifier failures
as specified in PRD §4.2 (Distributed Orchestrator).
"""

import asyncio
import time
from typing import Callable, TypeVar, Optional, Dict, Any
from functools import wraps

from prime_rl.core.verifier import VerifierError, VerifierNetworkError, VerifierTimeoutError

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: Optional[float] = None,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay


async def retry_with_backoff_async(
    func: Callable[..., T],
    *args,
    retry_config: Optional[RetryConfig] = None,
    **kwargs,
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Positional arguments for func
        retry_config: Optional retry configuration
        **kwargs: Keyword arguments for func
        
    Returns:
        Function result
        
    Raises:
        VerifierError: If all retries are exhausted
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    delay = retry_config.initial_delay
    last_error = None
    
    for attempt in range(retry_config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except (VerifierNetworkError, asyncio.TimeoutError) as e:
            last_error = e
            if attempt < retry_config.max_retries:
                await asyncio.sleep(delay)
                delay = min(
                    delay * retry_config.backoff_factor,
                    retry_config.max_delay or float("inf"),
                )
            else:
                if isinstance(e, asyncio.TimeoutError):
                    raise VerifierTimeoutError(f"Operation timed out after {retry_config.max_retries + 1} attempts") from e
                raise
        except VerifierError:
            # Don't retry on non-network verifier errors
            raise
        except Exception as e:
            # Don't retry on unexpected errors
            raise VerifierError(f"Unexpected error: {e}") from e
    
    if last_error:
        raise VerifierError(f"Operation failed after {retry_config.max_retries + 1} attempts") from last_error
    
    raise VerifierError("Retry logic failed unexpectedly")


def retry_with_backoff_sync(
    func: Callable[..., T],
    *args,
    retry_config: Optional[RetryConfig] = None,
    **kwargs,
) -> T:
    """
    Retry a sync function with exponential backoff.
    
    Args:
        func: Function to retry
        *args: Positional arguments for func
        retry_config: Optional retry configuration
        **kwargs: Keyword arguments for func
        
    Returns:
        Function result
        
    Raises:
        VerifierError: If all retries are exhausted
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    delay = retry_config.initial_delay
    last_error = None
    
    for attempt in range(retry_config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (VerifierNetworkError, TimeoutError) as e:
            last_error = e
            if attempt < retry_config.max_retries:
                time.sleep(delay)
                delay = min(
                    delay * retry_config.backoff_factor,
                    retry_config.max_delay or float("inf"),
                )
            else:
                if isinstance(e, TimeoutError):
                    raise VerifierTimeoutError(f"Operation timed out after {retry_config.max_retries + 1} attempts") from e
                raise
        except VerifierError:
            # Don't retry on non-network verifier errors
            raise
        except Exception as e:
            # Don't retry on unexpected errors
            raise VerifierError(f"Unexpected error: {e}") from e
    
    if last_error:
        raise VerifierError(f"Operation failed after {retry_config.max_retries + 1} attempts") from last_error
    
    raise VerifierError("Retry logic failed unexpectedly")

