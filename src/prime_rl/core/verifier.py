"""
Verifier client interface for remote reward/grading APIs.

This module defines the VerifierClient abstract base class for integrating with
remote verification services that provide reward signals and grading capabilities.
This enables asynchronous, remote verification as specified in the PRD (§2.2, §3.1, §4.2).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import asyncio
import time


class VerificationResult:
    """
    Result from a verification request.
    
    Attributes:
        reward: Reward value (float)
        success: Whether the task was successful (bool)
        metadata: Additional information (e.g., explanation, confidence, error details)
        trace_id: Optional trace identifier for auditability
    """
    
    def __init__(
        self,
        reward: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ):
        self.reward = reward
        self.success = success
        self.metadata = metadata or {}
        self.trace_id = trace_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "reward": self.reward,
            "success": self.success,
            "metadata": self.metadata,
            "trace_id": self.trace_id,
        }


class VerificationRequest:
    """
    Standardized verification request schema.
    
    This follows the PRD specification (§4.2, Verifier API) for standardizing
    the protocol for remote environment verification.
    
    Attributes:
        observation: Environment observation
        action: Agent action
        trace_id: Optional trace identifier for idempotency/auditability
        metadata: Optional additional metadata
    """
    
    def __init__(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.observation = observation
        self.action = action
        self.trace_id = trace_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "observation": self.observation,
            "action": self.action,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
        }


class VerifierClient(ABC):
    """
    Abstract base class for verifier clients.
    
    Provides a standardized interface for remote verification services that grade
    agent behavior and provide reward signals. Supports both synchronous and
    asynchronous verification with idempotency, retries, and timeouts.
    
    This interface follows the PRD specification (§2.2, Remote Verification; §3.1, VerifierClient).
    The request/response schema is standardized per PRD §4.2.
    """

    @abstractmethod
    def verify(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Synchronously verify an observation-action pair.
        
        Args:
            observation: Environment observation
            action: Agent action
            trace_id: Optional trace identifier for idempotency/auditability
            timeout: Optional timeout in seconds
            retry_config: Optional retry configuration (max_retries, backoff_factor, etc.)
            
        Returns:
            VerificationResult with reward and success status
            
        Raises:
            VerifierTimeoutError: If verification times out
            VerifierNetworkError: If network error occurs
            VerifierError: If verification fails (invalid request, etc.)
        """
        pass

    @abstractmethod
    async def verify_async(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Asynchronously verify an observation-action pair.
        
        Args:
            observation: Environment observation
            action: Agent action
            trace_id: Optional trace identifier for idempotency/auditability
            timeout: Optional timeout in seconds
            retry_config: Optional retry configuration (max_retries, backoff_factor, etc.)
            
        Returns:
            VerificationResult with reward and success status
            
        Raises:
            VerifierTimeoutError: If verification times out
            VerifierNetworkError: If network error occurs
            VerifierError: If verification fails (invalid request, etc.)
        """
        pass

    @abstractmethod
    async def verify_batch(
        self,
        observations: List[Any],
        actions: List[Any],
        trace_ids: Optional[List[Optional[str]]] = None,
        timeout: Optional[float] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> List[VerificationResult]:
        """
        Verify a batch of observation-action pairs (may be more efficient than individual calls).
        
        Args:
            observations: List of observations
            actions: List of actions
            trace_ids: Optional list of trace identifiers
            timeout: Optional timeout in seconds (per item or total, implementation-dependent)
            retry_config: Optional retry configuration
            
        Returns:
            List of VerificationResult objects
            
        Raises:
            VerifierTimeoutError: If verification times out
            VerifierNetworkError: If network error occurs
            VerifierError: If verification fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up verifier client resources (e.g., close connections, cancel pending requests).
        """
        pass


class VerifierError(Exception):
    """Base exception for verifier-related errors."""
    pass


class VerifierTimeoutError(VerifierError):
    """Raised when a verification request times out."""
    pass


class VerifierNetworkError(VerifierError):
    """Raised when a network error occurs during verification."""
    pass


def retry_with_backoff(
    func,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    timeout: Optional[float] = None,
):
    """
    Retry decorator with exponential backoff for verifier calls.
    
    This implements basic retry/backoff semantics as specified in PRD §4.2.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        initial_delay: Initial delay in seconds before first retry
        timeout: Optional timeout in seconds
        
    Returns:
        Function result or raises VerifierError after max_retries
    """
    def wrapper(*args, **kwargs):
        delay = initial_delay
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Check timeout if specified
                if timeout and (time.time() - start_time) > timeout:
                    raise VerifierTimeoutError(f"Verification timed out after {timeout}s")
                
                return result
            except VerifierNetworkError as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    raise
            except Exception as e:
                # Don't retry on non-network errors
                raise VerifierError(f"Verification failed: {e}") from e
        
        if last_error:
            raise last_error
    
    return wrapper

