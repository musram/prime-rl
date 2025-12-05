"""
HTTP-based verifier client for remote reward/grading APIs.

This module implements HttpVerifierClient, which sends verification requests to
remote HTTP endpoints and handles responses with retry logic and error handling.
"""

import json
import time
from typing import Any, Dict, List, Optional
import asyncio

try:
    import aiohttp
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    aiohttp = None  # type: ignore
    requests = None  # type: ignore

from prime_rl.core.verifier import (
    VerifierClient,
    VerificationRequest,
    VerificationResult,
    VerifierError,
    VerifierTimeoutError,
    VerifierNetworkError,
)
from prime_rl.orchestrator.retry import RetryConfig, retry_with_backoff_async, retry_with_backoff_sync
from loguru import logger


class HttpVerifierClient(VerifierClient):
    """
    HTTP-based verifier client for remote reward/grading APIs.
    
    Sends verification requests to remote HTTP endpoints (POST requests) and handles
    responses with retry logic, exponential backoff, and proper error handling.
    
    This implementation follows the VerifierClient interface and provides both
    synchronous and asynchronous verification methods.
    
    Example:
        ```python
        client = HttpVerifierClient(
            endpoint_url="https://api.example.com/verify",
            api_key="your-api-key"
        )
        
        result = client.verify(
            observation="user clicked button",
            action="click(button_id)",
            trace_id="trace-123"
        )
        print(f"Reward: {result.reward}, Success: {result.success}")
        ```
    """
    
    def __init__(
        self,
        endpoint_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        default_retry_config: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize HTTP verifier client.
        
        Args:
            endpoint_url: Base URL for verification endpoint (e.g., "https://api.example.com/verify")
            api_key: Optional API key for authentication (sent in Authorization header)
            timeout: Default timeout in seconds for requests
            default_retry_config: Optional default retry configuration dict with keys:
                - max_retries: int (default: 3)
                - backoff_factor: float (default: 2.0)
                - initial_delay: float (default: 1.0)
                - max_delay: Optional[float] (default: None)
            headers: Optional additional headers to include in requests
        """
        if not HTTP_AVAILABLE:
            raise ImportError(
                "HTTP libraries required for HttpVerifierClient. Install with: pip install requests aiohttp"
            )
        
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.default_retry_config = default_retry_config or {}
        self.headers = headers or {}
        
        # Set up default headers
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.headers.setdefault("Content-Type", "application/json")
        
        # Session for async requests (created lazily)
        self._async_session: Optional[Any] = None
    
    def _build_request_payload(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build request payload for verification endpoint.
        
        Args:
            observation: Environment observation
            action: Agent action
            trace_id: Optional trace identifier
            metadata: Optional additional metadata
            
        Returns:
            Dictionary payload for HTTP request
        """
        payload = {
            "observation": observation,
            "action": action,
        }
        
        if trace_id:
            payload["trace_id"] = trace_id
        
        if metadata:
            payload["metadata"] = metadata
        
        return payload
    
    def _parse_response(self, response_data: Dict[str, Any]) -> VerificationResult:
        """
        Parse HTTP response into VerificationResult.
        
        Args:
            response_data: Response JSON data
            
        Returns:
            VerificationResult object
            
        Raises:
            VerifierError: If response format is invalid
        """
        try:
            reward = float(response_data.get("reward", 0.0))
            success = bool(response_data.get("success", False))
            metadata = response_data.get("metadata", {})
            trace_id = response_data.get("trace_id")
            
            return VerificationResult(
                reward=reward,
                success=success,
                metadata=metadata,
                trace_id=trace_id,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise VerifierError(f"Invalid response format: {e}") from e
    
    def verify(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Synchronously verify an observation-action pair via HTTP.
        
        Args:
            observation: Environment observation
            action: Agent action
            trace_id: Optional trace identifier for idempotency/auditability
            timeout: Optional timeout in seconds (overrides default)
            retry_config: Optional retry configuration (overrides default)
            
        Returns:
            VerificationResult with reward and success status
            
        Raises:
            VerifierTimeoutError: If verification times out
            VerifierNetworkError: If network error occurs
            VerifierError: If verification fails (invalid request, etc.)
        """
        timeout = timeout or self.timeout
        retry_cfg = self._get_retry_config(retry_config)
        
        payload = self._build_request_payload(observation, action, trace_id)
        
        def _make_request():
            try:
                response = requests.post(
                    self.endpoint_url,
                    json=payload,
                    headers=self.headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                return self._parse_response(response.json())
            except requests.exceptions.Timeout as e:
                raise VerifierTimeoutError(f"Request timed out after {timeout}s") from e
            except requests.exceptions.RequestException as e:
                raise VerifierNetworkError(f"Network error: {e}") from e
            except json.JSONDecodeError as e:
                raise VerifierError(f"Invalid JSON response: {e}") from e
        
        # Apply retry logic
        return retry_with_backoff_sync(_make_request, retry_config=RetryConfig(**retry_cfg))
    
    async def verify_async(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Asynchronously verify an observation-action pair via HTTP.
        
        Args:
            observation: Environment observation
            action: Agent action
            trace_id: Optional trace identifier for idempotency/auditability
            timeout: Optional timeout in seconds (overrides default)
            retry_config: Optional retry configuration (overrides default)
            
        Returns:
            VerificationResult with reward and success status
            
        Raises:
            VerifierTimeoutError: If verification times out
            VerifierNetworkError: If network error occurs
            VerifierError: If verification fails (invalid request, etc.)
        """
        timeout = timeout or self.timeout
        retry_cfg = self._get_retry_config(retry_config)
        
        payload = self._build_request_payload(observation, action, trace_id)
        
        # Create async session if needed
        if self._async_session is None:
            self._async_session = aiohttp.ClientSession()
        
        async def _make_request():
            try:
                async with self._async_session.post(
                    self.endpoint_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    return self._parse_response(response_data)
            except asyncio.TimeoutError as e:
                raise VerifierTimeoutError(f"Request timed out after {timeout}s") from e
            except aiohttp.ClientError as e:
                raise VerifierNetworkError(f"Network error: {e}") from e
            except json.JSONDecodeError as e:
                raise VerifierError(f"Invalid JSON response: {e}") from e
        
        # Apply retry logic
        return await retry_with_backoff_async(_make_request, retry_config=RetryConfig(**retry_cfg))
    
    async def verify_batch(
        self,
        observations: List[Any],
        actions: List[Any],
        trace_ids: Optional[List[Optional[str]]] = None,
        timeout: Optional[float] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> List[VerificationResult]:
        """
        Verify a batch of observation-action pairs (parallel async requests).
        
        Args:
            observations: List of observations
            actions: List of actions
            trace_ids: Optional list of trace identifiers
            timeout: Optional timeout in seconds (per request)
            retry_config: Optional retry configuration
            
        Returns:
            List of VerificationResult objects
            
        Raises:
            VerifierTimeoutError: If verification times out
            VerifierNetworkError: If network error occurs
            VerifierError: If verification fails
        """
        if len(observations) != len(actions):
            raise VerifierError("observations and actions must have same length")
        
        if trace_ids and len(trace_ids) != len(observations):
            raise VerifierError("trace_ids must have same length as observations")
        
        # Create tasks for parallel execution
        tasks = []
        for i, (obs, act) in enumerate(zip(observations, actions)):
            trace_id = trace_ids[i] if trace_ids else None
            tasks.append(
                self.verify_async(
                    observation=obs,
                    action=act,
                    trace_id=trace_id,
                    timeout=timeout,
                    retry_config=retry_config,
                )
            )
        
        # Execute all requests in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to VerifierError
        verification_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Verification failed for item {i}: {result}")
                # Create error result
                verification_results.append(
                    VerificationResult(
                        reward=0.0,
                        success=False,
                        metadata={"error": str(result)},
                    )
                )
            else:
                verification_results.append(result)
        
        return verification_results
    
    def _get_retry_config(self, retry_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get retry configuration, merging with defaults."""
        config = self.default_retry_config.copy()
        if retry_config:
            config.update(retry_config)
        return config
    
    def close(self) -> None:
        """
        Clean up verifier client resources (close async session).
        """
        if self._async_session:
            # Close async session (run in event loop if needed)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule close
                    asyncio.create_task(self._async_session.close())
                else:
                    loop.run_until_complete(self._async_session.close())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(self._async_session.close())
            self._async_session = None

