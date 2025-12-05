"""
Prime Intellect Verifier Client.

This module implements PrimeIntellectVerifierClient, which connects to Prime Intellect
Verifier APIs (including rubric-based verifiers) to provide reward signals for RL training.
"""

import json
import os
from typing import Any, Dict, List, Optional

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


class PrimeIntellectVerifierClient(VerifierClient):
    """
    Verifier client for Prime Intellect Verifier APIs.
    
    Connects to Prime Intellect verifier endpoints (including rubric-based verifiers
    following RaR-style or BetterEvaluation patterns) to provide reward signals.
    
    This client follows the specification in `docs/verifiers_and_environments_integration.md`.
    
    Example:
        ```python
        client = PrimeIntellectVerifierClient(
            endpoint="https://api.primeintellect.ai/verifier/rubric",
            api_key="your-api-key",
        )
        
        result = client.verify(
            observation="prompt text",
            action="model completion",
            trace_id="trace-123",
        )
        print(f"Reward: {result.reward}, Metadata: {result.metadata}")
        ```
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        default_retry_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Prime Intellect verifier client.
        
        Args:
            endpoint: Verifier endpoint URL (e.g., "https://api.primeintellect.ai/verifier/rubric")
            api_key: Optional API key (can also be set via env var if api_key starts with "env:")
            timeout: Default timeout in seconds
            default_retry_config: Optional default retry configuration
        """
        if not HTTPX_AVAILABLE and requests is None:
            raise ImportError(
                "HTTP client required for PrimeIntellectVerifierClient. Install with: pip install httpx or requests"
            )
        
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.default_retry_config = default_retry_config or {}
        
        # Handle API key (support "env:VAR_NAME" syntax)
        if api_key and api_key.startswith("env:"):
            env_var = api_key[4:]
            self.api_key = os.getenv(env_var)
            if not self.api_key:
                raise ValueError(f"Environment variable {env_var} not set")
        else:
            self.api_key = api_key or os.getenv("PRIME_INTELLECT_API_KEY")
        
        # Setup HTTP client headers
        self.headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # HTTP client (created lazily for async)
        self._sync_client: Optional[Any] = None
        self._async_client: Optional[Any] = None
    
    def _get_sync_client(self):
        """Get or create synchronous HTTP client."""
        if self._sync_client is None:
            if HTTPX_AVAILABLE:
                self._sync_client = httpx.Client(
                    base_url=self.endpoint,
                    headers=self.headers,
                    timeout=self.timeout,
                )
            else:
                self._sync_client = requests.Session()
                self._sync_client.headers.update(self.headers)
                self._sync_client.timeout = self.timeout
        return self._sync_client
    
    async def _get_async_client(self):
        """Get or create asynchronous HTTP client."""
        if self._async_client is None:
            if HTTPX_AVAILABLE:
                self._async_client = httpx.AsyncClient(
                    base_url=self.endpoint,
                    headers=self.headers,
                    timeout=self.timeout,
                )
            else:
                raise ImportError("httpx required for async operations")
        return self._async_client
    
    def _build_verification_payload(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build verification request payload matching Prime Intellect Verifier API.
        
        Args:
            observation: Environment observation (prompt)
            action: Agent action (model completion)
            trace_id: Optional trace identifier
            metadata: Optional metadata (may contain rubric_metadata for rubric-based verification)
            
        Returns:
            Payload dictionary
        """
        payload = {
            "prompt": observation if isinstance(observation, str) else json.dumps(observation),
            "completion": action if isinstance(action, str) else json.dumps(action),
        }
        
        if trace_id:
            payload["trace_id"] = trace_id
        
        # Extract rubric metadata from metadata dict if present
        if metadata:
            if "rubric" in metadata:
                payload["rubric"] = metadata["rubric"]
            # Include other metadata fields
            for key, value in metadata.items():
                if key != "rubric":
                    payload[key] = value
        
        return payload
    
    def _parse_verification_response(self, data: Dict[str, Any]) -> VerificationResult:
        """
        Parse verification response into VerificationResult.
        
        Args:
            data: Response JSON data
            
        Returns:
            VerificationResult object
            
        Raises:
            VerifierError: If response format is invalid
        """
        try:
            # Extract reward (scalar)
            reward = float(data.get("reward", data.get("score", 0.0)))
            
            # Extract success (if available)
            success = bool(data.get("success", reward > 0.0))
            
            # Extract metadata (rubric scores, etc.)
            metadata = data.get("metadata", {})
            
            # Include rubric scores if present
            if "rubric_scores" in data:
                metadata["rubric_scores"] = data["rubric_scores"]
            
            if "criteria" in data:
                metadata["criteria"] = data["criteria"]
            
            # Include trace_id if present
            trace_id = data.get("trace_id")
            
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
        Synchronously verify an observation-action pair.
        
        Args:
            observation: Environment observation (prompt)
            action: Agent action (model completion). For rubric-based verification,
                can be a dict with "completion" and "rubric" keys:
                {"completion": "...", "rubric": {...}}
            trace_id: Optional trace identifier
            timeout: Optional timeout in seconds (overrides default)
            retry_config: Optional retry configuration
            
        Returns:
            VerificationResult with reward and success status
            
        Raises:
            VerifierTimeoutError: If verification times out
            VerifierNetworkError: If network error occurs
            VerifierError: If verification fails
        """
        timeout = timeout or self.timeout
        retry_cfg = self._get_retry_config(retry_config)
        
        # Extract rubric metadata from action if it's a dict with rubric key
        metadata = None
        if isinstance(action, dict) and "rubric" in action:
            metadata = {"rubric": action["rubric"]}
            # Use completion field if present, otherwise use action dict without rubric
            action = action.get("completion", {k: v for k, v in action.items() if k != "rubric"})
        
        payload = self._build_verification_payload(observation, action, trace_id, metadata)
        client = self._get_sync_client()
        
        def _make_request():
            try:
                if HTTPX_AVAILABLE:
                    response = client.post("", json=payload, timeout=timeout)
                    response.raise_for_status()
                    return self._parse_verification_response(response.json())
                else:
                    response = client.post(self.endpoint, json=payload, timeout=timeout)
                    response.raise_for_status()
                    return self._parse_verification_response(response.json())
            except httpx.TimeoutException if HTTPX_AVAILABLE else requests.exceptions.Timeout as e:
                raise VerifierTimeoutError(f"Request timed out after {timeout}s") from e
            except (httpx.HTTPError if HTTPX_AVAILABLE else requests.exceptions.RequestException) as e:
                raise VerifierNetworkError(f"Network error: {e}") from e
            except json.JSONDecodeError as e:
                raise VerifierError(f"Invalid JSON response: {e}") from e
        
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
        Asynchronously verify an observation-action pair.
        
        Args:
            observation: Environment observation (prompt)
            action: Agent action (model completion). For rubric-based verification,
                can be a dict with "completion" and "rubric" keys:
                {"completion": "...", "rubric": {...}}
            trace_id: Optional trace identifier
            timeout: Optional timeout in seconds (overrides default)
            retry_config: Optional retry configuration
            
        Returns:
            VerificationResult with reward and success status
            
        Raises:
            VerifierTimeoutError: If verification times out
            VerifierNetworkError: If network error occurs
            VerifierError: If verification fails
        """
        timeout = timeout or self.timeout
        retry_cfg = self._get_retry_config(retry_config)
        
        # Extract rubric metadata from action if it's a dict with rubric key
        metadata = None
        if isinstance(action, dict) and "rubric" in action:
            metadata = {"rubric": action["rubric"]}
            # Use completion field if present, otherwise use action dict without rubric
            action = action.get("completion", {k: v for k, v in action.items() if k != "rubric"})
        
        payload = self._build_verification_payload(observation, action, trace_id, metadata)
        client = await self._get_async_client()
        
        async def _make_request():
            try:
                response = await client.post("", json=payload, timeout=timeout)
                response.raise_for_status()
                return self._parse_verification_response(response.json())
            except httpx.TimeoutException as e:
                raise VerifierTimeoutError(f"Request timed out after {timeout}s") from e
            except httpx.HTTPError as e:
                raise VerifierNetworkError(f"Network error: {e}") from e
            except json.JSONDecodeError as e:
                raise VerifierError(f"Invalid JSON response: {e}") from e
        
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
            observations: List of observations (prompts)
            actions: List of actions (completions)
            trace_ids: Optional list of trace identifiers
            timeout: Optional timeout in seconds (per request)
            retry_config: Optional retry configuration
            
        Returns:
            List of VerificationResult objects
            
        Raises:
            VerifierError: If verification fails
        """
        if len(observations) != len(actions):
            raise VerifierError("observations and actions must have same length")
        
        if trace_ids and len(trace_ids) != len(observations):
            raise VerifierError("trace_ids must have same length as observations")
        
        # Create tasks for parallel execution
        import asyncio
        
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
        Clean up verifier client resources (close HTTP clients).
        """
        if self._sync_client:
            if HTTPX_AVAILABLE:
                self._sync_client.close()
            else:
                self._sync_client.close()
            self._sync_client = None
        
        if self._async_client:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._async_client.aclose())
                else:
                    loop.run_until_complete(self._async_client.aclose())
            except RuntimeError:
                asyncio.run(self._async_client.aclose())
            self._async_client = None

