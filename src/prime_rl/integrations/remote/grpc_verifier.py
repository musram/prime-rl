"""
gRPC-based verifier client for remote reward/grading APIs.

This module implements GrpcVerifierClient, which sends verification requests to
remote gRPC endpoints and handles responses with retry logic and error handling.
"""

from typing import Any, Dict, List, Optional
import asyncio

try:
    import grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    grpc = None  # type: ignore

from prime_rl.core.verifier import (
    VerifierClient,
    VerificationResult,
    VerifierError,
    VerifierTimeoutError,
    VerifierNetworkError,
)
from prime_rl.orchestrator.retry import RetryConfig, retry_with_backoff_async
from loguru import logger


class GrpcVerifierClient(VerifierClient):
    """
    gRPC-based verifier client for remote reward/grading APIs.
    
    Sends verification requests to remote gRPC endpoints and handles responses
    with retry logic, exponential backoff, and proper error handling.
    
    This implementation follows the VerifierClient interface and provides both
    synchronous and asynchronous verification methods.
    
    Note: This is a stub implementation. In production, you would generate
    gRPC stubs from a .proto file defining the verification service.
    
    Example:
        ```python
        client = GrpcVerifierClient(
            endpoint="localhost:50051",
            service_stub=VerificationServiceStub
        )
        
        result = client.verify(
            observation="user clicked button",
            action="click(button_id)",
            trace_id="trace-123"
        )
        ```
    """
    
    def __init__(
        self,
        endpoint: str,
        service_stub: Any,  # gRPC service stub class
        timeout: float = 30.0,
        default_retry_config: Optional[Dict[str, Any]] = None,
        channel_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize gRPC verifier client.
        
        Args:
            endpoint: gRPC endpoint (host:port)
            service_stub: gRPC service stub class (generated from .proto)
            timeout: Default timeout in seconds
            default_retry_config: Optional default retry configuration
            channel_options: Optional gRPC channel options
        """
        if not GRPC_AVAILABLE:
            raise ImportError(
                "gRPC required for GrpcVerifierClient. Install with: pip install grpcio"
            )
        
        self.endpoint = endpoint
        self.service_stub_class = service_stub
        self.timeout = timeout
        self.default_retry_config = default_retry_config or {}
        self.channel_options = channel_options or {}
        
        # Create gRPC channel
        self._channel: Optional[Any] = None
        self._stub: Optional[Any] = None
        
        # Session for async requests
        self._async_channel: Optional[Any] = None
        self._async_stub: Optional[Any] = None
    
    def _ensure_channel(self):
        """Ensure gRPC channel is created."""
        if self._channel is None:
            self._channel = grpc.insecure_channel(
                self.endpoint,
                options=list(self.channel_options.items()),
            )
            self._stub = self.service_stub_class(self._channel)
    
    async def _ensure_async_channel(self):
        """Ensure async gRPC channel is created."""
        if self._async_channel is None:
            self._async_channel = grpc.aio.insecure_channel(
                self.endpoint,
                options=list(self.channel_options.items()),
            )
            self._async_stub = self.service_stub_class(self._async_channel)
    
    def _build_request(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Build gRPC request message.
        
        This is a stub - in production, you would use a generated request message class.
        
        Args:
            observation: Environment observation
            action: Agent action
            trace_id: Optional trace identifier
            metadata: Optional additional metadata
            
        Returns:
            gRPC request message object
        """
        # Stub implementation - would use generated message class
        # Example: VerificationRequest(observation=observation, action=action, ...)
        return {
            "observation": observation,
            "action": action,
            "trace_id": trace_id,
            "metadata": metadata or {},
        }
    
    def _parse_response(self, response: Any) -> VerificationResult:
        """
        Parse gRPC response into VerificationResult.
        
        This is a stub - in production, you would parse from generated response message.
        
        Args:
            response: gRPC response message
            
        Returns:
            VerificationResult object
        """
        # Stub implementation - would parse from generated message
        # Example: response.reward, response.success, etc.
        if isinstance(response, dict):
            return VerificationResult(
                reward=float(response.get("reward", 0.0)),
                success=bool(response.get("success", False)),
                metadata=response.get("metadata", {}),
                trace_id=response.get("trace_id"),
            )
        else:
            # Assume response has attributes
            return VerificationResult(
                reward=float(getattr(response, "reward", 0.0)),
                success=bool(getattr(response, "success", False)),
                metadata=getattr(response, "metadata", {}),
                trace_id=getattr(response, "trace_id", None),
            )
    
    def verify(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Synchronously verify an observation-action pair via gRPC.
        
        Args:
            observation: Environment observation
            action: Agent action
            trace_id: Optional trace identifier
            timeout: Optional timeout in seconds
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
        
        self._ensure_channel()
        
        request = self._build_request(observation, action, trace_id)
        
        def _make_request():
            try:
                # Call gRPC method (stub implementation)
                # Example: response = self._stub.Verify(request, timeout=timeout)
                # For now, raise NotImplementedError
                raise NotImplementedError(
                    "GrpcVerifierClient.verify() requires generated gRPC stubs. "
                    "Please generate stubs from your .proto file and implement this method."
                )
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise VerifierTimeoutError(f"Request timed out after {timeout}s") from e
                else:
                    raise VerifierNetworkError(f"gRPC error: {e}") from e
        
        # Apply retry logic
        from prime_rl.orchestrator.retry import retry_with_backoff_sync
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
        Asynchronously verify an observation-action pair via gRPC.
        
        Args:
            observation: Environment observation
            action: Agent action
            trace_id: Optional trace identifier
            timeout: Optional timeout in seconds
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
        
        await self._ensure_async_channel()
        
        request = self._build_request(observation, action, trace_id)
        
        async def _make_request():
            try:
                # Call async gRPC method (stub implementation)
                # Example: response = await self._async_stub.Verify(request, timeout=timeout)
                # For now, raise NotImplementedError
                raise NotImplementedError(
                    "GrpcVerifierClient.verify_async() requires generated gRPC stubs. "
                    "Please generate stubs from your .proto file and implement this method."
                )
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise VerifierTimeoutError(f"Request timed out after {timeout}s") from e
                else:
                    raise VerifierNetworkError(f"gRPC error: {e}") from e
        
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
        Clean up verifier client resources (close gRPC channels).
        """
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
        
        if self._async_channel:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._async_channel.close())
                else:
                    loop.run_until_complete(self._async_channel.close())
            except RuntimeError:
                asyncio.run(self._async_channel.close())
            self._async_channel = None
            self._async_stub = None

