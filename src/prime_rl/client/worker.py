"""
Remote Environment Worker for Forward Deployment.

This module implements RemoteEnvironmentWorker, which wraps a local
EnvironmentAdapter and communicates with the PRIME-RL Orchestrator cluster
to stream traces and receive policy updates.
"""

import asyncio
import time
from typing import Any, Dict, Optional, List
from pathlib import Path
import json

try:
    import aiohttp
    import requests
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    aiohttp = None  # type: ignore
    requests = None  # type: ignore

from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.core.algorithms import UniversalRollout
from prime_rl.core.protocol import (
    WorkerRegistrationRequest,
    WorkerRegistrationResponse,
    TraceSubmissionRequest,
    TraceSubmissionResponse,
    PolicyRequest,
    PolicyResponse,
    ActionRequest,
    ActionResponse,
    ErrorResponse,
)
from prime_rl.orchestrator.retry import RetryConfig, retry_with_backoff_async, retry_with_backoff_sync
from loguru import logger


class RemoteEnvironmentWorker:
    """
    Remote Environment Worker for Forward Deployment Architecture.
    
    Wraps a local EnvironmentAdapter and communicates with the PRIME-RL
    Orchestrator cluster to:
    - Stream traces (UniversalRollout) to the server
    - Receive policy updates
    - Optionally use server-side inference (Inference-as-a-Service mode)
    
    Example:
        ```python
        from prime_rl.integrations import BrowserGymAdapter
        from prime_rl.client import RemoteEnvironmentWorker
        
        env = BrowserGymAdapter(...)
        worker = RemoteEnvironmentWorker(
            environment=env,
            server_url="https://cluster.example.com",
            api_key="your-api-key",
        )
        
        await worker.start()
        await worker.run_episode()
        await worker.stop()
        ```
    """
    
    def __init__(
        self,
        environment: EnvironmentAdapter,
        server_url: str,
        api_key: Optional[str] = None,
        worker_id: Optional[str] = None,
        use_remote_inference: bool = False,
        policy_poll_interval: float = 60.0,
        trace_buffer_size: int = 10,
        retry_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Remote Environment Worker.
        
        Args:
            environment: Local EnvironmentAdapter instance
            server_url: Base URL of PRIME-RL Orchestrator API
            api_key: Optional API key for authentication
            worker_id: Optional worker identifier (generated if not provided)
            use_remote_inference: If True, use server-side inference (Inference-as-a-Service)
            policy_poll_interval: Interval in seconds to poll for policy updates
            trace_buffer_size: Number of traces to buffer before sending
            retry_config: Optional retry configuration
        """
        if not HTTP_AVAILABLE:
            raise ImportError(
                "HTTP libraries required for RemoteEnvironmentWorker. Install with: pip install requests aiohttp"
            )
        
        self.environment = environment
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.worker_id = worker_id
        self.use_remote_inference = use_remote_inference
        self.policy_poll_interval = policy_poll_interval
        self.trace_buffer_size = trace_buffer_size
        self.retry_config = retry_config or {}
        
        # Session state
        self.session_id: Optional[str] = None
        self.current_policy_version: Optional[str] = None
        self.is_running = False
        
        # Trace buffer
        self.trace_buffer: List[UniversalRollout] = []
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Headers
        self.headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    async def _ensure_session(self):
        """Ensure HTTP session is created."""
        if self._session is None:
            self._session = aiohttp.ClientSession(headers=self.headers)
    
    async def start(self) -> None:
        """
        Start the worker and register with the server.
        
        Raises:
            RuntimeError: If registration fails
        """
        await self._ensure_session()
        
        # Register with server
        registration = WorkerRegistrationRequest(
            worker_id=self.worker_id,
            environment_id=getattr(self.environment, "environment_id", "unknown"),
            worker_type="environment",
            capabilities=["local_inference"] if not self.use_remote_inference else [],
        )
        
        async def _register():
            async with self._session.post(
                f"{self.server_url}/v1/workers/register",
                json=registration.model_dump(),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return WorkerRegistrationResponse(**data)
        
        retry_cfg = RetryConfig(**self.retry_config)
        try:
            registration_response = await retry_with_backoff_async(_register, retry_config=retry_cfg)
            self.session_id = registration_response.session_id
            self.current_policy_version = registration_response.policy_version
            logger.info(f"Worker registered: session_id={self.session_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to register worker: {e}") from e
        
        self.is_running = True
    
    async def stop(self) -> None:
        """Stop the worker and flush trace buffer."""
        if not self.is_running:
            return
        
        # Flush trace buffer
        await self.flush_traces()
        
        self.is_running = False
        
        # Close HTTP session
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info("Worker stopped")
    
    async def get_latest_policy(self) -> Optional[PolicyResponse]:
        """
        Get latest policy from server.
        
        Returns:
            PolicyResponse if policy is available, None otherwise
        """
        if not self.session_id:
            raise RuntimeError("Worker not registered. Call start() first.")
        
        await self._ensure_session()
        
        request = PolicyRequest(
            session_id=self.session_id,
            current_version=self.current_policy_version,
        )
        
        async def _get_policy():
            async with self._session.get(
                f"{self.server_url}/v1/policy/latest",
                params=request.model_dump(),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return PolicyResponse(**data)
        
        retry_cfg = RetryConfig(**self.retry_config)
        try:
            policy_response = await retry_with_backoff_async(_get_policy, retry_config=retry_cfg)
            if policy_response.has_update:
                self.current_policy_version = policy_response.version
                logger.info(f"Policy updated: version={policy_response.version}")
            return policy_response
        except Exception as e:
            logger.warning(f"Failed to get policy: {e}")
            return None
    
    async def get_action(self, observation: Any, trace_id: Optional[str] = None) -> Any:
        """
        Get action from server (Inference-as-a-Service mode).
        
        Args:
            observation: Environment observation
            trace_id: Optional trace identifier
            
        Returns:
            Inferred action
        """
        if not self.session_id:
            raise RuntimeError("Worker not registered. Call start() first.")
        
        if not self.use_remote_inference:
            raise RuntimeError("Remote inference not enabled. Set use_remote_inference=True.")
        
        await self._ensure_session()
        
        request = ActionRequest(
            session_id=self.session_id,
            observation=observation,
            trace_id=trace_id,
        )
        
        async def _get_action():
            async with self._session.post(
                f"{self.server_url}/v1/actions",
                json=request.model_dump(),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return ActionResponse(**data)
        
        retry_cfg = RetryConfig(**self.retry_config)
        try:
            action_response = await retry_with_backoff_async(_get_action, retry_config=retry_cfg)
            return action_response.action
        except Exception as e:
            logger.error(f"Failed to get action: {e}")
            raise
    
    async def submit_trace(self, rollout: UniversalRollout, trace_id: Optional[str] = None) -> bool:
        """
        Submit a trace to the server.
        
        Args:
            rollout: UniversalRollout to submit
            trace_id: Optional trace identifier
            
        Returns:
            True if submission was successful
        """
        if not self.session_id:
            raise RuntimeError("Worker not registered. Call start() first.")
        
        await self._ensure_session()
        
        request = TraceSubmissionRequest(
            session_id=self.session_id,
            rollout=rollout.to_dict(),
            trace_id=trace_id,
        )
        
        async def _submit():
            async with self._session.post(
                f"{self.server_url}/v1/traces",
                json=request.model_dump(),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return TraceSubmissionResponse(**data)
        
        retry_cfg = RetryConfig(**self.retry_config)
        try:
            response = await retry_with_backoff_async(_submit, retry_config=retry_cfg)
            logger.debug(f"Trace submitted: trace_id={response.trace_id}")
            return response.success
        except Exception as e:
            logger.error(f"Failed to submit trace: {e}")
            return False
    
    async def flush_traces(self) -> None:
        """Flush trace buffer by submitting all buffered traces."""
        if not self.trace_buffer:
            return
        
        logger.info(f"Flushing {len(self.trace_buffer)} traces")
        
        for rollout in self.trace_buffer:
            await self.submit_trace(rollout)
        
        self.trace_buffer.clear()
    
    async def buffer_trace(self, rollout: UniversalRollout) -> None:
        """
        Buffer a trace for batch submission.
        
        Args:
            rollout: UniversalRollout to buffer
        """
        self.trace_buffer.append(rollout)
        
        # Flush if buffer is full
        if len(self.trace_buffer) >= self.trace_buffer_size:
            await self.flush_traces()
    
    async def run_episode(self, max_steps: Optional[int] = None) -> UniversalRollout:
        """
        Run a single episode and return the rollout.
        
        Args:
            max_steps: Optional maximum steps per episode
            
        Returns:
            UniversalRollout for the episode
        """
        if not self.is_running:
            raise RuntimeError("Worker not started. Call start() first.")
        
        # Reset environment
        obs, info = self.environment.reset()
        
        # Get rollout from adapter if available
        trajectory = []
        done = False
        step_count = 0
        
        while not done and (max_steps is None or step_count < max_steps):
            # Get action (local inference or remote)
            if self.use_remote_inference:
                action = await self.get_action(obs)
            else:
                # Local inference (would use policy model here)
                # For now, use a placeholder
                action = self._get_local_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.environment.step(action)
            done = terminated or truncated
            
            trajectory.append({
                "observation": obs,
                "action": action,
                "reward": reward,
                "done": done,
            })
            
            step_count += 1
        
        # Convert trajectory to UniversalRollout
        rollout = self._trajectory_to_rollout(trajectory)
        
        # Buffer or submit trace
        await self.buffer_trace(rollout)
        
        return rollout
    
    def _get_local_action(self, observation: Any) -> Any:
        """Get action using local policy (placeholder implementation)."""
        # In a real implementation, this would use a local policy model
        # For now, return a placeholder action
        return "placeholder_action"
    
    def _trajectory_to_rollout(self, trajectory: List[Dict[str, Any]]) -> UniversalRollout:
        """Convert trajectory to UniversalRollout."""
        if not trajectory:
            raise ValueError("Empty trajectory")
        
        # Extract prompt (first observation)
        prompt = str(trajectory[0]["observation"])
        
        # Extract completion (all actions)
        actions = [str(step["action"]) for step in trajectory]
        completion = " ".join(actions) if len(actions) > 1 else (actions[0] if actions else "")
        
        # Extract rewards
        rewards = [step["reward"] for step in trajectory]
        total_reward = sum(rewards)
        
        # Build rollout
        if len(trajectory) > 1:
            observations = [str(step["observation"]) for step in trajectory]
            actions_list = [str(step["action"]) for step in trajectory]
            dones = [step["done"] for step in trajectory]
        else:
            observations = None
            actions_list = None
            dones = None
        
        return UniversalRollout(
            prompts=[prompt],
            completions=[completion],
            rewards=[total_reward],
            observations=observations,
            actions=actions_list,
            dones=dones,
            metadata={"num_steps": len(trajectory)},
        )
    
    async def run_continuous(self, poll_policy: bool = True) -> None:
        """
        Run continuous loop: poll for policy updates and run episodes.
        
        Args:
            poll_policy: Whether to poll for policy updates
        """
        if not self.is_running:
            raise RuntimeError("Worker not started. Call start() first.")
        
        last_policy_poll = 0.0
        
        try:
            while self.is_running:
                # Poll for policy updates
                if poll_policy:
                    current_time = time.time()
                    if current_time - last_policy_poll >= self.policy_poll_interval:
                        await self.get_latest_policy()
                        last_policy_poll = current_time
                
                # Run episode
                await self.run_episode()
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")
        except Exception as e:
            logger.error(f"Error in continuous loop: {e}")
            raise
        finally:
            await self.stop()

