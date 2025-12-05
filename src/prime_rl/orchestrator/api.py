"""
Worker API for Forward Deployment Architecture.

This module implements the FastAPI endpoints for the PRIME-RL Orchestrator
to accept traces from remote workers and serve policy updates.
"""

from typing import Dict, Optional, Any
from datetime import datetime
import os

try:
    from fastapi import FastAPI, HTTPException, Depends, Header
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    Depends = None  # type: ignore
    Header = None  # type: ignore
    HTTPBearer = None  # type: ignore
    HTTPAuthorizationCredentials = None  # type: ignore

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
from loguru import logger


# Global state (in production, use proper state management)
_worker_sessions: Dict[str, Dict[str, Any]] = {}
_trace_buffer: list = []
_current_policy_version: str = "v1.0.0"


def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> bool:
    """
    Verify API key from Authorization header.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        True if valid, raises HTTPException otherwise
    """
    api_key = os.getenv("PRIME_RL_API_KEY")
    
    # If no API key is set, allow all requests (development mode)
    if not api_key:
        logger.warning("PRIME_RL_API_KEY not set, allowing all requests (development mode)")
        return True
    
    # If credentials not provided, reject
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    # Verify token
    if credentials.credentials != api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return True


def create_worker_api() -> FastAPI:
    """
    Create and configure the Worker API FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI required for Worker API. Install with: pip install fastapi uvicorn"
        )
    
    app = FastAPI(
        title="PRIME-RL Worker API",
        description="API for remote workers to submit traces and receive policy updates",
        version="1.0.0",
    )
    
    @app.post("/v1/workers/register", response_model=WorkerRegistrationResponse)
    async def register_worker(
        request: WorkerRegistrationRequest,
        _: bool = Depends(verify_api_key),
    ) -> WorkerRegistrationResponse:
        """
        Register a new worker session.
        
        Args:
            request: Worker registration request
            
        Returns:
            Registration response with session ID
        """
        import uuid
        
        session_id = str(uuid.uuid4())
        
        # Store session
        _worker_sessions[session_id] = {
            "worker_id": request.worker_id,
            "environment_id": request.environment_id,
            "worker_type": request.worker_type,
            "capabilities": request.capabilities,
            "registered_at": datetime.utcnow(),
            "metadata": request.metadata,
        }
        
        logger.info(f"Worker registered: worker_id={request.worker_id}, session_id={session_id}")
        
        return WorkerRegistrationResponse(
            session_id=session_id,
            server_time=datetime.utcnow(),
            policy_version=_current_policy_version,
            config={
                "trace_buffer_size": 10,
                "policy_poll_interval": 60.0,
            },
        )
    
    @app.post("/v1/traces", response_model=TraceSubmissionResponse)
    async def submit_trace(
        request: TraceSubmissionRequest,
        _: bool = Depends(verify_api_key),
    ) -> TraceSubmissionResponse:
        """
        Submit a trace (UniversalRollout) to the server.
        
        Args:
            request: Trace submission request
            
        Returns:
            Submission response
        """
        # Verify session
        if request.session_id not in _worker_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Generate trace ID if not provided
        trace_id = request.trace_id or f"trace-{datetime.utcnow().timestamp()}"
        
        # Store trace (in production, would write to buffer/database)
        trace_entry = {
            "trace_id": trace_id,
            "session_id": request.session_id,
            "rollout": request.rollout,
            "metadata": request.metadata,
            "received_at": datetime.utcnow(),
        }
        _trace_buffer.append(trace_entry)
        
        logger.info(f"Trace received: trace_id={trace_id}, session_id={request.session_id}")
        
        return TraceSubmissionResponse(
            success=True,
            trace_id=trace_id,
            server_time=datetime.utcnow(),
            message="Trace received",
        )
    
    @app.get("/v1/policy/latest", response_model=PolicyResponse)
    async def get_latest_policy(
        session_id: str,
        current_version: Optional[str] = None,
        _: bool = Depends(verify_api_key),
    ) -> PolicyResponse:
        """
        Get latest policy weights/checkpoint.
        
        Args:
            session_id: Worker session identifier
            current_version: Current policy version (for caching)
            
        Returns:
            Policy response with checkpoint information
        """
        # Verify session
        if session_id not in _worker_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if policy has been updated
        has_update = current_version != _current_policy_version
        
        # In production, would return actual checkpoint path/URL
        # For now, return stub information
        return PolicyResponse(
            version=_current_policy_version,
            checkpoint_path=None,  # Would be actual path in production
            checkpoint_url=None,  # Would be actual URL in production
            lora_adapters=None,
            metadata={"step": 0},  # Would be actual step in production
            has_update=has_update,
        )
    
    @app.post("/v1/actions", response_model=ActionResponse)
    async def get_action(
        request: ActionRequest,
        _: bool = Depends(verify_api_key),
    ) -> ActionResponse:
        """
        Get action inference (Inference-as-a-Service mode).
        
        Args:
            request: Action request with observation
            
        Returns:
            Action response with inferred action
            
        Note: This is a stub implementation. In production, would call
        inference service to get action from policy model.
        """
        # Verify session
        if request.session_id not in _worker_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Stub: return placeholder action
        # In production, would:
        # 1. Call inference service with observation
        # 2. Get action from policy model
        # 3. Return action with logprobs
        
        logger.warning("Action inference not implemented (stub)")
        
        return ActionResponse(
            action="placeholder_action",  # Would be actual action in production
            logprobs=None,  # Would be actual logprobs in production
            metadata={},
        )
    
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "active_sessions": len(_worker_sessions),
            "buffered_traces": len(_trace_buffer),
        }
    
    return app


# Global API instance (can be imported and used)
app = create_worker_api() if FASTAPI_AVAILABLE else None

