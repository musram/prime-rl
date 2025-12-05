"""
Remote Worker Protocol for Forward Deployment Architecture.

This module defines the Pydantic models for the Worker API protocol,
enabling client-server communication between remote workers and the
PRIME-RL Orchestrator cluster.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

from prime_rl.core.algorithms import UniversalRollout


class WorkerRegistrationRequest(BaseModel):
    """Request to register a new worker session."""
    
    worker_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique worker identifier")
    environment_id: str = Field(..., description="Environment identifier (e.g., 'browser-gym-v1')")
    worker_type: str = Field(default="environment", description="Worker type: 'environment' or 'inference'")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional worker metadata")
    capabilities: List[str] = Field(default_factory=list, description="Worker capabilities (e.g., ['local_inference'])")
    
    class Config:
        json_schema_extra = {
            "example": {
                "worker_id": "worker-123",
                "environment_id": "browser-gym-v1",
                "worker_type": "environment",
                "metadata": {"hostname": "laptop-001"},
                "capabilities": ["local_inference"],
            }
        }


class WorkerRegistrationResponse(BaseModel):
    """Response to worker registration."""
    
    session_id: str = Field(..., description="Session identifier for this worker")
    server_time: datetime = Field(default_factory=datetime.utcnow, description="Server timestamp")
    policy_version: Optional[str] = Field(default=None, description="Current policy version")
    config: Dict[str, Any] = Field(default_factory=dict, description="Server configuration for worker")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-abc-123",
                "server_time": "2025-01-01T12:00:00Z",
                "policy_version": "v1.0.0",
                "config": {"trace_buffer_size": 100},
            }
        }


class TraceSubmissionRequest(BaseModel):
    """Request to submit a trace (UniversalRollout) to the server."""
    
    session_id: str = Field(..., description="Worker session identifier")
    rollout: Dict[str, Any] = Field(..., description="UniversalRollout as dictionary")
    trace_id: Optional[str] = Field(default=None, description="Optional trace identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional trace metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-abc-123",
                "rollout": {
                    "prompts": ["test prompt"],
                    "completions": ["test completion"],
                    "rewards": [1.0],
                },
                "trace_id": "trace-xyz",
                "metadata": {},
            }
        }


class TraceSubmissionResponse(BaseModel):
    """Response to trace submission."""
    
    success: bool = Field(..., description="Whether submission was successful")
    trace_id: str = Field(..., description="Trace identifier (generated if not provided)")
    server_time: datetime = Field(default_factory=datetime.utcnow, description="Server timestamp")
    message: Optional[str] = Field(default=None, description="Optional message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "trace_id": "trace-xyz",
                "server_time": "2025-01-01T12:00:00Z",
                "message": "Trace received",
            }
        }


class PolicyRequest(BaseModel):
    """Request for latest policy weights."""
    
    session_id: str = Field(..., description="Worker session identifier")
    current_version: Optional[str] = Field(default=None, description="Current policy version (for caching)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-abc-123",
                "current_version": "v1.0.0",
            }
        }


class PolicyResponse(BaseModel):
    """Response with policy information."""
    
    version: str = Field(..., description="Policy version")
    checkpoint_path: Optional[str] = Field(default=None, description="Path to checkpoint (if available)")
    checkpoint_url: Optional[str] = Field(default=None, description="URL to download checkpoint")
    lora_adapters: Optional[Dict[str, str]] = Field(default=None, description="LoRA adapter paths/URLs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Policy metadata")
    has_update: bool = Field(default=True, description="Whether policy has been updated")
    
    class Config:
        json_schema_extra = {
            "example": {
                "version": "v1.0.1",
                "checkpoint_path": "/checkpoints/v1.0.1",
                "checkpoint_url": "https://server.com/checkpoints/v1.0.1",
                "has_update": True,
                "metadata": {"step": 1000},
            }
        }


class ActionRequest(BaseModel):
    """Request for action inference (Inference-as-a-Service mode)."""
    
    session_id: str = Field(..., description="Worker session identifier")
    observation: Any = Field(..., description="Environment observation")
    trace_id: Optional[str] = Field(default=None, description="Optional trace identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-abc-123",
                "observation": "current state",
                "trace_id": "trace-xyz",
                "metadata": {},
            }
        }


class ActionResponse(BaseModel):
    """Response with inferred action."""
    
    action: Any = Field(..., description="Inferred action")
    logprobs: Optional[List[float]] = Field(default=None, description="Action log probabilities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": "click(button_id)",
                "logprobs": [0.9, 0.1],
                "metadata": {},
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid session ID",
                "error_code": "INVALID_SESSION",
                "details": {"session_id": "invalid"},
            }
        }

