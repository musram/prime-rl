"""
Unified configuration schema for PRIME-RL.

This module defines the unified TOML/YAML configuration schema that selects backend,
mode, algorithm, and dataset/environment as specified in the PRD (§3.1).
"""

from typing import Annotated, Literal, Optional
from pathlib import Path
from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    """Backend configuration (JAX or PyTorch)."""
    type: Literal["jax", "torch"] = "jax"
    mode: Literal["offline", "online"] = "offline"


class DatasetConfig(BaseModel):
    """Dataset configuration for offline RL."""
    path: Path = Field(..., description="Path to dataset file (JSONL)")
    schema_version: str = Field(default="v1", description="Schema version")
    batch_size: int = Field(default=32, ge=1, description="Batch size")
    shuffle: bool = Field(default=True, description="Whether to shuffle data")


class AlgorithmConfig(BaseModel):
    """Algorithm configuration."""
    name: Literal["dpo", "cql", "ppo", "grpo"] = "dpo"
    learning_rate: float = Field(default=1e-5, ge=0, description="Learning rate")
    # Algorithm-specific config can be extended
    beta: Optional[float] = Field(default=None, description="DPO beta parameter")


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = Field(..., description="Model name (e.g., HuggingFace model ID)")
    trust_remote_code: bool = Field(default=False, description="Trust remote code")


class OutputConfig(BaseModel):
    """Output configuration."""
    output_dir: Path = Field(default=Path("./output"), description="Output directory for checkpoints/logs")
    max_steps: Optional[int] = Field(default=None, ge=1, description="Maximum training steps")
    checkpoint_every: Optional[int] = Field(default=None, ge=1, description="Checkpoint every N steps")
    eval_every: Optional[int] = Field(default=None, ge=1, description="Evaluate every N steps")


class UnifiedConfig(BaseModel):
    """
    Unified configuration schema for PRIME-RL.
    
    This configuration selects backend, mode, algorithm, and dataset/environment
    as specified in the PRD (§3.1).
    """
    backend: BackendConfig = Field(default_factory=BackendConfig)
    dataset: Optional[DatasetConfig] = Field(default=None, description="Dataset config (for offline mode)")
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    model: Optional[ModelConfig] = Field(default=None, description="Model config")
    output: Optional[OutputConfig] = Field(default=None, description="Output configuration")
    
    # Convenience properties for backward compatibility
    @property
    def output_dir(self) -> Path:
        """Get output directory."""
        return self.output.output_dir if self.output else Path("./output")
    
    @property
    def max_steps(self) -> Optional[int]:
        """Get max steps."""
        return self.output.max_steps if self.output else None
    
    @property
    def checkpoint_every(self) -> Optional[int]:
        """Get checkpoint interval."""
        return self.output.checkpoint_every if self.output else None
    
    @property
    def eval_every(self) -> Optional[int]:
        """Get eval interval."""
        return self.output.eval_every if self.output else None

    class Config:
        """Pydantic config."""
        extra = "forbid"

