"""
TrainingRun abstraction for RLaaS multi-tenancy.

This module implements the TrainingRun abstraction with run_id/project_id for
multi-tenancy support as specified in the PRD (§2.3, §4.3).
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class TrainingRun:
    """
    Represents a single training run with metadata and state.
    
    This abstraction enables multi-tenancy by scoping all state by project_id/run_id.
    It can be backed by local filesystem (Phase 3) or remote database (future).
    
    Attributes:
        run_id: Unique identifier for this training run (UUID)
        project_id: Project identifier for multi-tenancy
        config: Training configuration (UnifiedConfig dict)
        code_revision: Git commit hash or code version
        dataset_reference: Reference to dataset (path or URI)
        environment_reference: Reference to environment (ID or URI)
        status: Current status ("pending", "running", "completed", "failed", "cancelled")
        created_at: Timestamp when run was created
        started_at: Timestamp when run started
        completed_at: Timestamp when run completed
        metadata: Additional metadata dictionary
    """
    
    run_id: str
    project_id: str
    config: Dict[str, Any]
    code_revision: Optional[str] = None
    dataset_reference: Optional[str] = None
    environment_reference: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        project_id: str,
        config: Dict[str, Any],
        code_revision: Optional[str] = None,
        dataset_reference: Optional[str] = None,
        environment_reference: Optional[str] = None,
    ) -> "TrainingRun":
        """
        Create a new TrainingRun.
        
        Args:
            project_id: Project identifier
            config: Training configuration
            code_revision: Optional code revision/commit hash
            dataset_reference: Optional dataset reference
            environment_reference: Optional environment reference
            
        Returns:
            New TrainingRun instance
        """
        return cls(
            run_id=str(uuid.uuid4()),
            project_id=project_id,
            config=config,
            code_revision=code_revision,
            dataset_reference=dataset_reference,
            environment_reference=environment_reference,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TrainingRun to dictionary."""
        return {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "config": self.config,
            "code_revision": self.code_revision,
            "dataset_reference": self.dataset_reference,
            "environment_reference": self.environment_reference,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingRun":
        """Create TrainingRun from dictionary."""
        return cls(
            run_id=data["run_id"],
            project_id=data["project_id"],
            config=data["config"],
            code_revision=data.get("code_revision"),
            dataset_reference=data.get("dataset_reference"),
            environment_reference=data.get("environment_reference"),
            status=data.get("status", "pending"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now()),
            started_at=datetime.fromisoformat(data["started_at"]) if isinstance(data.get("started_at"), str) else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if isinstance(data.get("completed_at"), str) else None,
            metadata=data.get("metadata", {}),
        )
    
    def mark_started(self) -> None:
        """Mark the run as started."""
        self.status = "running"
        self.started_at = datetime.now()
    
    def mark_completed(self) -> None:
        """Mark the run as completed."""
        self.status = "completed"
        self.completed_at = datetime.now()
    
    def mark_failed(self, error: Optional[str] = None) -> None:
        """Mark the run as failed."""
        self.status = "failed"
        self.completed_at = datetime.now()
        if error:
            self.metadata["error"] = error
    
    def mark_cancelled(self) -> None:
        """Mark the run as cancelled."""
        self.status = "cancelled"
        self.completed_at = datetime.now()

