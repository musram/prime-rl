"""Tests for TrainingRun abstraction."""

import pytest
from datetime import datetime

from prime_rl.core.training_run import TrainingRun


def test_training_run_creation():
    """Test TrainingRun creation."""
    run = TrainingRun.create(
        project_id="test-project",
        config={"backend": {"type": "jax", "mode": "offline"}},
        code_revision="abc123",
        dataset_reference="data/traces.jsonl",
    )
    assert run.project_id == "test-project"
    assert run.run_id is not None
    assert run.status == "pending"
    assert run.code_revision == "abc123"
    assert run.dataset_reference == "data/traces.jsonl"


def test_training_run_status_transitions():
    """Test TrainingRun status transitions."""
    run = TrainingRun.create(
        project_id="test-project",
        config={},
    )
    
    assert run.status == "pending"
    assert run.started_at is None
    
    run.mark_started()
    assert run.status == "running"
    assert run.started_at is not None
    
    run.mark_completed()
    assert run.status == "completed"
    assert run.completed_at is not None


def test_training_run_serialization():
    """Test TrainingRun serialization to/from dict."""
    run = TrainingRun.create(
        project_id="test-project",
        config={"key": "value"},
    )
    run.mark_started()
    
    data = run.to_dict()
    assert data["project_id"] == "test-project"
    assert data["status"] == "running"
    assert "created_at" in data
    
    restored = TrainingRun.from_dict(data)
    assert restored.project_id == run.project_id
    assert restored.run_id == run.run_id
    assert restored.status == run.status

