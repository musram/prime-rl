"""Tests for MetadataStore."""

import pytest
import tempfile
from pathlib import Path

from prime_rl.core.training_run import TrainingRun
from prime_rl.core.metadata_store import MetadataStore


def test_metadata_store_save_load():
    """Test saving and loading runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MetadataStore(Path(tmpdir))
        
        run = TrainingRun.create(
            project_id="test-project",
            config={"key": "value"},
        )
        
        store.save(run)
        
        loaded = store.load("test-project", run.run_id)
        assert loaded is not None
        assert loaded.run_id == run.run_id
        assert loaded.project_id == run.project_id


def test_metadata_store_list_runs():
    """Test listing runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MetadataStore(Path(tmpdir))
        
        # Create multiple runs
        runs = []
        for i in range(3):
            run = TrainingRun.create(
                project_id="test-project",
                config={"index": i},
            )
            if i == 1:
                run.mark_completed()
            store.save(run)
            runs.append(run)
        
        # List all runs
        all_runs = store.list_runs("test-project")
        assert len(all_runs) == 3
        
        # List completed runs
        completed_runs = store.list_runs("test-project", status="completed")
        assert len(completed_runs) == 1
        assert completed_runs[0].run_id == runs[1].run_id


def test_metadata_store_update_delete():
    """Test updating and deleting runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MetadataStore(Path(tmpdir))
        
        run = TrainingRun.create(
            project_id="test-project",
            config={},
        )
        store.save(run)
        
        # Update
        run.mark_started()
        store.update(run)
        
        loaded = store.load("test-project", run.run_id)
        assert loaded.status == "running"
        
        # Delete
        deleted = store.delete("test-project", run.run_id)
        assert deleted is True
        
        loaded = store.load("test-project", run.run_id)
        assert loaded is None

