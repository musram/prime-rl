"""Tests for PRIME-RL Python SDK."""

import pytest
import tempfile
from pathlib import Path

from prime_rl.core.sdk import PRIMERLClient
from prime_rl.core.training_run import TrainingRun


def test_sdk_client_initialization():
    """Test SDK client initialization."""
    client = PRIMERLClient(project_id="test-project")
    assert client.project_id == "test-project"
    assert client.metadata_store is not None


def test_sdk_create_run():
    """Test creating a run via SDK."""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = PRIMERLClient(
            project_id="test-project",
            metadata_dir=Path(tmpdir),
        )
        
        # Create a dummy config file
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text("""
[backend]
type = "jax"
mode = "offline"

[dataset]
path = "data/traces.jsonl"
""")
        
        run = client.create_run(config_file)
        assert run.project_id == "test-project"
        assert run.run_id is not None
        
        # Verify it's saved
        loaded = client.get_run(run.run_id)
        assert loaded is not None
        assert loaded.run_id == run.run_id


def test_sdk_list_runs():
    """Test listing runs via SDK."""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = PRIMERLClient(
            project_id="test-project",
            metadata_dir=Path(tmpdir),
        )
        
        # Create multiple runs
        config_file = Path(tmpdir) / "config.toml"
        config_file.write_text("[backend]\ntype = \"jax\"\nmode = \"offline\"\n")
        
        runs = []
        for _ in range(3):
            run = client.create_run(config_file)
            runs.append(run)
        
        # List all runs
        all_runs = client.list_runs()
        assert len(all_runs) >= 3
        
        # List with limit
        limited_runs = client.list_runs(limit=2)
        assert len(limited_runs) <= 2

