"""Unit tests for SurgeAIClient."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from prime_rl.integrations.marketplace.surge import SurgeAIClient
from prime_rl.core.interaction_trace import InteractionTrace


class TestSurgeAIClient:
    """Tests for SurgeAIClient."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = SurgeAIClient(api_key="test-key")
        
        assert client.api_key == "test-key"
        assert client.environment_id == "surge-ai-v1"
    
    def test_initialization_without_requests(self):
        """Test initialization fails without requests."""
        with patch("prime_rl.integrations.marketplace.surge.REQUESTS_AVAILABLE", False):
            with pytest.raises(ImportError):
                SurgeAIClient()
    
    def test_fetch_tasks(self):
        """Test fetching tasks (stub)."""
        client = SurgeAIClient(api_key="test-key")
        
        tasks = client.fetch_tasks(project_id="test-project", limit=5)
        
        assert isinstance(tasks, list)
        assert len(tasks) <= 5
        if len(tasks) > 0:
            assert "task_id" in tasks[0]
    
    def test_convert_task(self):
        """Test converting a task to InteractionTrace."""
        client = SurgeAIClient(api_key="test-key")
        
        task = {
            "task_id": "task-1",
            "project_id": "project-1",
            "status": "completed",
            "input": {"prompt": "Test prompt"},
            "output": {"completion": "Test completion"},
            "label": {"score": 0.8},
        }
        
        traces = client.convert([task])
        
        assert len(traces) == 1
        assert isinstance(traces[0], InteractionTrace)
        assert traces[0].trace_id == "task-1"
        assert len(traces[0].steps) > 0
    
    def test_convert_to_jsonl_csv(self):
        """Test converting CSV to JSONL."""
        client = SurgeAIClient(api_key="test-key")
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("task_id,project_id,status,input,output,score\n")
            f.write("task-1,project-1,completed,Test prompt,Test completion,0.8\n")
            csv_path = Path(f.name)
        
        try:
            # Create output path
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                output_path = Path(f.name)
            
            try:
                client.convert_to_jsonl(csv_path, output_path)
                
                # Verify output
                assert output_path.exists()
                with open(output_path, "r") as f:
                    lines = f.readlines()
                    assert len(lines) > 0
                    trace = json.loads(lines[0])
                    assert "trace_id" in trace
            finally:
                if output_path.exists():
                    output_path.unlink()
        finally:
            if csv_path.exists():
                csv_path.unlink()
    
    def test_convert_to_jsonl_json(self):
        """Test converting JSON to JSONL."""
        client = SurgeAIClient(api_key="test-key")
        
        # Create temporary JSON
        tasks = [
            {
                "task_id": "task-1",
                "project_id": "project-1",
                "status": "completed",
                "input": {"prompt": "Test prompt"},
                "output": {"completion": "Test completion"},
                "label": {"score": 0.8},
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(tasks, f)
            json_path = Path(f.name)
        
        try:
            # Create output path
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                output_path = Path(f.name)
            
            try:
                client.convert_to_jsonl(json_path, output_path)
                
                # Verify output
                assert output_path.exists()
                with open(output_path, "r") as f:
                    lines = f.readlines()
                    assert len(lines) > 0
            finally:
                if output_path.exists():
                    output_path.unlink()
        finally:
            if json_path.exists():
                json_path.unlink()
    
    def test_close(self):
        """Test closing client."""
        client = SurgeAIClient(api_key="test-key")
        
        client.close()
        
        # Should not raise
        assert True

