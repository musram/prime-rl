"""Unit tests for MercorClient."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from prime_rl.integrations.marketplace.mercor import MercorClient
from prime_rl.core.interaction_trace import InteractionTrace


class TestMercorClient:
    """Tests for MercorClient."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = MercorClient(api_key="test-key")
        
        assert client.api_key == "test-key"
        assert client.environment_id == "mercor-v1"
    
    def test_initialization_without_requests(self):
        """Test initialization fails without requests."""
        with patch("prime_rl.integrations.marketplace.mercor.REQUESTS_AVAILABLE", False):
            with pytest.raises(ImportError):
                MercorClient()
    
    def test_fetch_demonstrations(self):
        """Test fetching demonstrations (stub)."""
        client = MercorClient(api_key="test-key")
        
        demos = client.fetch_demonstrations(expert_id="expert-1", limit=5)
        
        assert isinstance(demos, list)
        assert len(demos) <= 5
        if len(demos) > 0:
            assert "demo_id" in demos[0]
    
    def test_convert_demonstration_single_step(self):
        """Test converting single-step demonstration."""
        client = MercorClient(api_key="test-key")
        
        demo = {
            "demo_id": "demo-1",
            "expert_id": "expert-1",
            "task_type": "coding",
            "prompt": "Write a function",
            "solution": "def func(): pass",
            "quality_score": 0.9,
        }
        
        traces = client.convert([demo])
        
        assert len(traces) == 1
        assert isinstance(traces[0], InteractionTrace)
        assert traces[0].trace_id == "demo-1"
        assert len(traces[0].steps) == 1
    
    def test_convert_demonstration_multi_step(self):
        """Test converting multi-step demonstration."""
        client = MercorClient(api_key="test-key")
        
        demo = {
            "demo_id": "demo-1",
            "expert_id": "expert-1",
            "task_type": "coding",
            "prompt": "Write a function",
            "solution": "def func(): pass",
            "steps": [
                {"action": "think", "content": "I need to..."},
                {"action": "code", "content": "def func():"},
            ],
            "quality_score": 0.9,
        }
        
        traces = client.convert([demo])
        
        assert len(traces) == 1
        assert len(traces[0].steps) > 1
    
    def test_convert_to_jsonl(self):
        """Test converting JSON to JSONL."""
        client = MercorClient(api_key="test-key")
        
        # Create temporary JSON
        demos = [
            {
                "demo_id": "demo-1",
                "expert_id": "expert-1",
                "task_type": "coding",
                "prompt": "Write a function",
                "solution": "def func(): pass",
                "quality_score": 0.9,
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(demos, f)
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
                    trace = json.loads(lines[0])
                    assert "trace_id" in trace
            finally:
                if output_path.exists():
                    output_path.unlink()
        finally:
            if json_path.exists():
                json_path.unlink()
    
    def test_close(self):
        """Test closing client."""
        client = MercorClient(api_key="test-key")
        
        client.close()
        
        # Should not raise
        assert True

