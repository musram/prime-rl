"""Unit tests for EpicAdapter."""

import pytest
from prime_rl.integrations.enterprise.epic import EpicAdapter
from prime_rl.core.algorithms import UniversalRollout


class TestEpicAdapter:
    """Tests for EpicAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = EpicAdapter(mock_mode=True)
        
        assert adapter.environment_id == "epic-ehr-v1"
        assert adapter.mock_mode is True
        assert adapter._current_patient is None
    
    def test_reset(self):
        """Test environment reset."""
        adapter = EpicAdapter(mock_mode=True)
        
        obs, info = adapter.reset()
        
        assert obs is not None
        assert "patient_id" in obs or "summary" in obs
        assert info["environment_id"] == "epic-ehr-v1"
        assert adapter._episode_started is True
    
    def test_reset_with_patient_id(self):
        """Test reset with specific patient ID."""
        adapter = EpicAdapter(mock_mode=True)
        
        obs, info = adapter.reset(options={"patient_id": "12345"})
        
        assert adapter._current_patient is not None
        assert adapter._current_patient["patient_id"] == "12345"
    
    def test_step_lookup_patient(self):
        """Test lookup_patient action."""
        adapter = EpicAdapter(mock_mode=True)
        adapter.reset()
        
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "lookup_patient",
            "patient_id": "12345",
        })
        
        assert reward > 0
        assert "status" in obs or "patient_id" in obs
    
    def test_step_review_chart(self):
        """Test review_chart action."""
        adapter = EpicAdapter(mock_mode=True)
        adapter.reset()
        
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "review_chart",
        })
        
        assert reward > 0
        assert "chart" in obs or "patient" in obs
    
    def test_step_enter_order(self):
        """Test enter_order action."""
        adapter = EpicAdapter(mock_mode=True)
        adapter.reset()
        
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "enter_order",
            "order_type": "lab",
            "details": {"test": "blood_work"},
        })
        
        assert reward > 0
        assert "order_entered" in obs or "status" in obs
    
    def test_step_document_note(self):
        """Test document_note action."""
        adapter = EpicAdapter(mock_mode=True)
        adapter.reset()
        
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "document_note",
            "note": "Patient is doing well.",
        })
        
        assert reward > 0
    
    def test_step_complete(self):
        """Test complete action terminates episode."""
        adapter = EpicAdapter(mock_mode=True)
        adapter.reset()
        
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "complete",
        })
        
        assert terminated is True
        assert reward > 0
    
    def test_get_rollout(self):
        """Test converting trajectory to UniversalRollout."""
        adapter = EpicAdapter(mock_mode=True)
        adapter.reset()
        
        # Execute some steps
        adapter.step({"action_type": "review_chart"})
        adapter.step({"action_type": "enter_order", "order_type": "lab"})
        
        rollout = adapter.get_rollout()
        
        assert isinstance(rollout, UniversalRollout)
        assert len(rollout.prompts) > 0
        assert len(rollout.completions) > 0
        assert len(rollout.rewards) > 0
    
    def test_render(self):
        """Test rendering environment state."""
        adapter = EpicAdapter(mock_mode=True)
        adapter.reset()
        
        state = adapter.render()
        
        assert state is not None
        assert "patient" in state or "chart" in state
    
    def test_close(self):
        """Test closing environment."""
        adapter = EpicAdapter(mock_mode=True)
        adapter.reset()
        
        adapter.close()
        
        assert adapter._current_patient is None
        assert adapter._episode_started is False

