"""Unit tests for WorldModelEnvAdapter."""

import pytest
from pathlib import Path
from unittest.mock import Mock

from prime_rl.world_models.base import WorldModelConfig, WorldModel
from prime_rl.world_models.env_adapter import WorldModelEnvAdapter


class MockWorldModel(WorldModel):
    """Mock world model for testing."""
    
    def predict_next(self, state_batch, action_batch):
        return (
            [{"ticket": {"status": "resolved"}}],
            [1.0],
            [True],
            {},
        )
    
    def train_step(self, batch):
        return {"loss": 0.0}
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def sample_initial_state(self, num_samples=1):
        return [{"ticket": {"status": "open"}}] * num_samples


class TestWorldModelEnvAdapter:
    """Tests for WorldModelEnvAdapter."""
    
    def test_initialization_with_model(self):
        """Test initialization with world model."""
        config = WorldModelConfig(
            algorithm="test",
            state_representation="crm_structured",
        )
        model = MockWorldModel(config)
        
        env = WorldModelEnvAdapter(world_model=model, config=config)
        assert env.world_model == model
        assert env.config == config
    
    def test_initialization_with_config(self):
        """Test initialization with config only."""
        config = WorldModelConfig(
            algorithm="test",
            state_representation="crm_structured",
        )
        
        # Register mock model
        from prime_rl.world_models.base import register_world_model
        register_world_model("test", MockWorldModel)
        
        env = WorldModelEnvAdapter(config=config)
        assert env.world_model is not None
    
    def test_reset(self):
        """Test reset method."""
        config = WorldModelConfig(
            algorithm="test",
            state_representation="crm_structured",
        )
        model = MockWorldModel(config)
        env = WorldModelEnvAdapter(world_model=model, config=config)
        
        obs, info = env.reset()
        assert obs is not None
        assert "world_model" in info
        assert env._episode_started is True
    
    def test_step(self):
        """Test step method."""
        config = WorldModelConfig(
            algorithm="test",
            state_representation="crm_structured",
        )
        model = MockWorldModel(config)
        env = WorldModelEnvAdapter(world_model=model, config=config)
        
        env.reset()
        obs, reward, done, truncated, info = env.step({"action_type": "reply_to_ticket"})
        
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
    
    def test_step_before_reset(self):
        """Test step before reset raises error."""
        config = WorldModelConfig(
            algorithm="test",
            state_representation="crm_structured",
        )
        model = MockWorldModel(config)
        env = WorldModelEnvAdapter(world_model=model, config=config)
        
        with pytest.raises(RuntimeError, match="must be reset"):
            env.step({})
    
    def test_close(self):
        """Test close method."""
        config = WorldModelConfig(
            algorithm="test",
            state_representation="crm_structured",
        )
        model = MockWorldModel(config)
        env = WorldModelEnvAdapter(world_model=model, config=config)
        
        env.reset()
        env.close()
        
        assert env._current_state is None
        assert env._episode_started is False

