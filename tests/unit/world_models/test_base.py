"""Unit tests for world model base classes."""

import pytest
from pathlib import Path
from unittest.mock import Mock

from prime_rl.world_models.base import (
    WorldModel,
    WorldModelConfig,
    WORLD_MODEL_REGISTRY,
    create_world_model,
    register_world_model,
)


class DummyWorldModel(WorldModel):
    """Dummy world model for testing."""
    
    def predict_next(self, state_batch, action_batch):
        return ([{}], [0.0], [False], {})
    
    def train_step(self, batch):
        return {"loss": 0.0}
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def sample_initial_state(self, num_samples=1):
        return [{}] * num_samples


class TestWorldModelConfig:
    """Tests for WorldModelConfig."""
    
    def test_config_creation(self):
        """Test creating a config."""
        config = WorldModelConfig(
            algorithm="test_algorithm",
            state_representation="test_rep",
        )
        assert config.algorithm == "test_algorithm"
        assert config.state_representation == "test_rep"
    
    def test_config_with_checkpoint(self):
        """Test config with checkpoint path."""
        config = WorldModelConfig(
            algorithm="test",
            state_representation="test",
            checkpoint_path=Path("test.pt"),
        )
        assert config.checkpoint_path == Path("test.pt")


class TestWorldModelRegistry:
    """Tests for world model registry."""
    
    def test_register_world_model(self):
        """Test registering a world model."""
        register_world_model("test_model", DummyWorldModel)
        assert "test_model" in WORLD_MODEL_REGISTRY
        assert WORLD_MODEL_REGISTRY["test_model"] == DummyWorldModel
    
    def test_create_world_model(self):
        """Test creating a world model from config."""
        register_world_model("test_model", DummyWorldModel)
        
        config = WorldModelConfig(
            algorithm="test_model",
            state_representation="test",
        )
        
        model = create_world_model(config)
        assert isinstance(model, DummyWorldModel)
        assert model.config == config
    
    def test_create_unknown_model(self):
        """Test creating unknown model raises error."""
        config = WorldModelConfig(
            algorithm="unknown_model",
            state_representation="test",
        )
        
        with pytest.raises(ValueError, match="Unknown world model algorithm"):
            create_world_model(config)


class TestDummyWorldModel:
    """Tests for dummy world model."""
    
    def test_predict_next(self):
        """Test predict_next method."""
        config = WorldModelConfig(
            algorithm="test",
            state_representation="test",
        )
        model = DummyWorldModel(config)
        
        next_states, rewards, dones, info = model.predict_next([{}], [{}])
        assert len(next_states) == 1
        assert len(rewards) == 1
        assert len(dones) == 1
    
    def test_train_step(self):
        """Test train_step method."""
        config = WorldModelConfig(
            algorithm="test",
            state_representation="test",
        )
        model = DummyWorldModel(config)
        
        batch = {
            "state": [[0.0]],
            "action": [[0.0]],
            "next_state": [[0.0]],
            "reward": [0.0],
            "done": [False],
        }
        
        metrics = model.train_step(batch)
        assert "loss" in metrics
    
    def test_sample_initial_state(self):
        """Test sample_initial_state method."""
        config = WorldModelConfig(
            algorithm="test",
            state_representation="test",
        )
        model = DummyWorldModel(config)
        
        states = model.sample_initial_state(num_samples=5)
        assert len(states) == 5

