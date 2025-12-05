"""
Tests for BrowserGymAdapter.

Tests the adapter with dummy Gym environments to ensure proper wrapping
and UniversalRollout conversion.
"""

import pytest
from typing import Any, Dict, Tuple
from unittest.mock import Mock

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYMNASIUM_AVAILABLE = False
    except ImportError:
        GYMNASIUM_AVAILABLE = False
        gym = None

if gym is not None:
    from prime_rl.integrations.browser_gym.adapter import BrowserGymAdapter
    from prime_rl.core.algorithms import UniversalRollout


@pytest.mark.skipif(gym is None, reason="Gymnasium or Gym not available")
class TestBrowserGymAdapter:
    """Tests for BrowserGymAdapter."""
    
    def test_initialization(self):
        """Test BrowserGymAdapter initialization."""
        # Create a mock environment
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        adapter = BrowserGymAdapter(mock_env)
        assert adapter.env == mock_env
        assert not adapter._episode_started
    
    def test_reset(self):
        """Test environment reset."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        if GYMNASIUM_AVAILABLE:
            mock_env.reset.return_value = ("obs", {"info": "test"})
        else:
            mock_env.reset.return_value = "obs"
        
        adapter = BrowserGymAdapter(mock_env)
        
        if GYMNASIUM_AVAILABLE:
            obs, info = adapter.reset(seed=42)
            assert obs == "obs"
            assert info == {"info": "test"}
            mock_env.reset.assert_called_once_with(seed=42, options=None)
        else:
            obs, info = adapter.reset()
            assert obs == "obs"
            mock_env.reset.assert_called_once()
        
        assert adapter._episode_started is True
        assert len(adapter._current_trajectory["observations"]) == 1
    
    def test_step(self):
        """Test environment step."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        if GYMNASIUM_AVAILABLE:
            mock_env.reset.return_value = ("obs0", {})
            mock_env.step.return_value = ("obs1", 1.0, False, False, {})
        else:
            mock_env.reset.return_value = "obs0"
            mock_env.step.return_value = ("obs1", 1.0, False, {})
        
        adapter = BrowserGymAdapter(mock_env)
        adapter.reset()
        
        obs, reward, terminated, truncated, info = adapter.step("action1")
        
        assert obs == "obs1"
        assert reward == 1.0
        assert terminated is False
        assert len(adapter._current_trajectory["actions"]) == 1
        assert len(adapter._current_trajectory["rewards"]) == 1
    
    def test_step_before_reset(self):
        """Test that stepping before reset raises error."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        adapter = BrowserGymAdapter(mock_env)
        
        with pytest.raises(RuntimeError, match="must be reset"):
            adapter.step("action")
    
    def test_get_rollout_single_step(self):
        """Test getting rollout for single-step trajectory."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        if GYMNASIUM_AVAILABLE:
            mock_env.reset.return_value = ("obs0", {})
            mock_env.step.return_value = ("obs1", 1.0, True, False, {})
        else:
            mock_env.reset.return_value = "obs0"
            mock_env.step.return_value = ("obs1", 1.0, True, {})
        
        adapter = BrowserGymAdapter(mock_env)
        adapter.reset()
        adapter.step("action1")
        
        rollout = adapter.get_rollout()
        
        assert isinstance(rollout, UniversalRollout)
        assert len(rollout.prompts) == 1
        assert len(rollout.completions) == 1
        assert len(rollout.rewards) == 1
        assert rollout.rewards[0] == 1.0
        assert "num_steps" in rollout.metadata
    
    def test_get_rollout_multi_step(self):
        """Test getting rollout for multi-step trajectory."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        if GYMNASIUM_AVAILABLE:
            mock_env.reset.return_value = ("obs0", {})
            mock_env.step.side_effect = [
                ("obs1", 0.5, False, False, {}),
                ("obs2", 0.5, True, False, {}),
            ]
        else:
            mock_env.reset.return_value = "obs0"
            mock_env.step.side_effect = [
                ("obs1", 0.5, False, {}),
                ("obs2", 0.5, True, {}),
            ]
        
        adapter = BrowserGymAdapter(mock_env)
        adapter.reset()
        adapter.step("action1")
        adapter.step("action2")
        
        rollout = adapter.get_rollout()
        
        assert isinstance(rollout, UniversalRollout)
        assert rollout.rewards[0] == 1.0  # Sum of rewards
        assert rollout.observations is not None
        assert rollout.actions is not None
        assert len(rollout.actions) == 2
    
    def test_get_rollout_before_steps(self):
        """Test getting rollout before any steps raises error."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        adapter = BrowserGymAdapter(mock_env)
        
        with pytest.raises(RuntimeError, match="No trajectory data"):
            adapter.get_rollout()
    
    def test_observation_space(self):
        """Test observation space property."""
        mock_env = Mock()
        mock_obs_space = Mock()
        mock_env.observation_space = mock_obs_space
        mock_env.action_space = Mock()
        
        adapter = BrowserGymAdapter(mock_env)
        assert adapter.observation_space == mock_obs_space
    
    def test_action_space(self):
        """Test action space property."""
        mock_env = Mock()
        mock_action_space = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = mock_action_space
        
        adapter = BrowserGymAdapter(mock_env)
        assert adapter.action_space == mock_action_space
    
    def test_render(self):
        """Test rendering."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        mock_env.render.return_value = "rendered_output"
        
        adapter = BrowserGymAdapter(mock_env)
        result = adapter.render()
        
        assert result == "rendered_output"
        mock_env.render.assert_called_once()
    
    def test_render_error(self):
        """Test rendering error handling."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        mock_env.render.side_effect = Exception("Render failed")
        
        adapter = BrowserGymAdapter(mock_env)
        result = adapter.render()
        
        assert result is None
    
    def test_close(self):
        """Test closing adapter."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        adapter = BrowserGymAdapter(mock_env)
        adapter.reset()
        
        adapter.close()
        
        mock_env.close.assert_called_once()
        assert not adapter._episode_started
    
    def test_custom_observation_to_string(self):
        """Test custom observation to string conversion."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        def custom_obs_to_str(obs):
            return f"CUSTOM_{obs}"
        
        adapter = BrowserGymAdapter(
            mock_env,
            observation_to_string=custom_obs_to_str,
        )
        
        if GYMNASIUM_AVAILABLE:
            mock_env.reset.return_value = ("test_obs", {})
        else:
            mock_env.reset.return_value = "test_obs"
        
        adapter.reset()
        rollout = adapter.get_rollout()
        
        assert "CUSTOM_" in rollout.prompts[0]
    
    def test_custom_action_to_string(self):
        """Test custom action to string conversion."""
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        
        def custom_act_to_str(act):
            return f"ACTION_{act}"
        
        adapter = BrowserGymAdapter(
            mock_env,
            action_to_string=custom_act_to_str,
        )
        
        if GYMNASIUM_AVAILABLE:
            mock_env.reset.return_value = ("obs0", {})
            mock_env.step.return_value = ("obs1", 1.0, True, False, {})
        else:
            mock_env.reset.return_value = "obs0"
            mock_env.step.return_value = ("obs1", 1.0, True, {})
        
        adapter.reset()
        adapter.step("test_action")
        rollout = adapter.get_rollout()
        
        assert "ACTION_" in rollout.completions[0]

