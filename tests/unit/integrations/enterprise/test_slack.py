"""Unit tests for SlackAdapter."""

import pytest
from unittest.mock import Mock, patch
from prime_rl.integrations.enterprise.slack import SlackAdapter
from prime_rl.core.algorithms import UniversalRollout


class TestSlackAdapter:
    """Tests for SlackAdapter."""
    
    def test_initialization_mock_mode(self):
        """Test adapter initialization in mock mode."""
        adapter = SlackAdapter(
            token="xoxb-test",
            channel_id="C1234567890",
            mock_mode=True,
        )
        
        assert adapter.token == "xoxb-test"
        assert adapter.channel_id == "C1234567890"
        assert adapter.mock_mode is True
    
    def test_initialization_without_slack_sdk(self):
        """Test initialization fails without slack_sdk in non-mock mode."""
        with patch("prime_rl.integrations.enterprise.slack.SLACK_AVAILABLE", False):
            with pytest.raises(ImportError):
                SlackAdapter(
                    token="xoxb-test",
                    channel_id="C1234567890",
                    mock_mode=False,
                )
    
    def test_reset(self):
        """Test environment reset."""
        adapter = SlackAdapter(
            token="xoxb-test",
            channel_id="C1234567890",
            mock_mode=True,
        )
        
        obs, info = adapter.reset()
        
        assert obs is not None
        assert "channel_id" in obs
        assert info["channel_id"] == "C1234567890"
        assert adapter._episode_started is True
    
    def test_reset_with_channel_id(self):
        """Test reset with specific channel ID."""
        adapter = SlackAdapter(
            token="xoxb-test",
            mock_mode=True,
        )
        
        obs, info = adapter.reset(options={"channel_id": "C9999999999"})
        
        assert adapter._current_channel == "C9999999999"
    
    def test_step_read_messages(self):
        """Test read_messages action."""
        adapter = SlackAdapter(
            token="xoxb-test",
            channel_id="C1234567890",
            mock_mode=True,
        )
        adapter.reset()
        
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "read_messages",
        })
        
        assert reward > 0
        assert "messages" in obs or "status" in obs
    
    def test_step_post_message(self):
        """Test post_message action."""
        adapter = SlackAdapter(
            token="xoxb-test",
            channel_id="C1234567890",
            mock_mode=True,
        )
        adapter.reset()
        
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "post_message",
            "text": "Hello, team!",
        })
        
        assert reward > 0
        assert "message_posted" in obs or "status" in obs
    
    def test_step_react(self):
        """Test react action."""
        adapter = SlackAdapter(
            token="xoxb-test",
            channel_id="C1234567890",
            mock_mode=True,
        )
        adapter.reset()
        
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "react",
            "message_ts": "1234567890.123456",
            "emoji": "thumbsup",
        })
        
        assert reward > 0
    
    def test_step_complete(self):
        """Test complete action terminates episode."""
        adapter = SlackAdapter(
            token="xoxb-test",
            channel_id="C1234567890",
            mock_mode=True,
        )
        adapter.reset()
        
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "complete",
        })
        
        assert terminated is True
        assert reward > 0
    
    def test_get_rollout(self):
        """Test converting trajectory to UniversalRollout."""
        adapter = SlackAdapter(
            token="xoxb-test",
            channel_id="C1234567890",
            mock_mode=True,
        )
        adapter.reset()
        
        # Execute some steps
        adapter.step({"action_type": "read_messages"})
        adapter.step({"action_type": "post_message", "text": "Hello!"})
        
        rollout = adapter.get_rollout()
        
        assert isinstance(rollout, UniversalRollout)
        assert len(rollout.prompts) > 0
        assert len(rollout.completions) > 0
    
    def test_render(self):
        """Test rendering environment state."""
        adapter = SlackAdapter(
            token="xoxb-test",
            channel_id="C1234567890",
            mock_mode=True,
        )
        adapter.reset()
        
        state = adapter.render()
        
        assert state is not None
        assert "channel_id" in state
    
    def test_close(self):
        """Test closing environment."""
        adapter = SlackAdapter(
            token="xoxb-test",
            channel_id="C1234567890",
            mock_mode=True,
        )
        adapter.reset()
        
        adapter.close()
        
        assert adapter._current_channel is None
        assert adapter._episode_started is False

