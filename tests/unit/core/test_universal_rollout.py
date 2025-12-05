"""Tests for UniversalRollout data structure."""

from prime_rl.core.algorithms import UniversalRollout


def test_universal_rollout_creation():
    """Test UniversalRollout creation."""
    rollout = UniversalRollout(
        prompts=["prompt1", "prompt2"],
        completions=["completion1", "completion2"],
        rewards=[1.0, 2.0],
        metadata={"key": "value"},
    )
    assert len(rollout.prompts) == 2
    assert len(rollout.completions) == 2
    assert len(rollout.rewards) == 2
    assert rollout.metadata == {"key": "value"}


def test_universal_rollout_to_dict():
    """Test UniversalRollout conversion to dictionary."""
    rollout = UniversalRollout(
        prompts=["prompt1"],
        completions=["completion1"],
        rewards=[1.0],
        observations=["obs1"],
        actions=["act1"],
        dones=[True],
    )
    data = rollout.to_dict()
    assert "prompts" in data
    assert "completions" in data
    assert "rewards" in data
    assert "observations" in data
    assert "actions" in data
    assert "dones" in data

