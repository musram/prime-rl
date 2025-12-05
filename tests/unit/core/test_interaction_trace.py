"""Tests for Interaction Trace data structures and readers."""

import json
import tempfile
from pathlib import Path

import pytest

from prime_rl.core.interaction_trace import (
    FinalOutcome,
    InteractionTrace,
    TraceStep,
    load_trace_from_jsonl_line,
    load_traces_from_jsonl,
)


def test_trace_step_from_dict():
    """Test TraceStep creation from dictionary."""
    data = {
        "t": 0,
        "observation": "test_obs",
        "action": "test_action",
        "reward": 1.0,
        "done": False,
        "logprobs": [0.1, 0.2],
        "metadata": {"key": "value"},
    }
    step = TraceStep.from_dict(data)
    assert step.t == 0
    assert step.observation == "test_obs"
    assert step.action == "test_action"
    assert step.reward == 1.0
    assert step.done is False
    assert step.logprobs == [0.1, 0.2]
    assert step.metadata == {"key": "value"}


def test_interaction_trace_from_dict():
    """Test InteractionTrace creation from dictionary."""
    data = {
        "trace_id": "test-uuid",
        "environment_id": "test-env",
        "schema_version": "v1",
        "metadata": {"source": "test"},
        "steps": [
            {
                "t": 0,
                "observation": "obs1",
                "action": "action1",
                "reward": 0.0,
                "done": False,
            },
            {
                "t": 1,
                "observation": "obs2",
                "action": "action2",
                "reward": 1.0,
                "done": True,
            },
        ],
        "final_outcome": "success",
        "labels": {"human_score": 0.9},
    }
    trace = InteractionTrace.from_dict(data)
    assert trace.trace_id == "test-uuid"
    assert trace.environment_id == "test-env"
    assert trace.schema_version == "v1"
    assert len(trace.steps) == 2
    assert trace.final_outcome == "success"
    assert trace.labels == {"human_score": 0.9}


def test_interaction_trace_missing_required_field():
    """Test that missing required fields raise ValueError."""
    data = {
        "trace_id": "test-uuid",
        # Missing environment_id
        "schema_version": "v1",
        "steps": [],
    }
    with pytest.raises(ValueError, match="Missing required field"):
        InteractionTrace.from_dict(data)


def test_interaction_trace_to_universal_rollout():
    """Test conversion to UniversalRollout."""
    trace = InteractionTrace(
        trace_id="test-uuid",
        environment_id="test-env",
        schema_version="v1",
        steps=[
            TraceStep(
                t=0,
                observation="prompt_text",
                action="completion_text",
                reward=1.0,
                done=True,
            )
        ],
        final_outcome="success",
    )
    rollout = trace.to_universal_rollout()
    assert len(rollout.prompts) == 1
    assert len(rollout.completions) == 1
    assert len(rollout.rewards) == 1
    assert rollout.rewards[0] == 1.0
    assert rollout.metadata["trace_id"] == "test-uuid"


def test_load_trace_from_jsonl_line():
    """Test loading trace from JSONL line."""
    data = {
        "trace_id": "test-uuid",
        "environment_id": "test-env",
        "schema_version": "v1",
        "steps": [
            {
                "t": 0,
                "observation": "obs",
                "action": "act",
                "reward": 1.0,
                "done": True,
            }
        ],
    }
    line = json.dumps(data)
    trace = load_trace_from_jsonl_line(line)
    assert trace.trace_id == "test-uuid"


def test_load_traces_from_jsonl():
    """Test loading multiple traces from JSONL file."""
    traces_data = [
        {
            "trace_id": f"uuid-{i}",
            "environment_id": "test-env",
            "schema_version": "v1",
            "steps": [
                {
                    "t": 0,
                    "observation": f"obs-{i}",
                    "action": f"act-{i}",
                    "reward": float(i),
                    "done": True,
                }
            ],
        }
        for i in range(3)
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for data in traces_data:
            f.write(json.dumps(data) + "\n")
        temp_path = f.name
    
    try:
        traces = load_traces_from_jsonl(temp_path)
        assert len(traces) == 3
        assert traces[0].trace_id == "uuid-0"
        assert traces[2].trace_id == "uuid-2"
    finally:
        Path(temp_path).unlink()

