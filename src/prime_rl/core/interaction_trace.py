"""
Interaction Trace data standard for offline RL.

This module implements the Interaction Trace schema defined in the PRD (§3.2).
It provides data structures and readers for converting interaction traces into
UniversalRollout instances for training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import json
import uuid


class FinalOutcome(str, Enum):
    """Final outcome of an interaction trace."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TraceStep:
    """
    A single step in an interaction trace.
    
    Attributes:
        t: Time step index
        observation: Environment observation (opaque to PRIME-RL, env-specific)
        action: Agent action (can be structured or stringified)
        reward: Reward signal at this step
        done: Whether episode is done
        logprobs: Optional policy logprobs if available
        metadata: Optional step-specific metadata
    """
    t: int
    observation: Any
    action: Any
    reward: float
    done: bool
    logprobs: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceStep":
        """Create TraceStep from dictionary."""
        return cls(
            t=data["t"],
            observation=data["observation"],
            action=data["action"],
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            logprobs=data.get("logprobs"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert TraceStep to dictionary."""
        result = {
            "t": self.t,
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "metadata": self.metadata,
        }
        if self.logprobs is not None:
            result["logprobs"] = self.logprobs
        return result


@dataclass
class InteractionTrace:
    """
    A complete interaction trace (trajectory).
    
    This represents a single episode of agent-environment interaction, following
    the schema defined in PRD §3.2.
    
    Attributes:
        trace_id: Unique identifier for this trace (UUID)
        environment_id: Identifier for the environment (e.g., "browser-gym-v1")
        schema_version: Schema version string (e.g., "v1")
        metadata: Trace-level metadata (source, timestamps, etc.)
        steps: List of interaction steps
        final_outcome: Final outcome of the trace
        labels: Optional supervision labels (human_score, preference_group_id, etc.)
    """
    trace_id: str
    environment_id: str
    schema_version: str
    steps: List[TraceStep]
    metadata: Dict[str, Any] = field(default_factory=dict)
    final_outcome: Optional[str] = None
    labels: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionTrace":
        """
        Create InteractionTrace from dictionary.
        
        Validates required fields and tolerates partial traces (missing optional fields).
        """
        # Validate required fields
        required_fields = ["trace_id", "environment_id", "schema_version", "steps"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate steps
        if not isinstance(data["steps"], list) or len(data["steps"]) == 0:
            raise ValueError("steps must be a non-empty list")

        # Parse steps
        steps = [TraceStep.from_dict(step_data) for step_data in data["steps"]]

        # Parse final_outcome (validate if present)
        final_outcome = data.get("final_outcome")
        if final_outcome is not None:
            try:
                FinalOutcome(final_outcome)
            except ValueError:
                # Allow non-enum values but log warning
                pass

        return cls(
            trace_id=data["trace_id"],
            environment_id=data["environment_id"],
            schema_version=data["schema_version"],
            steps=steps,
            metadata=data.get("metadata", {}),
            final_outcome=final_outcome,
            labels=data.get("labels", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert InteractionTrace to dictionary."""
        result = {
            "trace_id": self.trace_id,
            "environment_id": self.environment_id,
            "schema_version": self.schema_version,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.final_outcome is not None:
            result["final_outcome"] = self.final_outcome
        if self.labels:
            result["labels"] = self.labels
        return result

    def to_universal_rollout(self) -> "UniversalRollout":
        """
        Convert InteractionTrace to UniversalRollout.
        
        For single-step traces, extracts prompt/completion from first/last step.
        For multi-step traces, includes full trajectory data.
        """
        from prime_rl.core.algorithms import UniversalRollout

        # Extract prompt and completion
        # For now, we'll use the first observation as prompt and last action as completion
        # This is a simple heuristic; actual conversion may depend on environment type
        if len(self.steps) == 0:
            raise ValueError("Cannot convert empty trace to rollout")

        first_step = self.steps[0]
        last_step = self.steps[-1]

        # Extract prompt (first observation or metadata)
        prompt = str(first_step.observation) if first_step.observation else ""
        if not prompt and "prompt" in self.metadata:
            prompt = str(self.metadata["prompt"])

        # Extract completion (last action or accumulated actions)
        completion = str(last_step.action) if last_step.action else ""
        if not completion and len(self.steps) > 1:
            # Concatenate all actions as completion
            completion = " ".join(str(step.action) for step in self.steps if step.action)

        # Extract rewards (use final reward or sum)
        reward = last_step.reward
        if len(self.steps) > 1:
            # Optionally sum rewards across steps
            reward = sum(step.reward for step in self.steps)

        # Build metadata
        metadata = {
            "trace_id": self.trace_id,
            "environment_id": self.environment_id,
            "final_outcome": self.final_outcome,
            "num_steps": len(self.steps),
            **self.labels,
        }

        # Include full trajectory if multi-step
        observations = [step.observation for step in self.steps] if len(self.steps) > 1 else None
        actions = [step.action for step in self.steps] if len(self.steps) > 1 else None
        dones = [step.done for step in self.steps] if len(self.steps) > 1 else None

        return UniversalRollout(
            prompts=[prompt],
            completions=[completion],
            rewards=[reward],
            observations=observations,
            actions=actions,
            dones=dones,
            metadata=metadata,
        )


def load_trace_from_jsonl_line(line: str) -> InteractionTrace:
    """
    Load a single InteractionTrace from a JSONL line.
    
    Args:
        line: JSONL line (JSON string)
        
    Returns:
        InteractionTrace instance
        
    Raises:
        ValueError: If line is invalid or missing required fields
        json.JSONDecodeError: If line is not valid JSON
    """
    data = json.loads(line.strip())
    return InteractionTrace.from_dict(data)


def load_traces_from_jsonl(file_path: str) -> List[InteractionTrace]:
    """
    Load multiple InteractionTrace objects from a JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of InteractionTrace objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If any line is invalid
    """
    traces = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                trace = load_trace_from_jsonl_line(line)
                traces.append(trace)
            except Exception as e:
                raise ValueError(f"Error parsing line {line_num} in {file_path}: {e}") from e
    return traces

