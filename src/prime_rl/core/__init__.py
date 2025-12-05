"""
Core abstractions for PRIME-RL.

This package provides the interface layer that decouples core logic from specific
backends (PyTorch/JAX) and environment implementations. All backend-specific code
should implement these interfaces.
"""

from prime_rl.core.algorithms import (
    LossDict,
    OfflineRLAlgorithm,
    RLAlgorithm,
    TrainingMetrics,
    UniversalRollout,
)
from prime_rl.core.environment import (
    AsyncEnvironmentAdapter,
    EnvironmentAdapter,
)
from prime_rl.core.interaction_trace import (
    FinalOutcome,
    InteractionTrace,
    TraceStep,
    load_trace_from_jsonl_line,
    load_traces_from_jsonl,
)
from prime_rl.core.trainer import (
    CheckpointHook,
    EvaluationHook,
    LoggingHook,
    Trainer,
)
from prime_rl.core.verifier import (
    VerifierClient,
    VerifierError,
    VerifierNetworkError,
    VerifierTimeoutError,
    VerificationResult,
)

__all__ = [
    # Algorithms and data structures
    "UniversalRollout",
    "TrainingMetrics",
    "RLAlgorithm",
    "OfflineRLAlgorithm",
    "LossDict",
    # Trainer interface
    "Trainer",
    "CheckpointHook",
    "LoggingHook",
    "EvaluationHook",
    # Environment interface
    "EnvironmentAdapter",
    "AsyncEnvironmentAdapter",
    # Verifier interface
    "VerifierClient",
    "VerificationResult",
    "VerifierError",
    "VerifierTimeoutError",
    "VerifierNetworkError",
    # Interaction Trace
    "InteractionTrace",
    "TraceStep",
    "FinalOutcome",
    "load_trace_from_jsonl_line",
    "load_traces_from_jsonl",
]

