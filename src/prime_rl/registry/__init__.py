"""
Scenario registry for PRIME-RL.

Provides a central catalog of available scenarios (environments, tasks, benchmarks)
with metadata for discovery and configuration.
"""

from prime_rl.registry.registry import ScenarioRegistry, ScenarioMetadata

__all__ = [
    "ScenarioRegistry",
    "ScenarioMetadata",
]

