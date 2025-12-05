"""
PRIME-RL Client SDK for Forward Deployment.

This module provides the client-side SDK for remote workers to communicate
with the PRIME-RL Orchestrator cluster.
"""

from prime_rl.client.worker import RemoteEnvironmentWorker

__all__ = [
    "RemoteEnvironmentWorker",
]

