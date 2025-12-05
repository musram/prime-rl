"""
Remote verification integrations.

This module provides implementations of VerifierClient for remote HTTP/gRPC verification services.
"""

from prime_rl.integrations.remote.http_verifier import HttpVerifierClient

__all__ = [
    "HttpVerifierClient",
]

