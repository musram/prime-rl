"""
Integrations layer for PRIME-RL.

This module provides "batteries-included" adapters for common environment standards
and remote verification services, making prime-rl ready to plug into the browser use
and remote grading ecosystem immediately.
"""

from prime_rl.integrations.remote.http_verifier import HttpVerifierClient
from prime_rl.integrations.remote.grpc_verifier import GrpcVerifierClient
from prime_rl.integrations.browser_gym.adapter import BrowserGymAdapter
from prime_rl.integrations.langgraph.adapter import LangGraphAdapter
from prime_rl.integrations.mcp.adapter import MCPAdapter
from prime_rl.integrations.prime_intellect import (
    PrimeIntellectEnvAdapter,
    PrimeIntellectVerifierClient,
)
from prime_rl.integrations.data_converters import (
    DataConverter,
    SalesforceLogConverter,
    MCPLogConverter,
)

# Enterprise integrations
from prime_rl.integrations.enterprise import EpicAdapter, SlackAdapter

# Marketplace integrations
from prime_rl.integrations.marketplace import SurgeAIClient, MercorClient

__all__ = [
    "HttpVerifierClient",
    "GrpcVerifierClient",
    "BrowserGymAdapter",
    "LangGraphAdapter",
    "MCPAdapter",
    "PrimeIntellectEnvAdapter",
    "PrimeIntellectVerifierClient",
    "DataConverter",
    "SalesforceLogConverter",
    "MCPLogConverter",
    # Enterprise integrations
    "EpicAdapter",
    "SlackAdapter",
    # Marketplace integrations
    "SurgeAIClient",
    "MercorClient",
]

