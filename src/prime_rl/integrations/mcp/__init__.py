"""
MCP (Model Context Protocol) environment integrations.

This module provides adapters for MCP servers, allowing MCP-based tools
and resources to be used as RL environments.
"""

from prime_rl.integrations.mcp.adapter import MCPAdapter

__all__ = [
    "MCPAdapter",
]

