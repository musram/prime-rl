"""
Data converters for external data sources.

This module provides utilities for converting data from external sources
(e.g., Salesforce logs, MCP server logs) into the Interaction Trace JSONL format.
"""

from prime_rl.integrations.data_converters.base import DataConverter
from prime_rl.integrations.data_converters.salesforce import SalesforceLogConverter
from prime_rl.integrations.data_converters.mcp_logs import MCPLogConverter

__all__ = [
    "DataConverter",
    "SalesforceLogConverter",
    "MCPLogConverter",
]

