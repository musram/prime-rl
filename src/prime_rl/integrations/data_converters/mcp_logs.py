"""
MCP server log data converter.

This module provides utilities for converting MCP server logs into
Interaction Trace JSONL format.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from prime_rl.integrations.data_converters.base import DataConverter
from prime_rl.core.interaction_trace import InteractionTrace, TraceStep


class MCPLogConverter(DataConverter):
    """
    Converter for MCP server logs to Interaction Trace format.
    
    Converts MCP tool call logs, resource access logs, or protocol logs
    into InteractionTrace objects for offline RL training.
    
    Example:
        ```python
        converter = MCPLogConverter()
        converter.convert_to_jsonl(
            source_path=Path("mcp_logs.json"),
            output_path=Path("traces.jsonl"),
        )
        ```
    """
    
    def __init__(
        self,
        environment_id: str = "mcp-server-v1",
        trace_id_field: Optional[str] = None,
    ):
        """
        Initialize MCP log converter.
        
        Args:
            environment_id: Environment identifier for traces
            trace_id_field: Optional field name for trace ID in source data
        """
        self.environment_id = environment_id
        self.trace_id_field = trace_id_field or "request_id"
    
    def convert(self, source_data: List[Dict[str, Any]]) -> List[InteractionTrace]:
        """
        Convert MCP log data to InteractionTrace objects.
        
        Args:
            source_data: List of MCP log records
            
        Returns:
            List of InteractionTrace objects
        """
        traces = []
        
        # Group logs by trace_id (request_id or session_id)
        grouped_logs: Dict[str, List[Dict[str, Any]]] = {}
        
        for record in source_data:
            trace_id = record.get(self.trace_id_field, str(uuid.uuid4()))
            if trace_id not in grouped_logs:
                grouped_logs[trace_id] = []
            grouped_logs[trace_id].append(record)
        
        # Convert each group to a trace
        for trace_id, logs in grouped_logs.items():
            try:
                trace = self._convert_log_group(trace_id, logs)
                if self.validate_trace(trace):
                    traces.append(trace)
            except Exception as e:
                # Skip invalid traces
                continue
        
        return traces
    
    def _convert_log_group(
        self,
        trace_id: str,
        logs: List[Dict[str, Any]],
    ) -> InteractionTrace:
        """Convert a group of MCP logs to InteractionTrace."""
        steps = []
        
        for i, log in enumerate(logs):
            # Extract tool call or resource access
            tool_name = log.get("tool", log.get("method", "unknown"))
            arguments = log.get("arguments", log.get("params", {}))
            
            # Extract result
            result = log.get("result", log.get("response", {}))
            error = log.get("error")
            
            # Create observation (tool result or error)
            observation = {
                "tool": tool_name,
                "result": result,
                "error": error,
            }
            
            # Create action (tool call with arguments)
            action = {
                "tool": tool_name,
                "arguments": arguments,
            }
            
            # Compute reward (success = 1.0, error = -1.0)
            reward = 1.0 if not error else -1.0
            
            # Check if done (last log or error)
            done = (i == len(logs) - 1) or (error is not None)
            
            step = TraceStep(
                t=i,
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                metadata={
                    "timestamp": log.get("timestamp"),
                    "duration_ms": log.get("duration_ms"),
                },
            )
            steps.append(step)
        
        # Determine final outcome
        final_outcome = "success"
        if steps and steps[-1].reward < 0:
            final_outcome = "failure"
        
        # Create trace
        trace = InteractionTrace(
            trace_id=trace_id,
            environment_id=self.environment_id,
            schema_version="v1",
            steps=steps,
            metadata={
                "source": "mcp_server",
                "num_tool_calls": len(steps),
            },
            final_outcome=final_outcome,
        )
        
        return trace
    
    def convert_to_jsonl(
        self,
        source_path: Path,
        output_path: Path,
        **kwargs,
    ) -> None:
        """
        Convert MCP log file to Interaction Trace JSONL format.
        
        Args:
            source_path: Path to MCP log file (JSON)
            output_path: Path to output JSONL file
            **kwargs: Additional options
        """
        # Read source data
        with open(source_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "logs" in data:
                records = data["logs"]
            else:
                records = [data]
        
        # Convert to traces
        traces = self.convert(records)
        
        # Write to JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict(), default=str) + "\n")

