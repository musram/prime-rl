"""
Salesforce log data converter.

This module provides utilities for converting Salesforce logs into
Interaction Trace JSONL format.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from prime_rl.integrations.data_converters.base import DataConverter
from prime_rl.core.interaction_trace import InteractionTrace, TraceStep


class SalesforceLogConverter(DataConverter):
    """
    Converter for Salesforce logs to Interaction Trace format.
    
    Converts Salesforce API logs, user interaction logs, or workflow logs
    into InteractionTrace objects for offline RL training.
    
    Example:
        ```python
        converter = SalesforceLogConverter()
        converter.convert_to_jsonl(
            source_path=Path("salesforce_logs.json"),
            output_path=Path("traces.jsonl"),
        )
        ```
    """
    
    def __init__(
        self,
        environment_id: str = "salesforce-v1",
        trace_id_field: Optional[str] = None,
        timestamp_field: Optional[str] = None,
    ):
        """
        Initialize Salesforce log converter.
        
        Args:
            environment_id: Environment identifier for traces
            trace_id_field: Optional field name for trace ID in source data
            timestamp_field: Optional field name for timestamp in source data
        """
        self.environment_id = environment_id
        self.trace_id_field = trace_id_field or "Id"
        self.timestamp_field = timestamp_field or "CreatedDate"
    
    def convert(self, source_data: List[Dict[str, Any]]) -> List[InteractionTrace]:
        """
        Convert Salesforce log data to InteractionTrace objects.
        
        Args:
            source_data: List of Salesforce log records
            
        Returns:
            List of InteractionTrace objects
        """
        traces = []
        
        for record in source_data:
            try:
                trace = self._convert_record(record)
                if self.validate_trace(trace):
                    traces.append(trace)
            except Exception as e:
                # Skip invalid records
                continue
        
        return traces
    
    def _convert_record(self, record: Dict[str, Any]) -> InteractionTrace:
        """Convert a single Salesforce record to InteractionTrace."""
        # Extract trace ID
        trace_id = record.get(self.trace_id_field, str(uuid.uuid4()))
        
        # Extract timestamp
        timestamp = record.get(self.timestamp_field)
        if timestamp:
            try:
                # Parse Salesforce timestamp format
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                timestamp = None
        
        # Extract action (e.g., API call, user action)
        action = record.get("Action", record.get("Operation", "unknown"))
        
        # Extract observation (e.g., object type, fields)
        observation = {
            "object_type": record.get("ObjectType", record.get("SObjectType", "Unknown")),
            "fields": {k: v for k, v in record.items() if k not in [self.trace_id_field, self.timestamp_field, "Action", "Operation"]},
        }
        
        # Extract reward (if available, e.g., success indicator)
        reward = 1.0 if record.get("Success", True) else 0.0
        
        # Create trace step
        step = TraceStep(
            t=0,
            observation=observation,
            action=action,
            reward=reward,
            done=True,
            metadata={
                "timestamp": timestamp.isoformat() if timestamp else None,
                "user_id": record.get("UserId"),
                "ip_address": record.get("SourceIp"),
            },
        )
        
        # Create trace
        trace = InteractionTrace(
            trace_id=trace_id,
            environment_id=self.environment_id,
            schema_version="v1",
            steps=[step],
            metadata={
                "source": "salesforce",
                "record_type": record.get("RecordType", "unknown"),
            },
            final_outcome="success" if reward > 0 else "failure",
        )
        
        return trace
    
    def convert_to_jsonl(
        self,
        source_path: Path,
        output_path: Path,
        **kwargs,
    ) -> None:
        """
        Convert Salesforce log file to Interaction Trace JSONL format.
        
        Args:
            source_path: Path to Salesforce log file (JSON or CSV)
            output_path: Path to output JSONL file
            **kwargs: Additional options (e.g., batch_size, filter_predicate)
        """
        # Read source data
        if source_path.suffix == ".json":
            with open(source_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "records" in data:
                    records = data["records"]
                elif isinstance(data, list):
                    records = data
                else:
                    records = [data]
        elif source_path.suffix == ".csv":
            import csv
            records = []
            with open(source_path, "r") as f:
                reader = csv.DictReader(f)
                records = list(reader)
        else:
            raise ValueError(f"Unsupported file format: {source_path.suffix}")
        
        # Convert to traces
        traces = self.convert(records)
        
        # Write to JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict(), default=str) + "\n")

