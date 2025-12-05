"""
Surge AI marketplace client for data ingestion.

This module implements SurgeAIClient, which fetches completed labeling
tasks from Surge AI and converts them into InteractionTrace JSONL format
for offline RL training.
"""

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None  # type: ignore

from prime_rl.core.interaction_trace import InteractionTrace, TraceStep
from prime_rl.integrations.data_converters.base import DataConverter
from loguru import logger


class SurgeAIClient(DataConverter):
    """
    Client for Surge AI marketplace to fetch labeling tasks.
    
    Fetches completed labeling tasks from Surge AI and converts them
    into InteractionTrace JSONL format for offline RL training.
    
    Example:
        ```python
        client = SurgeAIClient(
            api_key="your-surge-api-key",
        )
        
        # Download tasks and convert to traces
        client.convert_to_jsonl(
            source_path=Path("surge_tasks.csv"),
            output_path=Path("traces.jsonl"),
        )
        ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.surge.ai",
        environment_id: str = "surge-ai-v1",
    ):
        """
        Initialize Surge AI client.
        
        Args:
            api_key: Surge AI API key (or set SURGE_AI_API_KEY env var)
            api_url: Surge AI API base URL
            environment_id: Environment identifier for traces
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests required for SurgeAIClient. Install with: pip install requests"
            )
        
        self.api_key = api_key or os.getenv("SURGE_AI_API_KEY")
        self.api_url = api_url.rstrip("/")
        self.environment_id = environment_id
        
        # HTTP session
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
            })
    
    def fetch_tasks(
        self,
        project_id: Optional[str] = None,
        status: str = "completed",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch labeling tasks from Surge AI API.
        
        Args:
            project_id: Optional project ID to filter tasks
            status: Task status filter (default: "completed")
            limit: Maximum number of tasks to fetch
            
        Returns:
            List of task dictionaries
            
        Note: This is a stub implementation. In production, would call
        actual Surge AI API endpoints.
        """
        # Stub: In production, would call Surge AI API
        # Example: GET /v1/tasks?project_id=...&status=completed&limit=100
        
        logger.warning("SurgeAIClient.fetch_tasks() is a stub. Implement actual API calls.")
        
        # Return mock data for testing
        return [
            {
                "task_id": f"task-{i}",
                "project_id": project_id or "default",
                "status": status,
                "input": {"prompt": f"Task {i} prompt"},
                "output": {"completion": f"Task {i} completion"},
                "label": {"score": 0.8 + (i % 3) * 0.1},
                "created_at": datetime.utcnow().isoformat(),
            }
            for i in range(min(limit, 10))
        ]
    
    def convert(self, source_data: List[Dict[str, Any]]) -> List[InteractionTrace]:
        """
        Convert Surge AI tasks to InteractionTrace objects.
        
        Args:
            source_data: List of Surge AI task dictionaries
            
        Returns:
            List of InteractionTrace objects
        """
        traces = []
        
        for task in source_data:
            try:
                trace = self._convert_task(task)
                if self.validate_trace(trace):
                    traces.append(trace)
            except Exception as e:
                logger.warning(f"Failed to convert task {task.get('task_id')}: {e}")
                continue
        
        return traces
    
    def _convert_task(self, task: Dict[str, Any]) -> InteractionTrace:
        """Convert a single Surge AI task to InteractionTrace."""
        task_id = task.get("task_id", str(uuid.uuid4()))
        
        # Extract input/output
        input_data = task.get("input", {})
        output_data = task.get("output", {})
        
        prompt = input_data.get("prompt", str(input_data))
        completion = output_data.get("completion", str(output_data))
        
        # Extract reward from label
        label = task.get("label", {})
        reward = float(label.get("score", label.get("reward", 0.0)))
        
        # Create trace step
        step = TraceStep(
            t=0,
            observation=prompt,
            action=completion,
            reward=reward,
            done=True,
            metadata={
                "task_id": task_id,
                "project_id": task.get("project_id"),
                "label": label,
            },
        )
        
        # Determine final outcome
        final_outcome = "success" if reward > 0.5 else "failure"
        
        # Create trace
        trace = InteractionTrace(
            trace_id=task_id,
            environment_id=self.environment_id,
            schema_version="v1",
            steps=[step],
            metadata={
                "source": "surge_ai",
                "project_id": task.get("project_id"),
                "created_at": task.get("created_at"),
            },
            final_outcome=final_outcome,
            labels={
                "score": reward,
                "preference_group_id": task.get("preference_group_id"),
            },
        )
        
        return trace
    
    def convert_to_jsonl(
        self,
        source_path: Path,
        output_path: Path,
        **kwargs,
    ) -> None:
        """
        Convert Surge AI CSV/JSON file to Interaction Trace JSONL format.
        
        Args:
            source_path: Path to Surge AI export file (CSV or JSON)
            output_path: Path to output JSONL file
            **kwargs: Additional options (e.g., project_id, status)
        """
        # Read source data
        if source_path.suffix == ".csv":
            tasks = self._read_csv(source_path)
        elif source_path.suffix == ".json":
            with open(source_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    tasks = data
                elif isinstance(data, dict) and "tasks" in data:
                    tasks = data["tasks"]
                else:
                    tasks = [data]
        else:
            raise ValueError(f"Unsupported file format: {source_path.suffix}")
        
        # Convert to traces
        traces = self.convert(tasks)
        
        # Write to JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict(), default=str) + "\n")
        
        logger.info(f"Converted {len(traces)} traces to {output_path}")
    
    def _read_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Read Surge AI CSV export."""
        tasks = []
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse CSV row into task format
                task = {
                    "task_id": row.get("task_id", str(uuid.uuid4())),
                    "project_id": row.get("project_id"),
                    "status": row.get("status", "completed"),
                    "input": {
                        "prompt": row.get("input", row.get("prompt", "")),
                    },
                    "output": {
                        "completion": row.get("output", row.get("completion", "")),
                    },
                    "label": {
                        "score": float(row.get("score", row.get("label", 0.0))),
                    },
                    "created_at": row.get("created_at", datetime.utcnow().isoformat()),
                }
                tasks.append(task)
        
        return tasks
    
    def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            self._session.close()

