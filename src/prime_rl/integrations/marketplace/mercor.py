"""
Mercor marketplace client for expert demonstration ingestion.

This module implements MercorClient, which ingests expert demonstration
logs from Mercor and converts them into InteractionTrace JSONL format
for offline RL training.
"""

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


class MercorClient(DataConverter):
    """
    Client for Mercor marketplace to fetch expert demonstrations.
    
    Ingests expert demonstration logs from Mercor and converts them
    into InteractionTrace JSONL format for offline RL training.
    
    Example:
        ```python
        client = MercorClient(
            api_key="your-mercor-api-key",
        )
        
        # Download demonstrations and convert to traces
        client.convert_to_jsonl(
            source_path=Path("mercor_demos.json"),
            output_path=Path("traces.jsonl"),
        )
        ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.mercor.ai",
        environment_id: str = "mercor-v1",
    ):
        """
        Initialize Mercor client.
        
        Args:
            api_key: Mercor API key (or set MERCOR_API_KEY env var)
            api_url: Mercor API base URL
            environment_id: Environment identifier for traces
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests required for MercorClient. Install with: pip install requests"
            )
        
        self.api_key = api_key or os.getenv("MERCOR_API_KEY")
        self.api_url = api_url.rstrip("/")
        self.environment_id = environment_id
        
        # HTTP session
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
            })
    
    def fetch_demonstrations(
        self,
        expert_id: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch expert demonstrations from Mercor API.
        
        Args:
            expert_id: Optional expert ID to filter demonstrations
            task_type: Optional task type filter
            limit: Maximum number of demonstrations to fetch
            
        Returns:
            List of demonstration dictionaries
            
        Note: This is a stub implementation. In production, would call
        actual Mercor API endpoints.
        """
        # Stub: In production, would call Mercor API
        # Example: GET /v1/demonstrations?expert_id=...&task_type=...&limit=100
        
        logger.warning("MercorClient.fetch_demonstrations() is a stub. Implement actual API calls.")
        
        # Return mock data for testing
        return [
            {
                "demo_id": f"demo-{i}",
                "expert_id": expert_id or f"expert-{i % 3}",
                "task_type": task_type or "coding",
                "prompt": f"Task {i}: Write a function to...",
                "solution": f"def solution_{i}():\n    return True",
                "steps": [
                    {"action": "think", "content": "I need to..."},
                    {"action": "code", "content": "def solution..."},
                ],
                "quality_score": 0.9 - (i % 3) * 0.1,
                "created_at": datetime.utcnow().isoformat(),
            }
            for i in range(min(limit, 10))
        ]
    
    def convert(self, source_data: List[Dict[str, Any]]) -> List[InteractionTrace]:
        """
        Convert Mercor demonstrations to InteractionTrace objects.
        
        Args:
            source_data: List of Mercor demonstration dictionaries
            
        Returns:
            List of InteractionTrace objects
        """
        traces = []
        
        for demo in source_data:
            try:
                trace = self._convert_demonstration(demo)
                if self.validate_trace(trace):
                    traces.append(trace)
            except Exception as e:
                logger.warning(f"Failed to convert demo {demo.get('demo_id')}: {e}")
                continue
        
        return traces
    
    def _convert_demonstration(self, demo: Dict[str, Any]) -> InteractionTrace:
        """Convert a single Mercor demonstration to InteractionTrace."""
        demo_id = demo.get("demo_id", str(uuid.uuid4()))
        
        # Extract prompt and solution
        prompt = demo.get("prompt", "")
        solution = demo.get("solution", "")
        
        # Extract steps if available (multi-step demonstration)
        steps_data = demo.get("steps", [])
        
        if steps_data and len(steps_data) > 1:
            # Multi-step demonstration
            steps = []
            for i, step in enumerate(steps_data):
                action = step.get("action", step.get("content", ""))
                content = step.get("content", step.get("result", ""))
                
                trace_step = TraceStep(
                    t=i,
                    observation=prompt if i == 0 else steps[i-1].observation,
                    action=action,
                    reward=0.0,  # Intermediate steps have no reward
                    done=False,
                    metadata={
                        "step_type": step.get("action"),
                        "content": content,
                    },
                )
                steps.append(trace_step)
            
            # Final step with reward
            quality_score = float(demo.get("quality_score", 0.0))
            final_step = TraceStep(
                t=len(steps),
                observation=steps[-1].observation if steps else prompt,
                action=solution,
                reward=quality_score,
                done=True,
                metadata={
                    "solution": solution,
                    "quality_score": quality_score,
                },
            )
            steps.append(final_step)
        else:
            # Single-step demonstration
            quality_score = float(demo.get("quality_score", 0.0))
            steps = [
                TraceStep(
                    t=0,
                    observation=prompt,
                    action=solution,
                    reward=quality_score,
                    done=True,
                    metadata={
                        "solution": solution,
                        "quality_score": quality_score,
                    },
                )
            ]
        
        # Determine final outcome
        final_outcome = "success" if quality_score > 0.5 else "failure"
        
        # Create trace
        trace = InteractionTrace(
            trace_id=demo_id,
            environment_id=self.environment_id,
            schema_version="v1",
            steps=steps,
            metadata={
                "source": "mercor",
                "expert_id": demo.get("expert_id"),
                "task_type": demo.get("task_type"),
                "created_at": demo.get("created_at"),
            },
            final_outcome=final_outcome,
            labels={
                "quality_score": quality_score,
                "expert_id": demo.get("expert_id"),
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
        Convert Mercor demonstration file to Interaction Trace JSONL format.
        
        Args:
            source_path: Path to Mercor export file (JSON)
            output_path: Path to output JSONL file
            **kwargs: Additional options
        """
        # Read source data
        with open(source_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                demonstrations = data
            elif isinstance(data, dict) and "demonstrations" in data:
                demonstrations = data["demonstrations"]
            else:
                demonstrations = [data]
        
        # Convert to traces
        traces = self.convert(demonstrations)
        
        # Write to JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict(), default=str) + "\n")
        
        logger.info(f"Converted {len(traces)} traces to {output_path}")
    
    def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            self._session.close()

