"""
Local metadata store for training runs.

This module provides a local filesystem-based metadata store for TrainingRun objects.
It can be upgraded to a remote database in the future for hosted RLaaS.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from prime_rl.core.training_run import TrainingRun


class MetadataStore:
    """
    Local filesystem-based metadata store for training runs.
    
    Stores TrainingRun metadata as JSON files in a directory structure:
    {base_dir}/{project_id}/{run_id}.json
    
    This can be upgraded to a remote database (PostgreSQL, MongoDB, etc.) for hosted RLaaS.
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize metadata store.
        
        Args:
            base_dir: Base directory for storing metadata
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, run: TrainingRun) -> None:
        """
        Save a TrainingRun to the store.
        
        Args:
            run: TrainingRun to save
        """
        project_dir = self.base_dir / run.project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        run_file = project_dir / f"{run.run_id}.json"
        with open(run_file, "w") as f:
            json.dump(run.to_dict(), f, indent=2)
    
    def load(self, project_id: str, run_id: str) -> Optional[TrainingRun]:
        """
        Load a TrainingRun from the store.
        
        Args:
            project_id: Project identifier
            run_id: Run identifier
            
        Returns:
            TrainingRun if found, None otherwise
        """
        run_file = self.base_dir / project_id / f"{run_id}.json"
        if not run_file.exists():
            return None
        
        with open(run_file, "r") as f:
            data = json.load(f)
        return TrainingRun.from_dict(data)
    
    def list_runs(
        self,
        project_id: str,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[TrainingRun]:
        """
        List training runs for a project.
        
        Args:
            project_id: Project identifier
            status: Optional status filter
            limit: Optional limit on number of runs
            
        Returns:
            List of TrainingRun objects
        """
        project_dir = self.base_dir / project_id
        if not project_dir.exists():
            return []
        
        runs = []
        for run_file in sorted(project_dir.glob("*.json"), reverse=True):
            try:
                with open(run_file, "r") as f:
                    data = json.load(f)
                run = TrainingRun.from_dict(data)
                
                if status is None or run.status == status:
                    runs.append(run)
                
                if limit and len(runs) >= limit:
                    break
            except Exception:
                # Skip invalid files
                continue
        
        return runs
    
    def update(self, run: TrainingRun) -> None:
        """
        Update an existing TrainingRun in the store.
        
        Args:
            run: TrainingRun to update
        """
        self.save(run)
    
    def delete(self, project_id: str, run_id: str) -> bool:
        """
        Delete a TrainingRun from the store.
        
        Args:
            project_id: Project identifier
            run_id: Run identifier
            
        Returns:
            True if deleted, False if not found
        """
        run_file = self.base_dir / project_id / f"{run_id}.json"
        if run_file.exists():
            run_file.unlink()
            return True
        return False

