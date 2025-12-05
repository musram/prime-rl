"""
Python SDK for embedding PRIME-RL in other systems.

This module provides a high-level Python API for programmatically using PRIME-RL,
enabling integration with other systems and workflows as specified in the PRD (§4.3).
"""

from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import sys

from prime_rl.core.config import UnifiedConfig
from prime_rl.core.training_run import TrainingRun
from prime_rl.core.metadata_store import MetadataStore
from prime_rl.train import train as train_func


class PRIMERLClient:
    """
    Python SDK client for PRIME-RL.
    
    Provides a programmatic interface for creating and managing training runs,
    enabling PRIME-RL to be embedded in other systems.
    """
    
    def __init__(
        self,
        project_id: str = "default",
        metadata_store: Optional[MetadataStore] = None,
        metadata_dir: Optional[Path] = None,
    ):
        """
        Initialize PRIME-RL client.
        
        Args:
            project_id: Default project identifier
            metadata_store: Optional metadata store instance
            metadata_dir: Optional directory for local metadata store
        """
        self.project_id = project_id
        if metadata_store is None:
            metadata_dir = metadata_dir or Path.home() / ".prime_rl" / "metadata"
            self.metadata_store = MetadataStore(metadata_dir)
        else:
            self.metadata_store = metadata_store
    
    def create_run(
        self,
        config_path: Path,
        project_id: Optional[str] = None,
        code_revision: Optional[str] = None,
    ) -> TrainingRun:
        """
        Create a new training run.
        
        Args:
            config_path: Path to configuration file
            project_id: Optional project identifier (uses default if not provided)
            code_revision: Optional code revision/commit hash
            
        Returns:
            TrainingRun instance
        """
        # Load config
        import tomli
        with open(config_path, "rb") as f:
            config_dict = tomli.load(f)
        
        project_id = project_id or self.project_id
        
        # Extract references from config
        dataset_reference = None
        if "dataset" in config_dict and "path" in config_dict["dataset"]:
            dataset_reference = str(config_dict["dataset"]["path"])
        
        environment_reference = None
        if "environment" in config_dict:
            environment_reference = config_dict["environment"].get("id")
        
        # Create run
        run = TrainingRun.create(
            project_id=project_id,
            config=config_dict,
            code_revision=code_revision,
            dataset_reference=dataset_reference,
            environment_reference=environment_reference,
        )
        
        # Save to metadata store
        self.metadata_store.save(run)
        
        return run
    
    def start_run(
        self,
        run: TrainingRun,
        mode: Optional[str] = None,
        blocking: bool = True,
    ) -> TrainingRun:
        """
        Start a training run.
        
        Args:
            run: TrainingRun to start
            mode: Optional mode override ("offline" or "online")
            blocking: If True, wait for completion; if False, start in background
            
        Returns:
            Updated TrainingRun
        """
        run.mark_started()
        self.metadata_store.update(run)
        
        # Create temporary config file
        import tempfile
        import tomli_w
        
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(run.config, f)
            temp_config_path = Path(f.name)
        
        try:
            if blocking:
                # Run synchronously
                try:
                    train_func(temp_config_path, mode=mode)
                    run.mark_completed()
                except Exception as e:
                    run.mark_failed(str(e))
                    raise
                finally:
                    self.metadata_store.update(run)
            else:
                # Run asynchronously (spawn subprocess)
                import subprocess
                cmd = [
                    sys.executable,
                    "-m",
                    "prime_rl.train",
                    "--config",
                    str(temp_config_path),
                ]
                if mode:
                    cmd.extend(["--mode", mode])
                
                subprocess.Popen(cmd)
                # Note: In a real implementation, we'd track the subprocess
                # and update run status when it completes
        
        finally:
            # Clean up temp file
            temp_config_path.unlink()
        
        return run
    
    def get_run(self, run_id: str, project_id: Optional[str] = None) -> Optional[TrainingRun]:
        """
        Get a training run by ID.
        
        Args:
            run_id: Run identifier
            project_id: Optional project identifier (uses default if not provided)
            
        Returns:
            TrainingRun if found, None otherwise
        """
        project_id = project_id or self.project_id
        return self.metadata_store.load(project_id, run_id)
    
    def list_runs(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[TrainingRun]:
        """
        List training runs.
        
        Args:
            project_id: Optional project identifier (uses default if not provided)
            status: Optional status filter
            limit: Optional limit on number of runs
            
        Returns:
            List of TrainingRun objects
        """
        project_id = project_id or self.project_id
        return self.metadata_store.list_runs(project_id, status=status, limit=limit)
    
    def cancel_run(self, run_id: str, project_id: Optional[str] = None) -> bool:
        """
        Cancel a training run.
        
        Args:
            run_id: Run identifier
            project_id: Optional project identifier (uses default if not provided)
            
        Returns:
            True if cancelled, False if not found
        """
        project_id = project_id or self.project_id
        run = self.metadata_store.load(project_id, run_id)
        if run:
            run.mark_cancelled()
            self.metadata_store.update(run)
            return True
        return False

