"""
Unified training entrypoint for PRIME-RL.

This module provides the `prime-rl train` CLI command that supports both offline
and online RL training with unified configuration as specified in the PRD (§4.3).
"""

import argparse
from pathlib import Path
from typing import Optional
import tomli
import subprocess

from prime_rl.core.config import UnifiedConfig
from prime_rl.core.interaction_trace import load_traces_from_jsonl
from prime_rl.core.training_run import TrainingRun
from prime_rl.core.metadata_store import MetadataStore
from prime_rl.core.eval_to_prod import EvalToProdTracker
from prime_rl.backends.jax import JaxTrainer, JaxDPO
from prime_rl.backends.jax.data_loader import JaxDataLoader
from prime_rl.backends.torch import TorchTrainer, TorchPPO, TorchGRPO
from prime_rl.utils.logger import setup_logger
from loguru import logger


def load_config(config_path: Path) -> UnifiedConfig:
    """
    Load unified configuration from TOML file.
    
    Args:
        config_path: Path to TOML configuration file
        
    Returns:
        UnifiedConfig instance
    """
    with open(config_path, "rb") as f:
        config_dict = tomli.load(f)
    
    # Handle both old format (output_dir at top level) and new format (output section)
    if "output" not in config_dict and any(k in config_dict for k in ["output_dir", "max_steps", "checkpoint_every", "eval_every"]):
        # Convert old format to new format
        output_dict = {}
        for key in ["output_dir", "max_steps", "checkpoint_every", "eval_every"]:
            if key in config_dict:
                output_dict[key] = config_dict.pop(key)
        if output_dict:
            config_dict["output"] = output_dict
    
    return UnifiedConfig(**config_dict)


def train_offline_jax(config: UnifiedConfig) -> None:
    """
    Run offline RL training with JAX backend.
    
    Args:
        config: Unified configuration
    """
    logger.info("Starting offline RL training with JAX backend")
    
    # Validate configuration
    if config.dataset is None:
        raise ValueError("Dataset configuration required for offline mode")
    
    if config.model is None:
        raise ValueError("Model configuration required")
    
    # Load dataset
    logger.info(f"Loading dataset from {config.dataset.path}")
    data_loader = JaxDataLoader(
        file_path=str(config.dataset.path),
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
    )
    
    # Initialize algorithm
    algorithm_config = {
        "learning_rate": config.algorithm.learning_rate,
        "beta": config.algorithm.beta,
    }
    
    if config.algorithm.name == "dpo":
        algorithm = JaxDPO(algorithm_config)
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm.name}")
    
    # Initialize trainer
    trainer_config = {
        "max_steps": config.max_steps,
        "checkpoint_every": config.checkpoint_every,
        "eval_every": config.eval_every,
        "output_dir": str(config.output_dir),
        "model": config.model.dict() if config.model else None,
    }
    
    trainer = JaxTrainer(algorithm=algorithm, config=trainer_config)
    
    # Run training loop
    logger.info("Starting training loop")
    trainer.run_training_loop()
    logger.info("Training completed")


def train_online_torch(config: UnifiedConfig) -> None:
    """
    Run online RL training with PyTorch backend.
    
    Args:
        config: Unified configuration
    """
    logger.info("Starting online RL training with PyTorch backend")
    
    # Validate configuration
    if config.model is None:
        raise ValueError("Model configuration required")
    
    # Initialize algorithm
    algorithm_config = {
        "learning_rate": config.algorithm.learning_rate,
    }
    
    # Add algorithm-specific config
    if config.algorithm.name == "ppo":
        algorithm_config.update({
            "clip_epsilon": config.algorithm.clip_epsilon or 0.2,
            "value_coef": config.algorithm.value_coef or 0.5,
            "entropy_coef": config.algorithm.entropy_coef or 0.01,
        })
        if config.algorithm.config:
            algorithm_config.update(config.algorithm.config)
        algorithm = TorchPPO(algorithm_config)
    elif config.algorithm.name == "grpo":
        if config.algorithm.config:
            algorithm_config.update(config.algorithm.config)
        algorithm = TorchGRPO(algorithm_config)
    else:
        raise ValueError(f"Unsupported algorithm for online mode: {config.algorithm.name}")
    
    # Initialize trainer
    trainer_config = {
        "max_steps": config.max_steps,
        "checkpoint_every": config.checkpoint_every,
        "eval_every": config.eval_every,
        "output_dir": str(config.output_dir),
        "model": config.model.dict() if config.model else None,
    }
    
    trainer = TorchTrainer(algorithm=algorithm, config=trainer_config)
    
    # Run training loop
    logger.info("Starting training loop")
    trainer.run_training_loop()
    logger.info("Training completed")


def train(
    config_path: Path,
    mode: Optional[str] = None,
    project_id: Optional[str] = None,
    run_id: Optional[str] = None,
    enable_tracking: bool = True,
) -> Optional[TrainingRun]:
    """
    Main training function.
    
    Args:
        config_path: Path to TOML configuration file
        mode: Optional mode override ("offline" or "online")
        project_id: Optional project identifier for multi-tenancy
        run_id: Optional run identifier (creates new if not provided)
        enable_tracking: Whether to enable eval-to-prod tracking
        
    Returns:
        TrainingRun instance if tracking enabled, None otherwise
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override mode if provided
    if mode is not None:
        config.backend.mode = mode
    
    # Setup logger
    log_file = config.output_dir / "logs" / "train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logger("info", log_file=str(log_file))
    
    logger.info(f"PRIME-RL Training")
    logger.info(f"Backend: {config.backend.type}, Mode: {config.backend.mode}")
    logger.info(f"Algorithm: {config.algorithm.name}")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Create training run if tracking enabled
    run = None
    tracker = None
    if enable_tracking:
        project_id = project_id or "default"
        
        # Get code revision if available
        code_revision = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=config_path.parent,
            )
            if result.returncode == 0:
                code_revision = result.stdout.strip()
        except Exception:
            pass
        
        # Create or load run
        metadata_store = MetadataStore(config.output_dir / "metadata")
        if run_id:
            run = metadata_store.load(project_id, run_id)
            if not run:
                logger.warning(f"Run {run_id} not found, creating new run")
                run = None
        
        if run is None:
            dataset_reference = str(config.dataset.path) if config.dataset else None
            # Convert config to dict
            config_dict = config.model_dump() if hasattr(config, "model_dump") else (
                config.dict() if hasattr(config, "dict") else {}
            )
            run = TrainingRun.create(
                project_id=project_id,
                config=config_dict,
                code_revision=code_revision,
                dataset_reference=dataset_reference,
            )
            metadata_store.save(run)
        
        run.mark_started()
        metadata_store.update(run)
        
        logger.info(f"Training Run ID: {run.run_id}")
        logger.info(f"Project ID: {run.project_id}")
        
        # Initialize eval-to-prod tracker
        try:
            tracker = EvalToProdTracker(run)
        except Exception as e:
            logger.warning(f"Failed to initialize eval-to-prod tracker: {e}")
            tracker = None
    
    try:
        # Route to appropriate backend/mode
        if config.backend.type == "jax" and config.backend.mode == "offline":
            train_offline_jax(config)
        elif config.backend.type == "torch" and config.backend.mode == "online":
            train_online_torch(config)
        else:
            raise ValueError(
                f"Unsupported backend/mode combination: {config.backend.type}/{config.backend.mode}. "
                f"Supported combinations: jax/offline, torch/online"
            )
        
        # Mark run as completed
        if run:
            run.mark_completed()
            if metadata_store:
                metadata_store.update(run)
        
        if tracker:
            tracker.finish()
    
    except Exception as e:
        # Mark run as failed
        if run:
            run.mark_failed(str(e))
            if metadata_store:
                metadata_store.update(run)
        raise
    
    return run


def main() -> None:
    """
    CLI entrypoint for `prime-rl train`.
    
    Usage:
        prime-rl train --mode offline --config path/to/config.toml [--project-id PROJECT] [--run-id RUN] [--no-tracking]
    """
    parser = argparse.ArgumentParser(description="PRIME-RL unified training CLI")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["offline", "online"],
        help="Training mode: offline (from dataset) or online (from environment)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="Project identifier for multi-tenancy",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (creates new if not provided)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable eval-to-prod tracking",
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        mode=args.mode,
        project_id=args.project_id,
        run_id=args.run_id,
        enable_tracking=not args.no_tracking,
    )


if __name__ == "__main__":
    main()

