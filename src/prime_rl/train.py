"""
Unified training entrypoint for PRIME-RL.

This module provides the `prime-rl train` CLI command that supports both offline
and online RL training with unified configuration as specified in the PRD (§4.3).
"""

import argparse
from pathlib import Path
from typing import Optional
import tomli

from prime_rl.core.config import UnifiedConfig
from prime_rl.core.interaction_trace import load_traces_from_jsonl
from prime_rl.backends.jax import JaxTrainer, JaxDPO
from prime_rl.backends.jax.data_loader import JaxDataLoader
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
    Run online RL training with PyTorch backend (Phase 2 - not yet implemented).
    
    Args:
        config: Unified configuration
    """
    raise NotImplementedError("Online PyTorch training will be implemented in Phase 2")


def train(config_path: Path, mode: Optional[str] = None) -> None:
    """
    Main training function.
    
    Args:
        config_path: Path to TOML configuration file
        mode: Optional mode override ("offline" or "online")
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
    
    # Route to appropriate backend/mode
    if config.backend.type == "jax" and config.backend.mode == "offline":
        train_offline_jax(config)
    elif config.backend.type == "torch" and config.backend.mode == "online":
        train_online_torch(config)
    else:
        raise ValueError(
            f"Unsupported backend/mode combination: {config.backend.type}/{config.backend.mode}. "
            f"Phase 1 supports: jax/offline"
        )


def main() -> None:
    """
    CLI entrypoint for `prime-rl train`.
    
    Usage:
        prime-rl train --mode offline --config path/to/config.toml
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
    
    args = parser.parse_args()
    
    train(config_path=args.config, mode=args.mode)


if __name__ == "__main__":
    main()

