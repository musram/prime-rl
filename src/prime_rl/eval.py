"""
Evaluation-only entrypoint for PRIME-RL.

This module provides the `prime-rl eval` CLI command for running evaluation
without training, as specified in the gap analysis.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import tomli

from prime_rl.core.config import UnifiedConfig
from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.core.verifier import VerifierClient
from prime_rl.core.eval_to_prod import EvalToProdTracker
from prime_rl.core.training_run import TrainingRun
from prime_rl.core.metadata_store import MetadataStore
from prime_rl.core.registry import create_environment_adapter, create_verifier_client
from prime_rl.utils.logger import setup_logger
from loguru import logger


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory or file
        
    Returns:
        Dictionary with checkpoint data
    """
    # Stub implementation - in production would load actual model weights
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    return {"checkpoint_path": str(checkpoint_path)}


def run_evaluation(
    config: UnifiedConfig,
    checkpoint_path: Path,
    num_episodes: int = 10,
    project_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run evaluation on a trained model.
    
    Args:
        config: UnifiedConfig instance
        checkpoint_path: Path to model checkpoint
        num_episodes: Number of evaluation episodes
        project_id: Optional project identifier
        run_id: Optional run identifier
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Create environment adapter
    if config.environment is None:
        raise ValueError("Environment configuration required for evaluation")
    
    env_config = config.environment.dict()
    env_type = env_config.pop("type", "unknown")
    env_adapter = create_environment_adapter(env_type, **env_config)
    
    # Create verifier client
    verifier_client = None
    if config.verifier:
        verifier_config = config.verifier.dict()
        verifier_type = verifier_config.pop("type", "unknown")
        verifier_client = create_verifier_client(verifier_type, **verifier_config)
    
    # Setup tracking
    tracker = None
    if project_id and run_id:
        metadata_store = MetadataStore()
        run = metadata_store.load_run(run_id)
        if run:
            tracker = EvalToProdTracker(run)
    
    # Run evaluation episodes
    logger.info(f"Running evaluation: {num_episodes} episodes")
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    for episode_idx in range(num_episodes):
        obs, info = env_adapter.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # In a real implementation, would use the loaded model to generate actions
            # For now, this is a stub that demonstrates the evaluation loop
            action = {"action_type": "default"}  # Placeholder
            
            obs, reward, terminated, truncated, info = env_adapter.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # Verify if verifier is available
            if verifier_client:
                try:
                    result = verifier_client.verify(obs, action)
                    # Could use verifier reward instead of env reward
                    pass
                except Exception as e:
                    logger.warning(f"Verification failed: {e}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        successes.append(info.get("success", False))
        
        logger.info(
            f"Episode {episode_idx + 1}/{num_episodes}: "
            f"reward={episode_reward:.2f}, length={episode_length}, success={successes[-1]}"
        )
        
        # Log to tracker
        if tracker:
            tracker.log_eval_metrics(
                step=episode_idx,
                eval_metrics={
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "success": 1.0 if successes[-1] else 0.0,
                },
            )
    
    # Compute summary metrics
    metrics = {
        "mean_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0,
        "mean_length": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0,
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "num_episodes": num_episodes,
    }
    
    logger.info("Evaluation completed")
    logger.info(f"Mean reward: {metrics['mean_reward']:.2f}")
    logger.info(f"Mean length: {metrics['mean_length']:.2f}")
    logger.info(f"Success rate: {metrics['success_rate']:.2%}")
    
    # Cleanup
    env_adapter.close()
    if tracker:
        tracker.finish()
    
    return metrics


def eval_command(
    config_path: Path,
    checkpoint_path: Path,
    num_episodes: int = 10,
    project_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """
    Evaluation command entrypoint.
    
    Args:
        config_path: Path to TOML configuration file
        checkpoint_path: Path to model checkpoint
        num_episodes: Number of evaluation episodes
        project_id: Optional project identifier
        run_id: Optional run identifier
    """
    # Load configuration
    with open(config_path, "rb") as f:
        config_dict = tomli.load(f)
    
    config = UnifiedConfig(**config_dict)
    
    # Setup logger
    log_file = config.output_dir / "logs" / "eval.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logger("info", log_file=str(log_file))
    
    logger.info("PRIME-RL Evaluation")
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Episodes: {num_episodes}")
    
    # Run evaluation
    metrics = run_evaluation(
        config=config,
        checkpoint_path=checkpoint_path,
        num_episodes=num_episodes,
        project_id=project_id,
        run_id=run_id,
    )
    
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")


def main() -> None:
    """
    CLI entrypoint for `prime-rl eval`.
    
    Usage:
        prime-rl eval --config path/to/config.toml --checkpoint path/to/ckpt [--num-episodes N] [--project-id PROJECT] [--run-id RUN]
    """
    parser = argparse.ArgumentParser(description="PRIME-RL evaluation CLI")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint directory or file",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="Project identifier for tracking",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier for tracking",
    )
    
    args = parser.parse_args()
    
    eval_command(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        project_id=args.project_id,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()

