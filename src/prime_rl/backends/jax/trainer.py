"""
JAX trainer for offline RL.

This module implements the JaxTrainer class that manages the JAX training loop
with device mesh support (pmap/pjit/shard_map) for high-throughput offline RL.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import pickle
import json

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create dummy types for type checking
    jax = None  # type: ignore
    jnp = None  # type: ignore
    random = None  # type: ignore

from prime_rl.core.trainer import Trainer
from prime_rl.core.algorithms import TrainingMetrics
from loguru import logger


class JaxTrainer(Trainer):
    """
    JAX-based trainer for offline RL.
    
    Manages the training loop with JAX device mesh support (pmap/pjit/shard_map),
    mixed precision, and gradient accumulation for large-batch training.
    
    This implements the Trainer interface for the JAX backend.
    
    TODO: Add pmap/pjit/shard_map support for multi-device training
    TODO: Add mixed precision support
    TODO: Add gradient accumulation for large batches
    """

    def __init__(
        self,
        algorithm: Any,  # JaxDPO or other JaxAlgorithm
        config: Dict[str, Any],
        data_loader: Any,  # JaxDataLoader
        rng_key: Optional[Any] = None,  # jax.random.PRNGKey when JAX is available
    ):
        """
        Initialize JAX trainer.
        
        Args:
            algorithm: JAX algorithm instance (e.g., JaxDPO)
            config: Training configuration dictionary
            data_loader: JaxDataLoader instance
            rng_key: Optional random key (generated if not provided)
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for JaxTrainer. Install with: pip install jax jaxlib"
            )
        self.algorithm = algorithm
        self.config = config
        self.data_loader = data_loader
        self.rng_key = rng_key if rng_key is not None else random.PRNGKey(0)
        self.step = 0
        self.state = None
        self.output_dir = Path(config.get("output_dir", "./output"))
        self.max_steps = config.get("max_steps")
        self.checkpoint_every = config.get("checkpoint_every")
        self.eval_every = config.get("eval_every")

    def run_training_loop(self) -> None:
        """
        Run the main JAX training loop.
        
        This implementation:
        - Initializes algorithm state (model, optimizer)
        - Loads dataset via JaxDataLoader
        - Iterates over batches
        - Calls algorithm.train_step() for each batch
        - Handles checkpointing, logging, and evaluation hooks
        """
        logger.info("Initializing JAX training loop")
        
        # Initialize algorithm state
        self.rng_key, init_key = random.split(self.rng_key)
        algorithm_config = {
            **self.config.get("model", {}),
            "learning_rate": self.algorithm.config.get("learning_rate", 1e-5),
            "beta": self.algorithm.config.get("beta", 0.1),
        }
        self.state = self.algorithm.init_state(init_key, algorithm_config)
        
        logger.info(f"Initialized algorithm state at step {self.state['step']}")
        logger.info(f"Model: {self.algorithm.model_name}")
        
        # Training loop
        step = self.state["step"]
        max_steps = self.max_steps if self.max_steps else float("inf")
        
        logger.info(f"Starting training loop (max_steps={max_steps if max_steps != float('inf') else 'unlimited'})")
        
        # Check if data loader has any batches
        if len(self.data_loader) == 0:
            logger.warning("Data loader has no batches. Check dataset and preference pair creation.")
            return
        
        try:
            for batch in self.data_loader:
                if step >= max_steps:
                    logger.info(f"Reached max_steps ({max_steps}), stopping training")
                    break
                
                # Skip empty batches
                if batch["chosen_input_ids"].shape[0] == 0:
                    logger.warning(f"Skipping empty batch at step {step}")
                    continue
                
                # Training step
                self.state, metrics = self.algorithm.train_step(self.state, batch)
                step = self.state["step"]
                
                # Convert metrics to dict
                if hasattr(metrics, "to_dict"):
                    metrics_dict = metrics.to_dict()
                elif hasattr(metrics, "__dict__"):
                    metrics_dict = metrics.__dict__
                else:
                    metrics_dict = {"loss": getattr(metrics, "loss", 0.0)}
                
                # Log metrics
                self.log_metrics(step, metrics_dict)
                
                # Checkpoint
                if self.checkpoint_every and step % self.checkpoint_every == 0:
                    self.checkpoint(step, metrics_dict)
                
                # Evaluate
                if self.eval_every and step % self.eval_every == 0:
                    eval_metrics = self.evaluate(step)
                    if eval_metrics:
                        logger.info(f"Evaluation at step {step}: {eval_metrics}")
        
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error at step {step}: {e}")
            raise
        finally:
            # Final checkpoint
            if self.state:
                final_metrics = {
                    "step": step,
                    "loss": 0.0,  # Would get from last step
                }
                self.checkpoint(step, final_metrics)
        
        logger.info(f"Training completed at step {step}")

    def checkpoint(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Save a checkpoint.
        
        Args:
            step: Current training step
            metrics: Current training metrics
        """
        checkpoint_dir = self.output_dir / "checkpoints" / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model parameters
        params_file = checkpoint_dir / "params.pkl"
        with open(params_file, "wb") as f:
            # Convert JAX arrays to numpy for serialization
            params_np = jax.tree_map(lambda x: jnp.array(x) if isinstance(x, jnp.ndarray) else x, self.state["params"])
            pickle.dump(params_np, f)
        
        # Save optimizer state
        opt_state_file = checkpoint_dir / "opt_state.pkl"
        with open(opt_state_file, "wb") as f:
            opt_state_np = jax.tree_map(lambda x: jnp.array(x) if isinstance(x, jnp.ndarray) else x, self.state["opt_state"])
            pickle.dump(opt_state_np, f)
        
        # Save metadata
        metadata = {
            "step": step,
            "metrics": metrics,
            "config": self.config,
        }
        metadata_file = checkpoint_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved at step {step} to {checkpoint_dir}")

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Log training metrics.
        
        Args:
            step: Current training step
            metrics: Metrics to log
        """
        # Log to console
        loss = metrics.get("loss", 0.0)
        grad_norm = metrics.get("grad_norm", 0.0)
        lr = metrics.get("learning_rate", 0.0)
        
        logger.info(
            f"Step {step} | Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f} | LR: {lr:.2e}"
        )
        
        # Log to file
        log_file = self.output_dir / "logs" / "training_metrics.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        log_entry = {"step": step, **metrics}
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def evaluate(self, step: int) -> Optional[Dict[str, Any]]:
        """
        Run evaluation.
        
        Args:
            step: Current training step
            
        Returns:
            Evaluation metrics dictionary, or None if evaluation is not configured
        """
        # TODO: Implement evaluation on validation set
        # For now, return None
        return None
