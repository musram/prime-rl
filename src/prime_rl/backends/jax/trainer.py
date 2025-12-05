"""
JAX trainer for offline RL.

This module implements the JaxTrainer class that manages the JAX training loop
with device mesh support (pmap/pjit/shard_map) for high-throughput offline RL.
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import pickle
import json

try:
    import jax
    import jax.numpy as jnp
    from jax import random, lax
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create dummy types for type checking
    jax = None  # type: ignore
    jnp = None  # type: ignore
    random = None  # type: ignore
    lax = None  # type: ignore
    optax = None  # type: ignore

from prime_rl.core.trainer import Trainer
from prime_rl.core.algorithms import TrainingMetrics
from prime_rl.backends.jax.utils import (
    get_dtype,
    get_num_devices,
    shard_batch,
    aggregate_gradients,
    aggregate_metrics,
)
from loguru import logger


class JaxTrainer(Trainer):
    """
    JAX-based trainer for offline RL.
    
    Manages the training loop with JAX device mesh support (pmap/pjit/shard_map),
    mixed precision, and gradient accumulation for large-batch training.
    
    This implements the Trainer interface for the JAX backend.
    
    Features:
    - Multi-device training with pmap (data parallelism)
    - Mixed precision (float16/bfloat16)
    - Gradient accumulation
    - Evaluation loop on validation dataset
    """

    def __init__(
        self,
        algorithm: Any,  # JaxDPO or other JaxAlgorithm
        config: Dict[str, Any],
        data_loader: Any,  # JaxDataLoader
        rng_key: Optional[Any] = None,  # jax.random.PRNGKey when JAX is available
        validation_data_loader: Optional[Any] = None,  # Optional validation data loader
    ):
        """
        Initialize JAX trainer.
        
        Args:
            algorithm: JAX algorithm instance (e.g., JaxDPO)
            config: Training configuration dictionary containing:
                - use_pmap: Whether to use pmap for multi-device training
                - dtype: Training dtype ("float32", "float16", "bfloat16")
                - gradient_accumulation_steps: Number of micro-batches to accumulate
                - output_dir: Output directory for checkpoints/logs
                - max_steps: Maximum training steps
                - checkpoint_every: Checkpoint every N steps
                - eval_every: Evaluate every N steps
            data_loader: JaxDataLoader instance for training data
            rng_key: Optional random key (generated if not provided)
            validation_data_loader: Optional JaxDataLoader for validation data
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for JaxTrainer. Install with: pip install jax jaxlib"
            )
        self.algorithm = algorithm
        self.config = config
        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader
        self.rng_key = rng_key if rng_key is not None else random.PRNGKey(0)
        self.step = 0
        self.state = None
        self.output_dir = Path(config.get("output_dir", "./output"))
        self.max_steps = config.get("max_steps")
        self.checkpoint_every = config.get("checkpoint_every")
        self.eval_every = config.get("eval_every")
        
        # Training optimizations
        self.use_pmap = config.get("use_pmap", False)
        self.dtype_str = config.get("dtype", "float32")
        self.dtype = get_dtype(self.dtype_str)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        
        # Multi-device setup
        self.num_devices = get_num_devices()
        if self.use_pmap and self.num_devices > 1:
            logger.info(f"Using pmap with {self.num_devices} devices")
        else:
            self.use_pmap = False
            if config.get("use_pmap", False) and self.num_devices == 1:
                logger.warning("use_pmap requested but only 1 device available, disabling pmap")
        
        # Update algorithm config with dtype
        if hasattr(self.algorithm, 'dtype'):
            self.algorithm.dtype = self.dtype

    def run_training_loop(self) -> None:
        """
        Run the main JAX training loop.
        
        This implementation:
        - Initializes algorithm state (model, optimizer)
        - Loads dataset via JaxDataLoader
        - Iterates over batches with gradient accumulation
        - Calls algorithm.train_step() for each batch
        - Handles checkpointing, logging, and evaluation hooks
        - Supports multi-device training with pmap
        """
        logger.info("Initializing JAX training loop")
        logger.info(f"Training dtype: {self.dtype_str}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        # Initialize algorithm state
        self.rng_key, init_key = random.split(self.rng_key)
        algorithm_config = {
            **self.config.get("model", {}),
            "learning_rate": self.algorithm.config.get("learning_rate", 1e-5),
            "beta": self.algorithm.config.get("beta", 0.1),
            "dtype": self.dtype,
        }
        self.state = self.algorithm.init_state(init_key, algorithm_config)
        
        logger.info(f"Initialized algorithm state at step {self.state['step']}")
        logger.info(f"Model: {self.algorithm.model_name}")
        
        # Wrap train_step with pmap if needed
        if self.use_pmap:
            # Create pmapped train_step
            train_step_fn = self._create_pmapped_train_step()
        else:
            train_step_fn = self._create_train_step()
        
        # Training loop
        step = self.state["step"]
        max_steps = self.max_steps if self.max_steps else float("inf")
        
        logger.info(f"Starting training loop (max_steps={max_steps if max_steps != float('inf') else 'unlimited'})")
        
        # Check if data loader has any batches
        if len(self.data_loader) == 0:
            logger.warning("Data loader has no batches. Check dataset and preference pair creation.")
            return
        
        # Gradient accumulation state
        accumulated_grads = None
        accumulated_loss = 0.0
        micro_batch_count = 0
        
        try:
            for batch in self.data_loader:
                if step >= max_steps:
                    logger.info(f"Reached max_steps ({max_steps}), stopping training")
                    break
                
                # Skip empty batches
                if batch["chosen_input_ids"].shape[0] == 0:
                    logger.warning(f"Skipping empty batch at step {step}")
                    continue
                
                # Shard batch across devices if using pmap
                if self.use_pmap:
                    batch = shard_batch(batch, self.num_devices)
                
                # Gradient accumulation: accumulate gradients over micro-batches
                if self.gradient_accumulation_steps > 1:
                    # Compute gradients for this micro-batch
                    grads, loss_dict = self._compute_gradients(self.state, batch)
                    
                    # Accumulate gradients
                    if accumulated_grads is None:
                        accumulated_grads = grads
                    else:
                        accumulated_grads = jax.tree_map(
                            lambda x, y: x + y,
                            accumulated_grads,
                            grads
                        )
                    
                    accumulated_loss += loss_dict.get("total", 0.0)
                    micro_batch_count += 1
                    
                    # Apply gradients only after accumulating enough micro-batches
                    if micro_batch_count < self.gradient_accumulation_steps:
                        continue
                    
                    # Average accumulated gradients
                    accumulated_grads = jax.tree_map(
                        lambda x: x / self.gradient_accumulation_steps,
                        accumulated_grads
                    )
                    accumulated_loss = accumulated_loss / self.gradient_accumulation_steps
                    
                    # Apply optimizer step
                    self.state = self._apply_gradients(self.state, accumulated_grads)
                    step = self.state["step"]
                    
                    # Create metrics dict for logging
                    metrics_dict = {
                        "loss": float(accumulated_loss),
                        "total": float(accumulated_loss),
                    }
                    
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
                    
                    # Reset accumulation state
                    accumulated_grads = None
                    accumulated_loss = 0.0
                    micro_batch_count = 0
                else:
                    # No gradient accumulation: standard training step
                    self.state, metrics = train_step_fn(self.state, batch)
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
                
                step = self.state["step"]
        
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
                    "loss": accumulated_loss if accumulated_loss > 0 else 0.0,
                }
                self.checkpoint(step, final_metrics)
        
        logger.info(f"Training completed at step {step}")

    def _create_train_step(self):
        """Create a standard (non-pmapped) train step function."""
        def train_step(state: Dict[str, Any], batch: Dict[str, Any]) -> tuple[Dict[str, Any], TrainingMetrics]:
            return self.algorithm.train_step(state, batch)
        return train_step

    def _create_pmapped_train_step(self):
        """Create a pmapped train step function for multi-device training."""
        def train_step(state: Dict[str, Any], batch: Dict[str, Any]) -> tuple[Dict[str, Any], TrainingMetrics]:
            # Replicate state across devices
            replicated_state = jax.tree_map(
                lambda x: jnp.array([x] * self.num_devices) if isinstance(x, jnp.ndarray) else x,
                state
            )
            
            # Run pmapped training step
            def pmapped_step(state_shard, batch_shard):
                return self.algorithm.train_step(state_shard, batch_shard)
            
            # Apply pmap
            pmapped_fn = jax.pmap(pmapped_step, axis_name="devices")
            
            # Execute
            new_state_shards, metrics_shards = pmapped_fn(replicated_state, batch)
            
            # Aggregate metrics across devices
            if isinstance(metrics_shards, dict):
                aggregated_metrics = aggregate_metrics(metrics_shards)
            else:
                aggregated_metrics = metrics_shards
            
            # Take first device's state (all should be same after sync)
            new_state = jax.tree_map(lambda x: x[0] if isinstance(x, jnp.ndarray) and x.ndim > 0 else x, new_state_shards)
            
            return new_state, TrainingMetrics(**aggregated_metrics) if isinstance(aggregated_metrics, dict) else aggregated_metrics
        
        return train_step

    def _compute_gradients(self, state: Dict[str, Any], batch: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
        """Compute gradients for a batch (used in gradient accumulation)."""
        # Get loss function
        loss_fn = lambda params: self.algorithm.compute_loss(
            params,
            state["ref_params"],
            state["model"],
            batch["chosen_input_ids"],
            batch["rejected_input_ids"],
            batch["chosen_labels"],
            batch["rejected_labels"],
        )[0]
        
        # Compute gradients
        grads = jax.grad(loss_fn)(state["params"])
        
        # Compute loss for logging
        _, loss_dict = self.algorithm.compute_loss(
            state["params"],
            state["ref_params"],
            state["model"],
            batch["chosen_input_ids"],
            batch["rejected_input_ids"],
            batch["chosen_labels"],
            batch["rejected_labels"],
        )
        
        return grads, loss_dict

    def _apply_gradients(self, state: Dict[str, Any], grads: Any) -> Dict[str, Any]:
        """Apply gradients using optimizer (used in gradient accumulation)."""
        # Aggregate gradients across devices if using pmap
        if self.use_pmap:
            grads = aggregate_gradients(grads)
        
        # Apply optimizer step
        updates, new_opt_state = state["optimizer"].update(grads, state["opt_state"], state["params"])
        new_params = optax.apply_updates(state["params"], updates)
        
        # Update state
        new_state = state.copy()
        new_state["params"] = new_params
        new_state["opt_state"] = new_opt_state
        new_state["step"] = state["step"] + 1
        
        return new_state

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
            "dtype": self.dtype_str,
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
        loss = metrics.get("loss", metrics.get("total", 0.0))
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
        Run evaluation on validation dataset.
        
        Args:
            step: Current training step
            
        Returns:
            Evaluation metrics dictionary, or None if no validation dataset is configured
        """
        if self.validation_data_loader is None:
            return None
        
        logger.info(f"Running evaluation at step {step}")
        
        # Evaluate on validation batches
        eval_losses = []
        eval_loss_dicts = []
        
        try:
            for batch in self.validation_data_loader:
                if batch["chosen_input_ids"].shape[0] == 0:
                    continue
                
                # Compute loss (no gradients)
                _, loss_dict = self.algorithm.compute_loss(
                    self.state["params"],
                    self.state["ref_params"],
                    self.state["model"],
                    batch["chosen_input_ids"],
                    batch["rejected_input_ids"],
                    batch["chosen_labels"],
                    batch["rejected_labels"],
                )
                
                eval_losses.append(float(loss_dict.get("total", 0.0)))
                eval_loss_dicts.append(loss_dict)
            
            if not eval_losses:
                logger.warning("No validation batches found")
                return None
            
            # Aggregate metrics
            avg_loss = sum(eval_losses) / len(eval_losses)
            eval_metrics = {
                "eval_loss": avg_loss,
                "eval_dpo_loss": avg_loss,
                "eval_reference_kl": sum(d.get("reference_kl", 0.0) for d in eval_loss_dicts) / len(eval_loss_dicts),
            }
            
            # Log eval metrics
            self.log_metrics(step, {f"eval_{k}": v for k, v in eval_metrics.items()})
            
            return eval_metrics
        
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return None
