"""
JAX implementation of Direct Preference Optimization (DPO).

This module implements DPO for offline RL using JAX, following the PRD specification (§2.1).
DPO learns from preference data by optimizing a policy to prefer chosen completions over rejected ones.
"""

from typing import Any, Dict, Tuple, Optional
import functools

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import optax
    import flax
    from flax import linen as nn
    from flax.training import train_state
    from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Create dummy types for type checking
    jax = None  # type: ignore
    jnp = None  # type: ignore
    random = None  # type: ignore
    optax = None  # type: ignore
    flax = None  # type: ignore
    nn = None  # type: ignore
    train_state = None  # type: ignore
    FlaxAutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from prime_rl.core.algorithms import OfflineRLAlgorithm, TrainingMetrics, LossDict


class JaxDPO(OfflineRLAlgorithm):
    """
    JAX implementation of Direct Preference Optimization (DPO).
    
    DPO is an offline RL algorithm that learns from preference data (e.g., human feedback
    or preference comparisons). This implementation uses JAX/Flax for high-throughput batch processing.
    
    DPO loss formulation:
    L_DPO = -log(sigma(beta * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
    
    where:
    - y_w: chosen (winning) completion
    - y_l: rejected (losing) completion
    - π_θ: policy model
    - π_ref: reference model (typically frozen)
    - beta: temperature parameter
    - sigma: sigmoid function
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize JAX DPO algorithm.
        
        Args:
            config: Algorithm configuration containing:
                - learning_rate: Learning rate for optimizer
                - beta: DPO temperature parameter (default: 0.1)
                - model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B")
                - trust_remote_code: Whether to trust remote code (default: False)
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for JaxDPO. Install with: pip install jax jaxlib optax transformers[flax]"
            )
        self.config = config
        self.beta = config.get("beta", 0.1)
        self.learning_rate = config.get("learning_rate", 1e-5)
        self.model_name = config.get("model_name", "gpt2")  # Default to small model for testing
        self.trust_remote_code = config.get("trust_remote_code", False)
        self.reference_model_name = config.get("reference_model_name", None)  # Optional separate reference model
        self.dtype = config.get("dtype", jnp.float32)  # Training dtype
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_state(self, rng: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize algorithm state (model parameters, optimizer state, etc.).
        
        Args:
            rng: Random number generator key
            config: Configuration dictionary (may override model_name, etc.)
            
        Returns:
            Initial algorithm state dictionary containing:
                - params: Model parameters
                - ref_params: Reference model parameters (frozen)
                - opt_state: Optimizer state
                - step: Current step
                - rng: Random key
        """
        # Use model_name from config if provided, otherwise use instance default
        model_name = config.get("model_name", self.model_name)
        trust_remote_code = config.get("trust_remote_code", self.trust_remote_code)
        reference_model_name = config.get("reference_model_name", self.reference_model_name)
        dtype = config.get("dtype", self.dtype)
        
        # Convert dtype string to JAX dtype if needed
        if isinstance(dtype, str):
            from prime_rl.backends.jax.utils import get_dtype
            dtype = get_dtype(dtype)
        
        # Load policy model
        try:
            model = FlaxAutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
            )
        except Exception as e:
            # Fallback to GPT2 if model loading fails
            import warnings
            warnings.warn(f"Failed to load {model_name}, falling back to gpt2: {e}")
            model = FlaxAutoModelForCausalLM.from_pretrained(
                "gpt2",
                dtype=dtype,
            )
        
        # Load reference model (separate or same as policy)
        if reference_model_name and reference_model_name != model_name:
            # Load separate reference model
            try:
                ref_model = FlaxAutoModelForCausalLM.from_pretrained(
                    reference_model_name,
                    trust_remote_code=trust_remote_code,
                    dtype=dtype,
                )
                if hasattr(ref_model, "params"):
                    ref_params = ref_model.params
                else:
                    # Fallback: initialize with dummy input
                    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
                    rng, init_rng = random.split(rng)
                    variables = ref_model.init(init_rng, dummy_input, train=True)
                    ref_params = variables["params"]
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load reference model {reference_model_name}, using policy model: {e}")
                # Fall back to using policy model as reference
                if hasattr(model, "params"):
                    ref_params = model.params
                else:
                    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
                    rng, init_rng = random.split(rng)
                    variables = model.init(init_rng, dummy_input, train=True)
                    ref_params = variables["params"]
        else:
            # Use policy model as reference (default behavior)
            if hasattr(model, "params"):
                ref_params = model.params
            else:
                dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
                rng, init_rng = random.split(rng)
                variables = model.init(init_rng, dummy_input, train=True)
                ref_params = variables["params"]
        
        # Initialize optimizer
        optimizer = optax.adamw(
            learning_rate=self.learning_rate,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.01,
        )
        
        # Get model parameters
        # Flax models loaded from_pretrained have params attribute
        # If not available, initialize with dummy input
        if hasattr(model, "params"):
            params = model.params
        else:
            # Fallback: initialize with dummy input
            dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
            rng, init_rng = random.split(rng)
            variables = model.init(init_rng, dummy_input, train=True)
            params = variables["params"]
        
        # Freeze reference model params (ensure they're never updated)
        ref_params = flax.core.freeze(flax.core.unfreeze(ref_params))
        
        # Initialize optimizer state
        opt_state = optimizer.init(params)
        
        return {
            "params": params,
            "ref_params": ref_params,
            "opt_state": opt_state,
            "step": 0,
            "rng": rng,
            "model": model,
            "optimizer": optimizer,
        }

    def compute_loss(
        self,
        params: Any,
        ref_params: Any,
        model: Any,
        chosen_input_ids: jnp.ndarray,
        rejected_input_ids: jnp.ndarray,
        chosen_labels: jnp.ndarray,
        rejected_labels: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute DPO loss given preference pairs.
        
        Args:
            params: Policy model parameters
            ref_params: Reference model parameters (frozen)
            model: Flax model instance
            chosen_input_ids: Tokenized chosen completions [batch, seq_len]
            rejected_input_ids: Tokenized rejected completions [batch, seq_len]
            chosen_labels: Labels for chosen (same as input_ids, -100 for padding) [batch, seq_len]
            rejected_labels: Labels for rejected [batch, seq_len]
            
        Returns:
            Tuple of (loss, loss_dict) where loss_dict contains component losses
        """
        def get_logprobs(input_ids: jnp.ndarray, labels: jnp.ndarray, model_params: Any) -> jnp.ndarray:
            """Compute log probabilities for a sequence."""
            # Forward pass
            outputs = model.apply(
                {"params": model_params},
                input_ids,
                train=False,
            )
            logits = outputs.logits
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            
            # Compute log probs: log_softmax then gather
            log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
            
            # Gather log probs for actual tokens
            batch_size, seq_len = shift_labels.shape
            batch_indices = jnp.arange(batch_size)[:, None]
            seq_indices = jnp.arange(seq_len)[None, :]
            
            # Handle out-of-bounds (shouldn't happen, but be safe)
            valid_labels = jnp.clip(shift_labels, 0, log_probs.shape[-1] - 1)
            token_logprobs = log_probs[batch_indices, seq_indices, valid_labels]
            
            # Mask out padding tokens (label == -100)
            mask = (shift_labels != -100).astype(jnp.float32)
            token_logprobs = token_logprobs * mask
            
            # Sum over sequence length
            return jnp.sum(token_logprobs, axis=1)
        
        # Compute log probs for chosen and rejected with policy model
        chosen_logprobs = get_logprobs(chosen_input_ids, chosen_labels, params)
        rejected_logprobs = get_logprobs(rejected_input_ids, rejected_labels, params)
        
        # Compute log probs for chosen and rejected with reference model
        chosen_ref_logprobs = get_logprobs(chosen_input_ids, chosen_labels, ref_params)
        rejected_ref_logprobs = get_logprobs(rejected_input_ids, rejected_labels, ref_params)
        
        # DPO loss: -log(sigma(beta * (log π_θ(y_w) - log π_θ(y_l) - log π_ref(y_w) + log π_ref(y_l))))
        log_ratio = (chosen_logprobs - rejected_logprobs) - (chosen_ref_logprobs - rejected_ref_logprobs)
        loss = -jax.nn.log_sigmoid(self.beta * log_ratio)
        loss = jnp.mean(loss)
        
        # Reference KL divergence (for monitoring)
        ref_kl = jnp.mean((chosen_logprobs - chosen_ref_logprobs))
        
        loss_dict = {
            "total": loss,
            "dpo": loss,
            "reference_kl": ref_kl,
            "chosen_logprobs": jnp.mean(chosen_logprobs),
            "rejected_logprobs": jnp.mean(rejected_logprobs),
        }
        
        return loss, loss_dict

    def train_step(
        self, state: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], TrainingMetrics]:
        """
        Perform a single DPO training step.
        
        Args:
            state: Current algorithm state
            batch: Training batch containing:
                - chosen_input_ids: Tokenized chosen completions
                - rejected_input_ids: Tokenized rejected completions
                - chosen_labels: Labels for chosen
                - rejected_labels: Labels for rejected
                
        Returns:
            Tuple of (updated_state, metrics)
        """
        params = state["params"]
        ref_params = state["ref_params"]
        model = state["model"]
        optimizer = state["optimizer"]
        opt_state = state["opt_state"]
        rng = state["rng"]
        step = state["step"]
        
        # Compute loss and gradients
        def loss_fn(p):
            loss, loss_dict = self.compute_loss(
                p,
                ref_params,
                model,
                batch["chosen_input_ids"],
                batch["rejected_input_ids"],
                batch["chosen_labels"],
                batch["rejected_labels"],
            )
            return loss, loss_dict
        
        # Get gradients
        (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Compute gradient norm
        grad_norm = optax.global_norm(grads)
        
        # Update state
        rng, _ = random.split(rng)
        new_state = {
            **state,
            "params": new_params,
            "opt_state": new_opt_state,
            "step": step + 1,
            "rng": rng,
        }
        
        # Create metrics
        metrics = TrainingMetrics(
            loss=float(loss),
            grad_norm=float(grad_norm),
            learning_rate=self.learning_rate,
            step=step + 1,
            framework_specific={
                "dpo_loss": float(loss_dict["dpo"]),
                "reference_kl": float(loss_dict["reference_kl"]),
                "chosen_logprobs": float(loss_dict["chosen_logprobs"]),
                "rejected_logprobs": float(loss_dict["rejected_logprobs"]),
            },
        )
        
        return new_state, metrics

    def process_dataset(self, dataset: Any) -> Any:
        """
        Pre-process a raw dataset (interaction traces) into DPO format.
        
        DPO requires preference pairs. This method extracts preference pairs from traces
        that have preference_group_id in their labels.
        
        Args:
            dataset: Raw dataset (list of InteractionTrace objects)
            
        Returns:
            Processed dataset with preference pairs
        """
        # For now, return dataset as-is
        # Full implementation would:
        # - Group traces by preference_group_id
        # - Create pairs of (chosen, rejected) based on rewards or labels
        # - Convert to tokenized format
        # TODO: Implement preference pair extraction from traces
        return dataset
