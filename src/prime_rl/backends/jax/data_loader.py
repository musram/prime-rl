"""
JAX data loader for Interaction Traces.

This module provides a high-performance data loader for the Interaction Trace format
(JSONL/Parquet) that shards traces across devices and prefetches to device memory.
"""

from typing import Iterator, List, Optional, Dict, Any
from pathlib import Path
import random

try:
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from prime_rl.core.interaction_trace import InteractionTrace, load_traces_from_jsonl
from prime_rl.core.algorithms import UniversalRollout


class JaxDataLoader:
    """
    High-performance data loader for Interaction Traces in JAX.
    
    Features:
    - Efficient JSONL/Parquet reading
    - Tokenization of prompts and completions
    - Conversion to JAX arrays
    - Sharding across devices (TODO: pmap/pjit support)
    - Prefetching to device memory
    - Conversion to UniversalRollout
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: Any,  # AutoTokenizer when transformers available
        batch_size: int = 32,
        shuffle: bool = True,
        max_length: int = 512,
        shard_across_devices: bool = True,
    ):
        """
        Initialize JAX data loader.
        
        Args:
            file_path: Path to JSONL file containing interaction traces
            tokenizer: HuggingFace tokenizer instance
            batch_size: Batch size for training
            shuffle: Whether to shuffle traces
            max_length: Maximum sequence length for tokenization
            shard_across_devices: Whether to shard data across JAX devices (TODO)
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for JaxDataLoader")
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length = max_length
        self.shard_across_devices = shard_across_devices
        self.traces: Optional[List[InteractionTrace]] = None

    def load_traces(self) -> List[InteractionTrace]:
        """
        Load all traces from the JSONL file.
        
        Returns:
            List of InteractionTrace objects
        """
        if self.traces is None:
            self.traces = load_traces_from_jsonl(str(self.file_path))
        return self.traces

    def _tokenize_batch(self, texts: List[str], is_completion: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of text strings
            is_completion: Whether these are completions (for label creation)
            
        Returns:
            Dictionary with 'input_ids' and 'labels' arrays
        """
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        
        input_ids = jnp.array(tokenized["input_ids"])
        
        # Create labels (same as input_ids for completions, -100 for padding)
        if is_completion:
            attention_mask = jnp.array(tokenized["attention_mask"])
            labels = jnp.where(attention_mask == 1, input_ids, -100)
        else:
            # For prompts, labels are -100 (we don't compute loss on prompts)
            labels = jnp.full_like(input_ids, -100)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def _create_preference_pairs(self, rollouts: List[UniversalRollout]) -> List[Dict[str, Any]]:
        """
        Create preference pairs from rollouts for DPO training.
        
        For now, creates pairs by comparing rewards. Traces with higher rewards
        are treated as "chosen" and lower rewards as "rejected".
        
        Args:
            rollouts: List of UniversalRollout objects
            
        Returns:
            List of preference pair dictionaries
        """
        pairs = []
        
        if len(rollouts) < 2:
            # Need at least 2 rollouts to create a pair
            return pairs
        
        # Simple strategy: pair rollouts by reward
        # Sort by reward and pair high with low
        sorted_rollouts = sorted(rollouts, key=lambda r: r.rewards[0] if r.rewards and len(r.rewards) > 0 else 0.0, reverse=True)
        
        # Create pairs: high reward (chosen) vs low reward (rejected)
        for i in range(0, len(sorted_rollouts) - 1, 2):
            if i + 1 < len(sorted_rollouts):
                chosen = sorted_rollouts[i]
                rejected = sorted_rollouts[i + 1]
                
                # Get rewards (default to 0.0 if missing)
                chosen_reward = chosen.rewards[0] if chosen.rewards and len(chosen.rewards) > 0 else 0.0
                rejected_reward = rejected.rewards[0] if rejected.rewards and len(rejected.rewards) > 0 else 0.0
                
                # Skip if rewards are equal (no preference)
                if chosen_reward > rejected_reward:
                    pairs.append({
                        "chosen_prompt": chosen.prompts[0] if chosen.prompts else "",
                        "chosen_completion": chosen.completions[0] if chosen.completions else "",
                        "rejected_prompt": rejected.prompts[0] if rejected.prompts else "",
                        "rejected_completion": rejected.completions[0] if rejected.completions else "",
                    })
        
        return pairs

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Iterate over batches of tokenized preference pairs.
        
        Yields:
            Batches containing:
                - chosen_input_ids: Tokenized chosen completions
                - rejected_input_ids: Tokenized rejected completions
                - chosen_labels: Labels for chosen
                - rejected_labels: Labels for rejected
        """
        traces = self.load_traces()
        
        # Shuffle if requested
        if self.shuffle:
            random.shuffle(traces)
        
        # Convert traces to rollouts
        rollouts = [trace.to_universal_rollout() for trace in traces]
        
        # Create preference pairs
        pairs = self._create_preference_pairs(rollouts)
        
            # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            
            if not batch_pairs:
                continue
            
            # Ensure we have at least one pair
            if len(batch_pairs) == 0:
                continue
            
            # Extract prompts and completions
            chosen_prompts = [p["chosen_prompt"] for p in batch_pairs]
            chosen_completions = [p["chosen_completion"] for p in batch_pairs]
            rejected_prompts = [p["rejected_prompt"] for p in batch_pairs]
            rejected_completions = [p["rejected_completion"] for p in batch_pairs]
            
            # Combine prompt + completion for full sequences
            chosen_texts = [p + c for p, c in zip(chosen_prompts, chosen_completions)]
            rejected_texts = [p + c for p, c in zip(rejected_prompts, rejected_completions)]
            
            # Tokenize
            chosen_tokenized = self._tokenize_batch(chosen_texts, is_completion=True)
            rejected_tokenized = self._tokenize_batch(rejected_texts, is_completion=True)
            
            # Create batch dictionary
            batch = {
                "chosen_input_ids": chosen_tokenized["input_ids"],
                "chosen_labels": chosen_tokenized["labels"],
                "rejected_input_ids": rejected_tokenized["input_ids"],
                "rejected_labels": rejected_tokenized["labels"],
            }
            
            yield batch

    def __len__(self) -> int:
        """Get the number of batches."""
        traces = self.load_traces()
        rollouts = [trace.to_universal_rollout() for trace in traces]
        pairs = self._create_preference_pairs(rollouts)
        return (len(pairs) + self.batch_size - 1) // self.batch_size
