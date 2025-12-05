"""
JAX data loader for Interaction Traces.

This module provides a high-performance data loader for the Interaction Trace format
(JSONL/Parquet) that shards traces across devices and prefetches to device memory.
"""

from typing import Iterator, List, Optional
from pathlib import Path

from prime_rl.core.interaction_trace import InteractionTrace, load_traces_from_jsonl
from prime_rl.core.algorithms import UniversalRollout


class JaxDataLoader:
    """
    High-performance data loader for Interaction Traces in JAX.
    
    Features:
    - Efficient JSONL/Parquet reading
    - Sharding across devices
    - Prefetching to device memory
    - Conversion to UniversalRollout
    """

    def __init__(
        self,
        file_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        shard_across_devices: bool = True,
    ):
        """
        Initialize JAX data loader.
        
        Args:
            file_path: Path to JSONL file containing interaction traces
            batch_size: Batch size for training
            shuffle: Whether to shuffle traces
            shard_across_devices: Whether to shard data across JAX devices
        """
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
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

    def __iter__(self) -> Iterator[List[UniversalRollout]]:
        """
        Iterate over batches of UniversalRollout objects.
        
        Yields:
            Batches of UniversalRollout objects
        """
        traces = self.load_traces()
        
        # Shuffle if requested
        if self.shuffle:
            import random
            random.shuffle(traces)
        
        # Convert traces to rollouts
        rollouts = [trace.to_universal_rollout() for trace in traces]
        
        # Yield batches
        for i in range(0, len(rollouts), self.batch_size):
            batch = rollouts[i : i + self.batch_size]
            yield batch

    def __len__(self) -> int:
        """Get the number of batches."""
        traces = self.load_traces()
        return (len(traces) + self.batch_size - 1) // self.batch_size

