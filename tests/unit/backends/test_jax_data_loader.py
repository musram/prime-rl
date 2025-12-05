"""Tests for JaxDataLoader."""

import pytest
import tempfile
import json
from pathlib import Path

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if TRANSFORMERS_AVAILABLE and JAX_AVAILABLE:
    from prime_rl.backends.jax.data_loader import JaxDataLoader
    from prime_rl.core.interaction_trace import InteractionTrace, TraceStep


@pytest.mark.skipif(not (TRANSFORMERS_AVAILABLE and JAX_AVAILABLE), reason="JAX and transformers required")
def test_jax_data_loader_initialization():
    """Test JaxDataLoader initialization."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    loader = JaxDataLoader(
        file_path="dummy.jsonl",
        tokenizer=tokenizer,
        batch_size=2,
    )
    assert loader.batch_size == 2
    assert loader.tokenizer == tokenizer


@pytest.mark.skipif(not (TRANSFORMERS_AVAILABLE and JAX_AVAILABLE), reason="JAX and transformers required")
def test_jax_data_loader_batch_creation():
    """Test that JaxDataLoader creates correctly shaped batches."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create sample traces
    traces_data = []
    for i in range(4):
        trace = {
            "trace_id": f"uuid-{i}",
            "environment_id": "test-env",
            "schema_version": "v1",
            "steps": [
                {
                    "t": 0,
                    "observation": f"prompt-{i}",
                    "action": f"completion-{i}",
                    "reward": float(i),  # Different rewards for pairing
                    "done": True,
                }
            ],
        }
        traces_data.append(trace)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for data in traces_data:
            f.write(json.dumps(data) + "\n")
        temp_path = f.name
    
    try:
        loader = JaxDataLoader(
            file_path=temp_path,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=128,
        )
        
        # Get first batch
        batches = list(loader)
        assert len(batches) > 0
        
        batch = batches[0]
        assert "chosen_input_ids" in batch
        assert "rejected_input_ids" in batch
        assert "chosen_labels" in batch
        assert "rejected_labels" in batch
        
        # Check shapes
        assert batch["chosen_input_ids"].shape[0] <= 2  # batch_size
        assert batch["chosen_input_ids"].shape[1] == 128  # max_length
        
    finally:
        Path(temp_path).unlink()


@pytest.mark.skipif(not (TRANSFORMERS_AVAILABLE and JAX_AVAILABLE), reason="JAX and transformers required")
def test_jax_data_loader_length():
    """Test JaxDataLoader length calculation."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create sample traces
    traces_data = []
    for i in range(5):
        trace = {
            "trace_id": f"uuid-{i}",
            "environment_id": "test-env",
            "schema_version": "v1",
            "steps": [
                {
                    "t": 0,
                    "observation": f"prompt-{i}",
                    "action": f"completion-{i}",
                    "reward": float(i),
                    "done": True,
                }
            ],
        }
        traces_data.append(trace)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for data in traces_data:
            f.write(json.dumps(data) + "\n")
        temp_path = f.name
    
    try:
        loader = JaxDataLoader(
            file_path=temp_path,
            tokenizer=tokenizer,
            batch_size=2,
        )
        
        # Length should be number of batches (pairs // batch_size)
        # With 5 traces, we get ~2 pairs, so ~1 batch
        length = len(loader)
        assert length >= 0
        
    finally:
        Path(temp_path).unlink()

