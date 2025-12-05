"""Tests for JaxTrainer."""

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
    from prime_rl.backends.jax import JaxTrainer, JaxDPO
    from prime_rl.backends.jax.data_loader import JaxDataLoader


@pytest.mark.skipif(not (TRANSFORMERS_AVAILABLE and JAX_AVAILABLE), reason="JAX and transformers required")
def test_jax_trainer_initialization():
    """Test JaxTrainer initialization."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create minimal data loader
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write one trace
        trace = {
            "trace_id": "test-uuid",
            "environment_id": "test-env",
            "schema_version": "v1",
            "steps": [
                {
                    "t": 0,
                    "observation": "test prompt",
                    "action": "test completion",
                    "reward": 1.0,
                    "done": True,
                }
            ],
        }
        f.write(json.dumps(trace) + "\n")
        temp_path = f.name
    
    try:
        algorithm = JaxDPO({
            "learning_rate": 1e-5,
            "beta": 0.1,
            "model_name": "gpt2",
        })
        
        data_loader = JaxDataLoader(
            file_path=temp_path,
            tokenizer=tokenizer,
            batch_size=1,
        )
        
        trainer = JaxTrainer(
            algorithm=algorithm,
            config={"output_dir": "./test_output", "max_steps": 1},
            data_loader=data_loader,
        )
        
        assert trainer.algorithm == algorithm
        assert trainer.data_loader == data_loader
        
    finally:
        Path(temp_path).unlink()


@pytest.mark.skipif(not (TRANSFORMERS_AVAILABLE and JAX_AVAILABLE), reason="JAX and transformers required")
@pytest.mark.slow
def test_jax_trainer_training_loop_smoke():
    """Smoke test for JaxTrainer training loop with toy data."""
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
                    "observation": f"Question {i}: What is 2+2?",
                    "action": f"Answer {i}: The answer is 4.",
                    "reward": float(i),  # Different rewards
                    "done": True,
                }
            ],
        }
        traces_data.append(trace)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write traces
        traces_file = Path(tmpdir) / "traces.jsonl"
        with open(traces_file, "w") as f:
            for data in traces_data:
                f.write(json.dumps(data) + "\n")
        
        # Initialize components
        algorithm = JaxDPO({
            "learning_rate": 1e-5,
            "beta": 0.1,
            "model_name": "gpt2",  # Small model for testing
        })
        
        data_loader = JaxDataLoader(
            file_path=str(traces_file),
            tokenizer=tokenizer,
            batch_size=2,
            max_length=64,  # Small for testing
        )
        
        output_dir = Path(tmpdir) / "output"
        trainer = JaxTrainer(
            algorithm=algorithm,
            config={
                "output_dir": str(output_dir),
                "max_steps": 2,  # Just 2 steps for smoke test
                "checkpoint_every": 1,
            },
            data_loader=data_loader,
        )
        
        # Run training loop (should complete without error)
        trainer.run_training_loop()
        
        # Check that output directory was created
        assert output_dir.exists()
        
        # Check that logs were created
        log_file = output_dir / "logs" / "training_metrics.jsonl"
        if log_file.exists():
            # Verify log file has content
            with open(log_file) as f:
                lines = f.readlines()
                assert len(lines) > 0

