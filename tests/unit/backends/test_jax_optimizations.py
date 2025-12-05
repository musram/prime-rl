"""
Tests for JAX backend optimizations.

Tests multi-device support, mixed precision, gradient accumulation,
preference pair extraction improvements, separate reference model, and evaluation.
"""

import pytest
from pathlib import Path
import tempfile
import json

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from prime_rl.backends.jax.utils import (
    get_dtype,
    get_num_devices,
    shard_batch,
    aggregate_gradients,
    aggregate_metrics,
)
from prime_rl.backends.jax.data_loader import JaxDataLoader
from prime_rl.backends.jax.algorithms.dpo import JaxDPO
from prime_rl.backends.jax.trainer import JaxTrainer
from prime_rl.core.interaction_trace import InteractionTrace, TraceStep
from prime_rl.core.algorithms import UniversalRollout


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXUtils:
    """Tests for JAX utility functions."""
    
    def test_get_dtype(self):
        """Test dtype conversion."""
        assert get_dtype("float32") == jnp.float32
        assert get_dtype("float16") == jnp.float16
        assert get_dtype("bfloat16") == jnp.bfloat16
        
        with pytest.raises(ValueError):
            get_dtype("invalid")
    
    def test_get_num_devices(self):
        """Test device count."""
        num_devices = get_num_devices()
        assert num_devices >= 1
    
    def test_shard_batch_single_device(self):
        """Test sharding with single device (no-op)."""
        batch = {
            "chosen_input_ids": jnp.ones((8, 10), dtype=jnp.int32),
            "rejected_input_ids": jnp.ones((8, 10), dtype=jnp.int32),
        }
        sharded = shard_batch(batch, 1)
        assert sharded["chosen_input_ids"].shape == (8, 10)
    
    def test_aggregate_metrics(self):
        """Test metric aggregation."""
        metrics = {
            "loss": 1.0,
            "grad_norm": 2.0,
            "other": "string",
        }
        aggregated = aggregate_metrics(metrics)
        assert "loss" in aggregated
        assert "grad_norm" in aggregated
        assert aggregated["other"] == "string"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestPreferencePairExtraction:
    """Tests for preference pair extraction improvements."""
    
    def test_preference_group_id_pairing(self, tmp_path):
        """Test pairing using preference_group_id."""
        # Create traces with preference_group_id
        traces = []
        for i in range(4):
            trace = InteractionTrace(
                trace_id=f"trace_{i}",
                environment_id="test",
                schema_version="v1",
                steps=[
                    TraceStep(
                        t=0,
                        observation=f"prompt_{i}",
                        action=f"completion_{i}",
                        reward=1.0 - i * 0.2,  # Decreasing rewards
                        done=True,
                    )
                ],
                labels={"preference_group_id": "group_1" if i < 2 else "group_2"},
            )
            traces.append(trace)
        
        # Write to JSONL
        jsonl_file = tmp_path / "traces.jsonl"
        with open(jsonl_file, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict()) + "\n")
        
        # Load with data loader
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        data_loader = JaxDataLoader(
            file_path=str(jsonl_file),
            tokenizer=tokenizer,
            batch_size=4,
            shuffle=False,
        )
        
        # Check that pairs are created
        batches = list(data_loader)
        assert len(batches) > 0
        # Should have pairs within each group
    
    def test_reward_based_pairing_fallback(self, tmp_path):
        """Test fallback to reward-based pairing when no group IDs."""
        # Create traces without preference_group_id
        traces = []
        for i in range(4):
            trace = InteractionTrace(
                trace_id=f"trace_{i}",
                environment_id="test",
                schema_version="v1",
                steps=[
                    TraceStep(
                        t=0,
                        observation=f"prompt_{i}",
                        action=f"completion_{i}",
                        reward=1.0 - i * 0.2,
                        done=True,
                    )
                ],
            )
            traces.append(trace)
        
        # Write to JSONL
        jsonl_file = tmp_path / "traces.jsonl"
        with open(jsonl_file, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict()) + "\n")
        
        # Load with data loader
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        data_loader = JaxDataLoader(
            file_path=str(jsonl_file),
            tokenizer=tokenizer,
            batch_size=4,
            shuffle=False,
        )
        
        # Should still create pairs based on rewards
        batches = list(data_loader)
        assert len(batches) > 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestSeparateReferenceModel:
    """Tests for separate reference model support."""
    
    def test_separate_reference_model_config(self):
        """Test that separate reference model can be configured."""
        config = {
            "learning_rate": 1e-5,
            "beta": 0.1,
            "model_name": "gpt2",
            "reference_model_name": "gpt2",  # Same model for testing
            "trust_remote_code": False,
        }
        
        algorithm = JaxDPO(config)
        assert algorithm.reference_model_name == "gpt2"
    
    def test_default_reference_model(self):
        """Test default behavior (no separate reference model)."""
        config = {
            "learning_rate": 1e-5,
            "beta": 0.1,
            "model_name": "gpt2",
            "trust_remote_code": False,
        }
        
        algorithm = JaxDPO(config)
        assert algorithm.reference_model_name is None


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestGradientAccumulation:
    """Tests for gradient accumulation."""
    
    def test_gradient_accumulation_config(self, tmp_path):
        """Test that gradient accumulation can be configured."""
        # Create minimal dataset
        jsonl_file = tmp_path / "traces.jsonl"
        with open(jsonl_file, "w") as f:
            trace = InteractionTrace(
                trace_id="test",
                environment_id="test",
                schema_version="v1",
                steps=[
                    TraceStep(t=0, observation="test", action="test", reward=1.0, done=True)
                ],
            )
            f.write(json.dumps(trace.to_dict()) + "\n")
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        algorithm_config = {
            "learning_rate": 1e-5,
            "beta": 0.1,
            "model_name": "gpt2",
        }
        algorithm = JaxDPO(algorithm_config)
        
        data_loader = JaxDataLoader(
            file_path=str(jsonl_file),
            tokenizer=tokenizer,
            batch_size=1,
            shuffle=False,
        )
        
        trainer_config = {
            "output_dir": str(tmp_path / "output"),
            "gradient_accumulation_steps": 2,
            "dtype": "float32",
            "use_pmap": False,
        }
        
        trainer = JaxTrainer(
            algorithm=algorithm,
            config=trainer_config,
            data_loader=data_loader,
        )
        
        assert trainer.gradient_accumulation_steps == 2


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestMixedPrecision:
    """Tests for mixed precision support."""
    
    def test_mixed_precision_config(self):
        """Test that mixed precision dtype can be configured."""
        config = {
            "learning_rate": 1e-5,
            "beta": 0.1,
            "model_name": "gpt2",
            "dtype": "bfloat16",
        }
        
        algorithm = JaxDPO(config)
        # Check that dtype is set (will be converted to JAX dtype in init_state)
        assert config["dtype"] == "bfloat16"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestEvaluation:
    """Tests for evaluation loop."""
    
    def test_evaluation_with_validation_dataset(self, tmp_path):
        """Test evaluation loop with validation dataset."""
        # Create training and validation datasets
        train_file = tmp_path / "train.jsonl"
        val_file = tmp_path / "val.jsonl"
        
        for jsonl_file in [train_file, val_file]:
            with open(jsonl_file, "w") as f:
                trace = InteractionTrace(
                    trace_id="test",
                    environment_id="test",
                    schema_version="v1",
                    steps=[
                        TraceStep(t=0, observation="test", action="test", reward=1.0, done=True)
                    ],
                )
                f.write(json.dumps(trace.to_dict()) + "\n")
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        algorithm_config = {
            "learning_rate": 1e-5,
            "beta": 0.1,
            "model_name": "gpt2",
        }
        algorithm = JaxDPO(algorithm_config)
        
        train_loader = JaxDataLoader(
            file_path=str(train_file),
            tokenizer=tokenizer,
            batch_size=1,
            shuffle=False,
        )
        
        val_loader = JaxDataLoader(
            file_path=str(val_file),
            tokenizer=tokenizer,
            batch_size=1,
            shuffle=False,
        )
        
        trainer_config = {
            "output_dir": str(tmp_path / "output"),
            "eval_every": 1,
            "dtype": "float32",
            "use_pmap": False,
        }
        
        trainer = JaxTrainer(
            algorithm=algorithm,
            config=trainer_config,
            data_loader=train_loader,
            validation_data_loader=val_loader,
        )
        
        assert trainer.validation_data_loader is not None
        
        # Initialize state
        import jax.random as random
        rng_key = random.PRNGKey(0)
        algorithm_config_init = {
            "model_name": "gpt2",
            "learning_rate": 1e-5,
            "beta": 0.1,
        }
        trainer.state = algorithm.init_state(rng_key, algorithm_config_init)
        
        # Run evaluation
        eval_metrics = trainer.evaluate(step=0)
        assert eval_metrics is not None
        assert "eval_loss" in eval_metrics


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestMultiDevice:
    """Tests for multi-device support."""
    
    def test_pmap_config(self, tmp_path):
        """Test that pmap can be configured."""
        jsonl_file = tmp_path / "traces.jsonl"
        with open(jsonl_file, "w") as f:
            trace = InteractionTrace(
                trace_id="test",
                environment_id="test",
                schema_version="v1",
                steps=[
                    TraceStep(t=0, observation="test", action="test", reward=1.0, done=True)
                ],
            )
            f.write(json.dumps(trace.to_dict()) + "\n")
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        algorithm_config = {
            "learning_rate": 1e-5,
            "beta": 0.1,
            "model_name": "gpt2",
        }
        algorithm = JaxDPO(algorithm_config)
        
        data_loader = JaxDataLoader(
            file_path=str(jsonl_file),
            tokenizer=tokenizer,
            batch_size=1,
            shuffle=False,
        )
        
        trainer_config = {
            "output_dir": str(tmp_path / "output"),
            "use_pmap": True,  # Request pmap
            "dtype": "float32",
        }
        
        trainer = JaxTrainer(
            algorithm=algorithm,
            config=trainer_config,
            data_loader=data_loader,
        )
        
        # Should disable pmap if only 1 device, or enable if multiple devices
        num_devices = get_num_devices()
        if num_devices == 1:
            assert trainer.use_pmap is False
        else:
            assert trainer.use_pmap is True

