"""Tests for TorchTrainer."""

import pytest

from prime_rl.backends.torch import TorchTrainer, TorchPPO


def test_torch_trainer_initialization():
    """Test TorchTrainer initialization."""
    algorithm = TorchPPO({"learning_rate": 1e-5, "clip_epsilon": 0.2})
    trainer = TorchTrainer(
        algorithm=algorithm,
        config={"max_steps": 1000},
    )
    assert trainer.algorithm == algorithm
    assert trainer.config == {"max_steps": 1000}


def test_torch_trainer_implements_interface():
    """Test that TorchTrainer implements Trainer interface."""
    from prime_rl.core.trainer import Trainer
    
    algorithm = TorchPPO({"learning_rate": 1e-5})
    trainer = TorchTrainer(algorithm=algorithm, config={})
    
    # Check that required methods exist
    assert hasattr(trainer, "run_training_loop")
    assert hasattr(trainer, "checkpoint")
    assert hasattr(trainer, "log_metrics")
    assert hasattr(trainer, "evaluate")
    
    # Check that it's an instance of Trainer
    assert isinstance(trainer, Trainer)

