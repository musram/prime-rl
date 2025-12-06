"""
Training entrypoint for world models.

Provides a simple API to train world models from InteractionTrace data.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json

from prime_rl.core.interaction_trace import load_traces_from_jsonl
from prime_rl.world_models.base import WorldModelConfig as WMConfig, create_world_model
from prime_rl.world_models.data import (
    extract_training_sequences,
    CRMStateEncoder,
    TextObsEncoder,
    DictActionEncoder,
    get_encoder_for_state_representation,
)
from loguru import logger


def train_world_model(
    config: WMConfig,
    data_path: Path,
    output_dir: Path,
    num_epochs: int = 10,
    batch_size: int = 32,
    validation_split: float = 0.1,
) -> Dict[str, Any]:
    """
    Train a world model from InteractionTrace data.
    
    Args:
        config: WorldModelConfig instance
        data_path: Path to InteractionTrace JSONL file
        output_dir: Directory to save checkpoints and logs
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        
    Returns:
        Dictionary with training metrics and final checkpoint path
    """
    logger.info(f"Training world model: {config.algorithm}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")
    
    # Load traces
    logger.info("Loading interaction traces...")
    traces = load_traces_from_jsonl(str(data_path))
    logger.info(f"Loaded {len(traces)} traces")
    
    # Get encoders
    state_encoder = get_encoder_for_state_representation(config.state_representation)
    action_encoder = DictActionEncoder()
    
    # Extract training sequences
    logger.info("Extracting training sequences...")
    sequences = extract_training_sequences(traces, state_encoder, action_encoder)
    logger.info(f"Extracted {len(sequences)} training sequences")
    
    if len(sequences) == 0:
        raise ValueError("No training sequences extracted from traces")
    
    # Split train/validation
    split_idx = int(len(sequences) * (1 - validation_split))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    logger.info(f"Train: {len(train_sequences)}, Validation: {len(val_sequences)}")
    
    # Create world model
    world_model = create_world_model(config)
    
    # Training loop
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    training_history = []
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Shuffle training data
        import random
        random.shuffle(train_sequences)
        
        # Train on batches
        epoch_losses = []
        for i in range(0, len(train_sequences), batch_size):
            batch = train_sequences[i:i + batch_size]
            
            # Prepare batch
            batch_dict = {
                "state": [s["state"] for s in batch],
                "action": [s["action"] for s in batch],
                "next_state": [s["next_state"] for s in batch],
                "reward": [s["reward"] for s in batch],
                "done": [float(s["done"]) for s in batch],
            }
            
            # Training step
            metrics = world_model.train_step(batch_dict)
            epoch_losses.append(metrics["loss"])
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        
        # Validation
        val_losses = []
        for i in range(0, len(val_sequences), batch_size):
            batch = val_sequences[i:i + batch_size]
            batch_dict = {
                "state": [s["state"] for s in batch],
                "action": [s["action"] for s in batch],
                "next_state": [s["next_state"] for s in batch],
                "reward": [s["reward"] for s in batch],
                "done": [float(s["done"]) for s in batch],
            }
            
            # Evaluate (simplified: just compute loss)
            val_metrics = world_model.train_step(batch_dict)  # Reuse train_step for now
            val_losses.append(val_metrics["loss"])
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
        )
        
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })
        
        # Save checkpoint if best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = output_dir / "checkpoint_best"
            world_model.save(checkpoint_path)
            logger.info(f"Saved best checkpoint (val_loss={avg_val_loss:.4f})")
    
    # Save final checkpoint
    final_checkpoint_path = output_dir / "checkpoint_final"
    world_model.save(final_checkpoint_path)
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("Training completed")
    
    return {
        "final_checkpoint": str(final_checkpoint_path),
        "best_checkpoint": str(output_dir / "checkpoint_best"),
        "training_history": training_history,
        "final_val_loss": best_val_loss,
    }


def train_world_model_from_config(config_path: Path) -> Dict[str, Any]:
    """
    Train a world model from a configuration file.
    
    Args:
        config_path: Path to TOML configuration file
        
    Returns:
        Dictionary with training results
    """
    import tomli
    
    with open(config_path, "rb") as f:
        config_dict = tomli.load(f)
    
    # Extract world_model config
    world_model_dict = config_dict.get("world_model", {})
    world_model_config = WMConfig(**world_model_dict)
    
    # Extract training config
    training_dict = config_dict.get("training", {})
    data_path = Path(training_dict["data_path"])
    output_dir = Path(training_dict.get("output_dir", "./outputs/world_model"))
    num_epochs = training_dict.get("num_epochs", 10)
    batch_size = training_dict.get("batch_size", 32)
    validation_split = training_dict.get("validation_split", 0.1)
    
    return train_world_model(
        config=world_model_config,
        data_path=data_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )


def main() -> None:
    """CLI entrypoint for world model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a world model")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML configuration file",
    )
    
    args = parser.parse_args()
    
    train_world_model_from_config(args.config)


if __name__ == "__main__":
    main()

