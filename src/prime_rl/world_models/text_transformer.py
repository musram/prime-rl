"""
Text Transformer World Model.

A transformer-based world model that predicts next text observation
and reward from text observations and actions.
"""

from typing import Any, Dict, List, Tuple
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

from prime_rl.world_models.base import WorldModel, WorldModelConfig, register_world_model
from prime_rl.world_models.data import TextObsEncoder, DictActionEncoder
from loguru import logger


class TextTransformerWorldModel(WorldModel):
    """
    Transformer-based world model for text observations.
    
    Uses a simple transformer to predict:
    - Next text observation
    - Reward
    - Done flag
    """
    
    def __init__(self, config: WorldModelConfig):
        """
        Initialize text transformer world model.
        
        Args:
            config: WorldModelConfig instance
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TextTransformerWorldModel. Install with: pip install torch")
        
        super().__init__(config)
        
        # Initialize encoders
        self.state_encoder = TextObsEncoder(
            max_length=config.algorithm_config.get("max_length", 512)
        )
        self.action_encoder = DictActionEncoder()
        
        # Model architecture (simplified transformer)
        vocab_size = 256  # Character vocabulary
        d_model = config.algorithm_config.get("d_model", 128)
        nhead = config.algorithm_config.get("nhead", 4)
        num_layers = config.algorithm_config.get("num_layers", 2)
        max_length = self.state_encoder.max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.next_obs_head = nn.Linear(d_model, vocab_size)  # Next observation tokens
        self.reward_head = nn.Linear(d_model, 1)  # Reward
        self.done_head = nn.Linear(d_model, 1)  # Done probability
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.embedding.parameters()) +
            list(self.transformer.parameters()) +
            list(self.next_obs_head.parameters()) +
            list(self.reward_head.parameters()) +
            list(self.done_head.parameters()),
            lr=config.algorithm_config.get("learning_rate", 1e-4),
        )
        
        # Training state
        self._trained = False
    
    def predict_next(
        self,
        state_batch: Any,
        action_batch: Any,
    ) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        """
        Predict next observation, reward, and done flag.
        
        Args:
            state_batch: Batch of current text observations
            action_batch: Batch of actions
            
        Returns:
            Tuple of (next_obs_batch, reward_batch, done_batch, info_dict)
        """
        self.transformer.eval()
        
        # Encode inputs
        if isinstance(state_batch[0], str) or isinstance(state_batch[0], dict):
            encoded_states = [self.state_encoder.encode(s) for s in state_batch]
        else:
            encoded_states = state_batch
        
        if isinstance(action_batch[0], dict) or isinstance(action_batch[0], str):
            encoded_actions = [self.action_encoder.encode(a) for a in action_batch]
        else:
            encoded_actions = action_batch
        
        # Convert to tensors
        state_tensor = torch.tensor(encoded_states, dtype=torch.long)
        action_tensor = torch.tensor(encoded_actions, dtype=torch.float32)
        
        # Embed states
        embedded = self.embedding(state_tensor)
        
        # Add action information (simplified: concatenate to first token)
        action_expanded = action_tensor.unsqueeze(1).expand(-1, embedded.size(1), -1)
        # Truncate action to match embedding dim
        action_expanded = action_expanded[:, :, :embedded.size(2)]
        if action_expanded.size(2) < embedded.size(2):
            padding = torch.zeros(
                embedded.size(0), embedded.size(1),
                embedded.size(2) - action_expanded.size(2),
                device=embedded.device
            )
            action_expanded = torch.cat([action_expanded, padding], dim=2)
        embedded = embedded + action_expanded[:, :, :embedded.size(2)]
        
        # Transformer forward
        with torch.no_grad():
            encoded = self.transformer(embedded)
        
        # Get last token representation
        last_token = encoded[:, -1, :]
        
        # Predict outputs
        next_obs_logits = self.next_obs_head(encoded)
        next_obs_tokens = torch.argmax(next_obs_logits, dim=-1)
        rewards = self.reward_head(last_token).squeeze(-1)
        done_logits = self.done_head(last_token)
        dones = (torch.sigmoid(done_logits) > 0.5).squeeze(-1)
        
        # Decode next observations
        next_obs = [self.state_encoder.decode(tokens.tolist()) for tokens in next_obs_tokens]
        
        # Convert to lists
        rewards_list = rewards.tolist()
        dones_list = dones.tolist()
        
        info = {
            "confidence": (1.0 - torch.sigmoid(done_logits)).mean().item(),
        }
        
        return next_obs, rewards_list, dones_list, info
    
    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform one training step.
        
        Args:
            batch: Training batch with keys: state, action, next_state, reward, done
            
        Returns:
            Dictionary of training metrics
        """
        self.transformer.train()
        
        # Encode inputs
        states = torch.tensor(batch["state"], dtype=torch.long)
        actions = torch.tensor(batch["action"], dtype=torch.float32)
        next_states = torch.tensor(batch["next_state"], dtype=torch.long)
        rewards = torch.tensor(batch["reward"], dtype=torch.float32)
        dones = torch.tensor(batch["done"], dtype=torch.float32)
        
        # Embed states
        embedded = self.embedding(states)
        
        # Add action information
        action_expanded = actions.unsqueeze(1).expand(-1, embedded.size(1), -1)
        action_expanded = action_expanded[:, :, :embedded.size(2)]
        if action_expanded.size(2) < embedded.size(2):
            padding = torch.zeros(
                embedded.size(0), embedded.size(1),
                embedded.size(2) - action_expanded.size(2),
                device=embedded.device
            )
            action_expanded = torch.cat([action_expanded, padding], dim=2)
        embedded = embedded + action_expanded[:, :, :embedded.size(2)]
        
        # Transformer forward
        encoded = self.transformer(embedded)
        last_token = encoded[:, -1, :]
        
        # Predict outputs
        next_obs_logits = self.next_obs_head(encoded)
        pred_rewards = self.reward_head(last_token).squeeze(-1)
        pred_dones = torch.sigmoid(self.done_head(last_token)).squeeze(-1)
        
        # Compute losses
        obs_loss = nn.functional.cross_entropy(
            next_obs_logits.reshape(-1, next_obs_logits.size(-1)),
            next_states.reshape(-1),
        )
        reward_loss = nn.functional.mse_loss(pred_rewards, rewards)
        done_loss = nn.functional.binary_cross_entropy(pred_dones, dones)
        
        total_loss = obs_loss + reward_loss + done_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self._trained = True
        
        return {
            "loss": total_loss.item(),
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "done_loss": done_loss.item(),
        }
    
    def save(self, path: Path) -> None:
        """
        Save world model to disk.
        
        Args:
            path: Directory path to save model
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        torch.save({
            "embedding": self.embedding.state_dict(),
            "transformer": self.transformer.state_dict(),
            "next_obs_head": self.next_obs_head.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "done_head": self.done_head.state_dict(),
        }, path / "model.pt")
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.dict(), f, indent=2, default=str)
        
        logger.info(f"Saved world model to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load world model from disk.
        
        Args:
            path: Directory path to load model from
        """
        model_path = path / "model.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path)
            self.embedding.load_state_dict(checkpoint["embedding"])
            self.transformer.load_state_dict(checkpoint["transformer"])
            self.next_obs_head.load_state_dict(checkpoint["next_obs_head"])
            self.reward_head.load_state_dict(checkpoint["reward_head"])
            self.done_head.load_state_dict(checkpoint["done_head"])
            self.transformer.eval()
            self._trained = True
            logger.info(f"Loaded world model from {path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    def sample_initial_state(self, num_samples: int = 1) -> Any:
        """
        Sample initial text observations.
        
        Args:
            num_samples: Number of observations to sample
            
        Returns:
            List of initial text observations
        """
        # Simplified: return default text observations
        return ["Initial observation"] * num_samples


# Register the world model
register_world_model("text_transformer", TextTransformerWorldModel)

