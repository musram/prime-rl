"""
CRM MLP World Model.

A simple MLP-based world model for CRM sandbox that predicts next state
and reward from current state and action.
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
from prime_rl.world_models.data import CRMStateEncoder, DictActionEncoder
from loguru import logger


class CrmMlpWorldModel(WorldModel):
    """
    MLP-based world model for CRM sandbox.
    
    Uses a simple multi-layer perceptron to predict:
    - Next state (structured CRM state)
    - Reward
    - Done flag
    """
    
    def __init__(self, config: WorldModelConfig):
        """
        Initialize CRM MLP world model.
        
        Args:
            config: WorldModelConfig instance
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for CrmMlpWorldModel. Install with: pip install torch")
        
        super().__init__(config)
        
        # Initialize encoders
        self.state_encoder = CRMStateEncoder()
        self.action_encoder = DictActionEncoder()
        
        # Model architecture
        state_dim = self.state_encoder.state_dim
        action_dim = self.action_encoder.action_dim
        hidden_dim = config.algorithm_config.get("hidden_dim", 256)
        
        # Input: state + action
        input_dim = state_dim + action_dim
        
        # Build MLP
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 2),  # state_dim + reward + done
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.algorithm_config.get("learning_rate", 1e-3),
        )
        
        # Training state
        self._trained = False
    
    def predict_next(
        self,
        state_batch: Any,
        action_batch: Any,
    ) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        """
        Predict next state, reward, and done flag.
        
        Args:
            state_batch: Batch of current states (list of state dicts or encoded states)
            action_batch: Batch of actions (list of action dicts or encoded actions)
            
        Returns:
            Tuple of (next_state_batch, reward_batch, done_batch, info_dict)
        """
        self.model.eval()
        
        # Encode inputs if needed
        if isinstance(state_batch[0], dict) or isinstance(state_batch[0], str):
            encoded_states = [self.state_encoder.encode(s) for s in state_batch]
        else:
            encoded_states = state_batch
        
        if isinstance(action_batch[0], dict) or isinstance(action_batch[0], str):
            encoded_actions = [self.action_encoder.encode(a) for a in action_batch]
        else:
            encoded_actions = action_batch
        
        # Convert to tensors
        state_tensor = torch.tensor(encoded_states, dtype=torch.float32)
        action_tensor = torch.tensor(encoded_actions, dtype=torch.float32)
        
        # Concatenate state and action
        input_tensor = torch.cat([state_tensor, action_tensor], dim=1)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Split output: next_state, reward, done
        next_state_encoded = output[:, :-2]
        reward_pred = output[:, -2]
        done_pred = torch.sigmoid(output[:, -1])  # Sigmoid for done probability
        
        # Decode next states
        next_states = [self.state_encoder.decode(ns.tolist()) for ns in next_state_encoded]
        
        # Convert to lists
        rewards = reward_pred.tolist()
        dones = (done_pred > 0.5).tolist()
        
        info = {
            "confidence": (1.0 - done_pred).mean().item(),
        }
        
        return next_states, rewards, dones, info
    
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
        self.model.train()
        
        # Encode states and actions
        states = torch.tensor(batch["state"], dtype=torch.float32)
        actions = torch.tensor(batch["action"], dtype=torch.float32)
        next_states = torch.tensor(batch["next_state"], dtype=torch.float32)
        rewards = torch.tensor(batch["reward"], dtype=torch.float32)
        dones = torch.tensor(batch["done"], dtype=torch.float32)
        
        # Concatenate state and action
        inputs = torch.cat([states, actions], dim=1)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Split outputs
        pred_next_states = outputs[:, :-2]
        pred_rewards = outputs[:, -2]
        pred_dones = torch.sigmoid(outputs[:, -1])
        
        # Compute losses
        state_loss = nn.functional.mse_loss(pred_next_states, next_states)
        reward_loss = nn.functional.mse_loss(pred_rewards, rewards)
        done_loss = nn.functional.binary_cross_entropy(pred_dones, dones)
        
        total_loss = state_loss + reward_loss + done_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self._trained = True
        
        return {
            "loss": total_loss.item(),
            "state_loss": state_loss.item(),
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
        
        # Save model state
        model_path = path / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
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
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self._trained = True
            logger.info(f"Loaded world model from {path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    def sample_initial_state(self, num_samples: int = 1) -> Any:
        """
        Sample initial states from empirical distribution.
        
        For now, returns simplified initial states. In production, would
        learn a distribution from training data.
        
        Args:
            num_samples: Number of states to sample
            
        Returns:
            List of initial state dictionaries
        """
        # Simplified: return default initial states
        return [
            {
                "ticket": {
                    "priority": "medium",
                    "status": "open",
                },
                "customer": {
                    "tier": "standard",
                },
            }
            for _ in range(num_samples)
        ]


# Register the world model
register_world_model("crm_mlp", CrmMlpWorldModel)

