"""
Data adapters and encoders for world model training.

This module provides utilities to convert InteractionTrace data into training
batches and pluggable encoders for different state representations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import json

from prime_rl.core.interaction_trace import InteractionTrace, TraceStep
from loguru import logger


class StateEncoder(ABC):
    """
    Abstract base class for state encoders.
    
    Encoders convert environment states (structured or text) into tensor representations
    suitable for world model training.
    """
    
    @abstractmethod
    def encode(self, state: Any) -> Any:
        """
        Encode a state to tensor representation.
        
        Args:
            state: State in original format (dict, string, etc.)
            
        Returns:
            Encoded state (tensor, array, etc.)
        """
        pass
    
    @abstractmethod
    def decode(self, encoded: Any) -> Any:
        """
        Decode an encoded state back to original format.
        
        Args:
            encoded: Encoded state (tensor, array, etc.)
            
        Returns:
            Decoded state in original format
        """
        pass
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """
        Get the dimension of encoded states.
        
        Returns:
            State dimension (for vector states) or shape tuple
        """
        pass


class ActionEncoder(ABC):
    """
    Abstract base class for action encoders.
    
    Encoders convert agent actions into tensor representations.
    """
    
    @abstractmethod
    def encode(self, action: Any) -> Any:
        """
        Encode an action to tensor representation.
        
        Args:
            action: Action in original format (dict, string, etc.)
            
        Returns:
            Encoded action (tensor, array, etc.)
        """
        pass
    
    @abstractmethod
    def decode(self, encoded: Any) -> Any:
        """
        Decode an encoded action back to original format.
        
        Args:
            encoded: Encoded action (tensor, array, etc.)
            
        Returns:
            Decoded action in original format
        """
        pass
    
    @property
    @abstractmethod
    def action_dim(self) -> int:
        """
        Get the dimension of encoded actions.
        
        Returns:
            Action dimension (for vector actions) or shape tuple
        """
        pass


class CRMStateEncoder(StateEncoder):
    """
    Encoder for CRM sandbox structured states.
    
    Encodes CRM state (tickets, customers, agents) into a fixed-size vector.
    """
    
    def __init__(self):
        """Initialize CRM state encoder."""
        # Define feature dimensions
        # This is a simplified encoding; in production would be more sophisticated
        self._state_dim = 128  # Placeholder
    
    def encode(self, state: Any) -> Any:
        """
        Encode CRM state to vector.
        
        Args:
            state: State dict with keys like "ticket", "customer", etc.
            
        Returns:
            Encoded state vector
        """
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except json.JSONDecodeError:
                state = {}
        
        if not isinstance(state, dict):
            return [0.0] * self._state_dim
        
        # Simplified encoding: extract key features
        # In production, would use proper embeddings/one-hot encodings
        features = []
        
        # Ticket features
        if "ticket" in state:
            ticket = state["ticket"]
            # Priority encoding (one-hot-like)
            priority_map = {"low": 0, "medium": 1, "high": 2, "urgent": 3}
            features.append(float(priority_map.get(ticket.get("priority", "medium"), 1)))
            # Status encoding
            status_map = {"open": 0, "assigned": 1, "in_progress": 2, "resolved": 3, "closed": 4}
            features.append(float(status_map.get(ticket.get("status", "open"), 0)))
        
        # Customer tier
        if "customer" in state:
            customer = state["customer"]
            tier_map = {"standard": 0, "premium": 1, "enterprise": 2}
            features.append(float(tier_map.get(customer.get("tier", "standard"), 0)))
        
        # Pad to fixed dimension
        while len(features) < self._state_dim:
            features.append(0.0)
        
        return features[:self._state_dim]
    
    def decode(self, encoded: Any) -> Any:
        """
        Decode encoded state back to CRM state dict.
        
        Args:
            encoded: Encoded state vector
            
        Returns:
            State dictionary (simplified reconstruction)
        """
        # Simplified decoding - in production would be more sophisticated
        return {
            "ticket": {
                "priority": "medium",
                "status": "open",
            },
            "customer": {
                "tier": "standard",
            },
        }
    
    @property
    def state_dim(self) -> int:
        """Get state dimension."""
        return self._state_dim


class TextObsEncoder(StateEncoder):
    """
    Encoder for text-based observations.
    
    Encodes text observations into token IDs or embeddings.
    """
    
    def __init__(self, max_length: int = 512):
        """
        Initialize text observation encoder.
        
        Args:
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self._state_dim = max_length  # Token sequence length
    
    def encode(self, state: Any) -> Any:
        """
        Encode text observation to token IDs.
        
        Args:
            state: Text observation (string) or dict with "observation" key
            
        Returns:
            Token IDs (simplified: character indices for now)
        """
        if isinstance(state, dict):
            text = str(state.get("observation", ""))
        else:
            text = str(state)
        
        # Simplified encoding: character indices
        # In production, would use proper tokenizer (e.g., from transformers)
        tokens = [ord(c) % 256 for c in text[:self.max_length]]
        
        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(0)
        
        return tokens[:self.max_length]
    
    def decode(self, encoded: Any) -> Any:
        """
        Decode token IDs back to text.
        
        Args:
            encoded: Token IDs
            
        Returns:
            Text observation
        """
        # Simplified decoding
        text = "".join(chr(t) if t > 0 else "" for t in encoded if t > 0)
        return text
    
    @property
    def state_dim(self) -> int:
        """Get state dimension."""
        return self._state_dim


class DictActionEncoder(ActionEncoder):
    """
    Encoder for dictionary-based actions.
    
    Encodes actions like {"action_type": "reply", "message": "..."} into vectors.
    """
    
    def __init__(self):
        """Initialize dict action encoder."""
        self._action_dim = 64  # Placeholder
    
    def encode(self, action: Any) -> Any:
        """
        Encode action to vector.
        
        Args:
            action: Action dict or JSON string
            
        Returns:
            Encoded action vector
        """
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                action = {"action_type": action}
        
        if not isinstance(action, dict):
            action = {"action_type": str(action)}
        
        # Simplified encoding: extract action_type and parameters
        features = []
        
        # Action type encoding
        action_type = action.get("action_type", "unknown")
        action_type_map = {
            "view_ticket": 0,
            "reply_to_ticket": 1,
            "escalate_ticket": 2,
            "assign_ticket": 3,
            "resolve_ticket": 4,
            "search_customers": 5,
        }
        features.append(float(action_type_map.get(action_type, -1)))
        
        # Pad to fixed dimension
        while len(features) < self._action_dim:
            features.append(0.0)
        
        return features[:self._action_dim]
    
    def decode(self, encoded: Any) -> Any:
        """
        Decode encoded action back to dict.
        
        Args:
            encoded: Encoded action vector
            
        Returns:
            Action dictionary (simplified reconstruction)
        """
        action_type_idx = int(encoded[0]) if len(encoded) > 0 else 0
        action_type_map = {
            0: "view_ticket",
            1: "reply_to_ticket",
            2: "escalate_ticket",
            3: "assign_ticket",
            4: "resolve_ticket",
            5: "search_customers",
        }
        return {"action_type": action_type_map.get(action_type_idx, "unknown")}
    
    @property
    def action_dim(self) -> int:
        """Get action dimension."""
        return self._action_dim


def extract_training_sequences(
    traces: List[InteractionTrace],
    state_encoder: StateEncoder,
    action_encoder: ActionEncoder,
) -> List[Dict[str, Any]]:
    """
    Extract training sequences from InteractionTrace data.
    
    Args:
        traces: List of InteractionTrace objects
        state_encoder: State encoder instance
        action_encoder: Action encoder instance
        
    Returns:
        List of training examples, each with keys:
        - state: Encoded current state
        - action: Encoded action
        - next_state: Encoded next state
        - reward: Reward value
        - done: Done flag
        - metadata: Original metadata
    """
    sequences = []
    
    for trace in traces:
        for i in range(len(trace.steps) - 1):
            current_step = trace.steps[i]
            next_step = trace.steps[i + 1]
            
            # Extract state from observation or metadata
            state = current_step.observation
            if isinstance(state, str):
                try:
                    state = json.loads(state)
                except json.JSONDecodeError:
                    pass
            
            # Extract next state
            next_state = next_step.observation
            if isinstance(next_state, str):
                try:
                    next_state = json.loads(next_state)
                except json.JSONDecodeError:
                    pass
            
            # Encode states and actions
            encoded_state = state_encoder.encode(state)
            encoded_action = action_encoder.encode(current_step.action)
            encoded_next_state = state_encoder.encode(next_state)
            
            sequences.append({
                "state": encoded_state,
                "action": encoded_action,
                "next_state": encoded_next_state,
                "reward": current_step.reward,
                "done": current_step.done,
                "metadata": current_step.metadata,
            })
    
    logger.info(f"Extracted {len(sequences)} training sequences from {len(traces)} traces")
    return sequences

