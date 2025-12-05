"""
LangGraph environment adapter.

This module implements LangGraphAdapter, which wraps LangGraph state machines
and maps LangGraph state to UniversalRollout format for RL training.
"""

from typing import Any, Dict, Optional, Tuple, Callable
import json

try:
    from langgraph.graph import StateGraph
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore
    add_messages = None  # type: ignore

from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.core.algorithms import UniversalRollout
from loguru import logger


class LangGraphAdapter(EnvironmentAdapter):
    """
    Adapter for LangGraph state machines as RL environments.
    
    Wraps a LangGraph StateGraph and provides a consistent interface for PRIME-RL.
    Maps LangGraph state (typically containing messages, agent state, etc.) to
    the UniversalRollout format.
    
    This adapter treats LangGraph nodes as environment steps, allowing RL agents
    to learn policies for navigating LangGraph workflows.
    
    Example:
        ```python
        from langgraph.graph import StateGraph
        
        graph = StateGraph(...)
        adapter = LangGraphAdapter(graph)
        
        obs, info = adapter.reset()
        obs, reward, terminated, truncated, info = adapter.step("user_message")
        
        rollout = adapter.get_rollout()
        ```
    """
    
    def __init__(
        self,
        graph: Any,  # StateGraph or compiled graph
        initial_state: Optional[Dict[str, Any]] = None,
        reward_function: Optional[Callable[[Dict[str, Any], Any], float]] = None,
        state_to_string: Optional[Callable[[Any], str]] = None,
        action_to_string: Optional[Callable[[Any], str]] = None,
    ):
        """
        Initialize LangGraph adapter.
        
        Args:
            graph: LangGraph StateGraph instance (compiled or uncompiled)
            initial_state: Optional initial state dictionary
            reward_function: Optional function(state, action) -> float for computing rewards
            state_to_string: Optional function to convert state to string.
                Default: JSON serialization
            action_to_string: Optional function to convert action to string.
                Default: uses str() or JSON serialization
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph required for LangGraphAdapter. Install with: pip install langgraph"
            )
        
        self.graph = graph
        self.initial_state = initial_state or {}
        self.reward_function = reward_function or self._default_reward_function
        self.state_to_string = state_to_string or self._default_state_to_string
        self.action_to_string = action_to_string or self._default_action_to_string
        
        # Compile graph if needed
        if not hasattr(self.graph, "invoke"):
            self.graph = self.graph.compile()
        
        # Track trajectory for rollout conversion
        self._current_trajectory: Dict[str, list] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "states": [],
        }
        self._episode_started = False
        self._current_state: Optional[Dict[str, Any]] = None
        self._initial_state: Optional[Dict[str, Any]] = None
    
    def _default_reward_function(self, state: Dict[str, Any], action: Any) -> float:
        """Default reward function (returns 0.0, should be overridden)."""
        # Check if state indicates completion or success
        if isinstance(state, dict):
            if "success" in state and state["success"]:
                return 1.0
            if "error" in state and state["error"]:
                return -1.0
        return 0.0
    
    def _default_state_to_string(self, state: Any) -> str:
        """Default state to string conversion."""
        if isinstance(state, str):
            return state
        elif isinstance(state, dict):
            # Extract messages if present (common in LangGraph)
            if "messages" in state:
                messages = state["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    # Get last message content
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict) and "content" in last_msg:
                        return str(last_msg["content"])
                    return str(messages[-1])
            # Otherwise serialize entire state
            return json.dumps(state, default=str)
        else:
            return str(state)
    
    def _default_action_to_string(self, action: Any) -> str:
        """Default action to string conversion."""
        if isinstance(action, str):
            return action
        elif isinstance(action, dict):
            return json.dumps(action)
        else:
            return str(action)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the LangGraph environment to initial state.
        
        Args:
            seed: Optional random seed (may be used in initial state)
            options: Optional dictionary of reset options (e.g., initial messages)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        # Reset trajectory tracking
        self._current_trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "states": [],
        }
        
        # Build initial state
        initial_state = self.initial_state.copy()
        if options:
            initial_state.update(options)
        
        if seed is not None:
            initial_state["seed"] = seed
        
        # Invoke graph with initial state to get first observation
        try:
            self._current_state = self.graph.invoke(initial_state)
        except Exception as e:
            logger.warning(f"Error invoking graph with initial state: {e}")
            self._current_state = initial_state
        
        self._initial_state = self._current_state.copy() if isinstance(self._current_state, dict) else self._current_state
        self._current_trajectory["observations"].append(self._current_state)
        self._current_trajectory["states"].append(self._current_state)
        self._episode_started = True
        
        obs_str = self.state_to_string(self._current_state)
        info = {
            "state": self._current_state,
            "graph_nodes": getattr(self.graph, "nodes", []),
        }
        
        return obs_str, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the LangGraph environment.
        
        Args:
            action: Action to take (typically a message or state update)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict)
        """
        if not self._episode_started:
            raise RuntimeError("Environment must be reset before stepping")
        
        if self._current_state is None:
            raise RuntimeError("Invalid state: current_state is None")
        
        # Convert action to LangGraph input format
        # Typically, actions are messages or state updates
        if isinstance(action, str):
            # Assume action is a message content
            graph_input = {"messages": [{"role": "user", "content": action}]}
        elif isinstance(action, dict):
            graph_input = action
        else:
            graph_input = {"input": action}
        
        # Merge with current state
        if isinstance(self._current_state, dict):
            graph_input = {**self._current_state, **graph_input}
        else:
            graph_input = self._current_state
        
        # Step graph
        try:
            next_state = self.graph.invoke(graph_input)
        except Exception as e:
            logger.error(f"Error invoking graph: {e}")
            next_state = self._current_state
            terminated = True
            truncated = False
            reward = -1.0  # Penalty for error
        else:
            # Compute reward
            reward = self.reward_function(next_state, action)
            
            # Check termination (e.g., graph reached end node or error state)
            terminated = self._check_termination(next_state)
            truncated = False  # LangGraph doesn't have truncation by default
        
        # Track trajectory
        self._current_trajectory["actions"].append(action)
        self._current_trajectory["observations"].append(next_state)
        self._current_trajectory["states"].append(next_state)
        self._current_trajectory["rewards"].append(reward)
        self._current_trajectory["dones"].append(terminated)
        
        self._current_state = next_state
        
        obs_str = self.state_to_string(next_state)
        info = {
            "state": next_state,
            "reward": reward,
        }
        
        return obs_str, reward, terminated, truncated, info
    
    def _check_termination(self, state: Dict[str, Any]) -> bool:
        """Check if state indicates episode termination."""
        if isinstance(state, dict):
            # Check common termination signals
            if "done" in state and state["done"]:
                return True
            if "error" in state and state["error"]:
                return True
            if "success" in state and state["success"]:
                return True
            # Check if graph reached end node
            if "next" in state and state["next"] == "END":
                return True
        return False
    
    def render(self) -> Optional[Any]:
        """
        Render the current state of the environment.
        
        Returns:
            Current state dictionary or None
        """
        return self._current_state
    
    def close(self) -> None:
        """
        Clean up environment resources.
        """
        self._episode_started = False
        self._current_state = None
        self._initial_state = None
        self._current_trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "states": [],
        }
    
    @property
    def observation_space(self) -> Any:
        """
        Get the observation space specification.
        
        Returns:
            Dictionary space (LangGraph state is typically dict-like)
        """
        # LangGraph doesn't have explicit spaces, return None or dict
        return None
    
    @property
    def action_space(self) -> Any:
        """
        Get the action space specification.
        
        Returns:
            Dictionary space (actions are typically dict-like)
        """
        # LangGraph doesn't have explicit spaces, return None or dict
        return None
    
    def get_rollout(self) -> UniversalRollout:
        """
        Convert current trajectory to UniversalRollout format.
        
        Returns:
            UniversalRollout object with current trajectory data
        """
        if not self._episode_started or len(self._current_trajectory["observations"]) == 0:
            raise RuntimeError("No trajectory data available. Reset and step the environment first.")
        
        # Extract prompt (initial state)
        prompt = self.state_to_string(self._initial_state)
        
        # Extract completion (all actions concatenated)
        actions_str = [self.action_to_string(act) for act in self._current_trajectory["actions"]]
        completion = " ".join(actions_str) if len(actions_str) > 1 else (actions_str[0] if actions_str else "")
        
        # Extract reward (sum of all rewards)
        total_reward = sum(self._current_trajectory["rewards"])
        
        # Build metadata
        metadata = {
            "num_steps": len(self._current_trajectory["actions"]),
            "episode_length": len(self._current_trajectory["rewards"]),
        }
        
        # For multi-step trajectories, include full trajectory data
        observations = None
        actions = None
        dones = None
        
        if len(self._current_trajectory["observations"]) > 1:
            observations = [self.state_to_string(obs) for obs in self._current_trajectory["observations"]]
            actions = [self.action_to_string(act) for act in self._current_trajectory["actions"]]
            dones = self._current_trajectory["dones"]
        
        return UniversalRollout(
            prompts=[prompt],
            completions=[completion],
            rewards=[total_reward],
            observations=observations,
            actions=actions,
            dones=dones,
            metadata=metadata,
        )

