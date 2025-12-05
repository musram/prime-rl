"""
MCP (Model Context Protocol) environment adapter.

This module implements MCPAdapter, which wraps MCP servers and maps MCP
tool calls and responses to UniversalRollout format for RL training.
"""

from typing import Any, Dict, Optional, Tuple, Callable
import json
import uuid

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore

from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.core.algorithms import UniversalRollout
from loguru import logger


class MCPAdapter(EnvironmentAdapter):
    """
    Adapter for MCP (Model Context Protocol) servers as RL environments.
    
    Wraps an MCP server connection and provides a consistent interface for PRIME-RL.
    Maps MCP tool calls and responses to the UniversalRollout format.
    
    This adapter allows RL agents to learn policies for using MCP tools and resources.
    
    Example:
        ```python
        from mcp import StdioServerParameters
        
        server_params = StdioServerParameters(...)
        adapter = MCPAdapter(server_params)
        
        obs, info = adapter.reset()
        obs, reward, terminated, truncated, info = adapter.step({
            "tool": "read_file",
            "arguments": {"path": "/tmp/file.txt"}
        })
        
        rollout = adapter.get_rollout()
        ```
    """
    
    def __init__(
        self,
        server_params: Any,  # StdioServerParameters or similar
        reward_function: Optional[Callable[[Dict[str, Any], Any, Any], float]] = None,
        state_to_string: Optional[Callable[[Any], str]] = None,
        action_to_string: Optional[Callable[[Any], str]] = None,
    ):
        """
        Initialize MCP adapter.
        
        Args:
            server_params: MCP server parameters (e.g., StdioServerParameters)
            reward_function: Optional function(state, action, result) -> float
            state_to_string: Optional function to convert state to string
            action_to_string: Optional function to convert action to string
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP library required for MCPAdapter. Install with: pip install mcp"
            )
        
        self.server_params = server_params
        self.reward_function = reward_function or self._default_reward_function
        self.state_to_string = state_to_string or self._default_state_to_string
        self.action_to_string = action_to_string or self._default_action_to_string
        
        # MCP session (created on reset)
        self._session: Optional[ClientSession] = None
        
        # Track trajectory for rollout conversion
        self._current_trajectory: Dict[str, list] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "tool_results": [],
        }
        self._episode_started = False
        self._current_state: Optional[Dict[str, Any]] = None
        self._available_tools: list = []
    
    def _default_reward_function(self, state: Dict[str, Any], action: Any, result: Any) -> float:
        """Default reward function."""
        # Check if tool call was successful
        if isinstance(result, dict):
            if "error" in result:
                return -1.0
            if "content" in result or "result" in result:
                return 0.5  # Small positive reward for successful calls
        return 0.0
    
    def _default_state_to_string(self, state: Any) -> str:
        """Default state to string conversion."""
        if isinstance(state, str):
            return state
        elif isinstance(state, dict):
            # Extract tool results or content
            if "content" in state:
                return str(state["content"])
            if "result" in state:
                return str(state["result"])
            return json.dumps(state, default=str)
        else:
            return str(state)
    
    def _default_action_to_string(self, action: Any) -> str:
        """Default action to string conversion."""
        if isinstance(action, str):
            return action
        elif isinstance(action, dict):
            # Format tool call
            if "tool" in action:
                tool_name = action["tool"]
                args = action.get("arguments", {})
                return f"{tool_name}({json.dumps(args)})"
            return json.dumps(action)
        else:
            return str(action)
    
    async def _initialize_session(self):
        """Initialize MCP session."""
        if self._session is None:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize session
                    await session.initialize()
                    # List available tools
                    tools_result = await session.list_tools()
                    self._available_tools = [tool.name for tool in tools_result.tools]
                    self._session = session
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the MCP environment.
        
        Args:
            seed: Optional random seed
            options: Optional reset options
            
        Returns:
            Tuple of (observation, info_dict)
        """
        import asyncio
        
        # Reset trajectory tracking
        self._current_trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "tool_results": [],
        }
        
        # Initialize session
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule initialization
                asyncio.create_task(self._initialize_session())
            else:
                loop.run_until_complete(self._initialize_session())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(self._initialize_session())
        
        # Initial observation: available tools
        self._current_state = {
            "available_tools": self._available_tools,
            "initialized": True,
        }
        self._current_trajectory["observations"].append(self._current_state)
        self._episode_started = True
        
        obs_str = self.state_to_string(self._current_state)
        info = {
            "available_tools": self._available_tools,
            "state": self._current_state,
        }
        
        return obs_str, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the MCP environment (call a tool).
        
        Args:
            action: Tool call action (dict with "tool" and "arguments" keys)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict)
        """
        if not self._episode_started:
            raise RuntimeError("Environment must be reset before stepping")
        
        if self._session is None:
            raise RuntimeError("MCP session not initialized")
        
        # Parse action
        if isinstance(action, str):
            # Try to parse as JSON
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                action = {"tool": action, "arguments": {}}
        
        if not isinstance(action, dict) or "tool" not in action:
            raise ValueError("Action must be a dict with 'tool' key")
        
        tool_name = action["tool"]
        tool_args = action.get("arguments", {})
        
        # Call tool via MCP
        import asyncio
        
        async def _call_tool():
            try:
                result = await self._session.call_tool(tool_name, tool_args)
                return result
            except Exception as e:
                logger.error(f"Error calling MCP tool {tool_name}: {e}")
                return {"error": str(e)}
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to handle this differently
                # For now, raise error suggesting async usage
                raise RuntimeError(
                    "MCPAdapter.step() called from async context. "
                    "Use step_async() instead or run in sync context."
                )
            else:
                tool_result = loop.run_until_complete(_call_tool())
        except RuntimeError:
            tool_result = asyncio.run(_call_tool())
        
        # Update state
        self._current_state = {
            "last_tool": tool_name,
            "last_result": tool_result,
            "available_tools": self._available_tools,
        }
        
        # Compute reward
        reward = self.reward_function(self._current_state, action, tool_result)
        
        # Check termination (e.g., error or explicit done signal)
        terminated = self._check_termination(tool_result)
        truncated = False
        
        # Track trajectory
        self._current_trajectory["actions"].append(action)
        self._current_trajectory["observations"].append(self._current_state)
        self._current_trajectory["rewards"].append(reward)
        self._current_trajectory["dones"].append(terminated)
        self._current_trajectory["tool_results"].append(tool_result)
        
        obs_str = self.state_to_string(self._current_state)
        info = {
            "tool": tool_name,
            "result": tool_result,
            "reward": reward,
        }
        
        return obs_str, reward, terminated, truncated, info
    
    def _check_termination(self, result: Any) -> bool:
        """Check if result indicates episode termination."""
        if isinstance(result, dict):
            if "error" in result and result["error"]:
                return True
            if "done" in result and result["done"]:
                return True
        return False
    
    def render(self) -> Optional[Any]:
        """
        Render the current state of the environment.
        
        Returns:
            Current state dictionary
        """
        return self._current_state
    
    def close(self) -> None:
        """
        Clean up environment resources (close MCP session).
        """
        import asyncio
        
        if self._session:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._session.close())
                else:
                    loop.run_until_complete(self._session.close())
            except RuntimeError:
                asyncio.run(self._session.close())
            self._session = None
        
        self._episode_started = False
        self._current_state = None
        self._current_trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "tool_results": [],
        }
    
    @property
    def observation_space(self) -> Any:
        """
        Get the observation space specification.
        
        Returns:
            None (MCP doesn't have explicit spaces)
        """
        return None
    
    @property
    def action_space(self) -> Any:
        """
        Get the action space specification.
        
        Returns:
            List of available tool names
        """
        return self._available_tools
    
    def get_rollout(self) -> UniversalRollout:
        """
        Convert current trajectory to UniversalRollout format.
        
        Returns:
            UniversalRollout object with current trajectory data
        """
        if not self._episode_started or len(self._current_trajectory["observations"]) == 0:
            raise RuntimeError("No trajectory data available. Reset and step the environment first.")
        
        # Extract prompt (initial state with available tools)
        initial_obs = self._current_trajectory["observations"][0]
        prompt = self.state_to_string(initial_obs)
        
        # Extract completion (all tool calls concatenated)
        actions_str = [self.action_to_string(act) for act in self._current_trajectory["actions"]]
        completion = " ".join(actions_str) if len(actions_str) > 1 else (actions_str[0] if actions_str else "")
        
        # Extract reward (sum of all rewards)
        total_reward = sum(self._current_trajectory["rewards"])
        
        # Build metadata
        metadata = {
            "num_steps": len(self._current_trajectory["actions"]),
            "episode_length": len(self._current_trajectory["rewards"]),
            "tools_used": [act.get("tool") for act in self._current_trajectory["actions"] if isinstance(act, dict)],
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

