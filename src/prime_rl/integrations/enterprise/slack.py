"""
Slack adapter for enterprise communication workflows.

This module implements SlackAdapter, which wraps the Slack API to allow
agents to read messages and post replies as actions, converting them to
UniversalRollout format for RL training.
"""

from typing import Any, Dict, Optional, Tuple, List
import json
import time

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    WebClient = None  # type: ignore
    SlackApiError = None  # type: ignore

from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.core.algorithms import UniversalRollout
from loguru import logger


class SlackAdapter(EnvironmentAdapter):
    """
    Adapter for Slack workspace as RL environment.
    
    Wraps the Slack API to enable RL agents to interact with Slack channels,
    reading messages and posting replies. Converts Slack interactions to
    UniversalRollout format for training.
    
    Example:
        ```python
        adapter = SlackAdapter(
            token="xoxb-your-token",
            channel_id="C1234567890",
        )
        
        obs, info = adapter.reset()
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "post_message",
            "text": "Hello, team!"
        })
        ```
    """
    
    def __init__(
        self,
        token: str,
        channel_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        mock_mode: bool = False,
    ):
        """
        Initialize Slack adapter.
        
        Args:
            token: Slack API token (xoxb-...)
            channel_id: Optional default channel ID
            workspace_id: Optional workspace ID
            mock_mode: If True, use mock data instead of real API calls
        """
        if not SLACK_AVAILABLE and not mock_mode:
            raise ImportError(
                "slack_sdk required for SlackAdapter. Install with: pip install slack_sdk"
            )
        
        self.token = token
        self.channel_id = channel_id
        self.workspace_id = workspace_id
        self.mock_mode = mock_mode
        
        # Slack client (created lazily)
        self._client: Optional[WebClient] = None
        if not mock_mode and SLACK_AVAILABLE:
            self._client = WebClient(token=token)
        
        # Current state
        self._current_channel: Optional[str] = None
        self._message_history: List[Dict[str, Any]] = []
        self._episode_started = False
        self._step_count = 0
        
        # Track trajectory
        self._trajectory: Dict[str, list] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the Slack environment.
        
        Args:
            seed: Optional random seed
            options: Optional reset options (e.g., channel_id)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        # Reset state
        self._message_history = []
        self._step_count = 0
        self._trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        # Get channel from options or use default
        channel = options.get("channel_id") if options else None
        channel = channel or self.channel_id
        
        if not channel:
            raise ValueError("channel_id must be provided in options or during initialization")
        
        self._current_channel = channel
        
        # Fetch recent messages
        if self.mock_mode:
            # Mock messages
            messages = [
                {
                    "text": "Hello, team!",
                    "user": "U12345",
                    "ts": str(time.time()),
                },
                {
                    "text": "Can someone help with the project?",
                    "user": "U67890",
                    "ts": str(time.time() - 60),
                },
            ]
        else:
            try:
                if self._client:
                    response = self._client.conversations_history(
                        channel=channel,
                        limit=10,
                    )
                    messages = response.get("messages", [])
                else:
                    messages = []
            except Exception as e:
                logger.warning(f"Failed to fetch messages: {e}")
                messages = []
        
        self._message_history = messages
        self._episode_started = True
        
        # Initial observation: channel context
        observation = {
            "channel_id": channel,
            "recent_messages": messages[-5:] if len(messages) > 5 else messages,
            "message_count": len(messages),
            "available_actions": ["read_messages", "post_message", "react", "complete"],
        }
        
        self._trajectory["observations"].append(observation)
        
        info = {
            "channel_id": channel,
            "workspace_id": self.workspace_id,
        }
        
        return observation, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the Slack environment.
        
        Args:
            action: Action dictionary with "action_type" and parameters
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict)
        """
        if not self._episode_started:
            raise RuntimeError("Environment must be reset before stepping")
        
        if not self._current_channel:
            raise RuntimeError("No channel selected")
        
        # Parse action
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                action = {"action_type": action}
        
        if not isinstance(action, dict):
            action = {"action_type": str(action)}
        
        action_type = action.get("action_type", "unknown")
        self._step_count += 1
        
        # Execute action
        reward = 0.0
        observation = {}
        terminated = False
        truncated = False
        
        if action_type == "read_messages":
            # Read messages (already in history)
            reward = 0.2
            observation = {
                "messages": self._message_history[-10:],
                "status": "read",
            }
        
        elif action_type == "post_message":
            text = action.get("text", "")
            
            if self.mock_mode:
                # Mock message posting
                new_message = {
                    "text": text,
                    "user": "bot",
                    "ts": str(time.time()),
                }
                self._message_history.append(new_message)
                reward = 0.8 if len(text) > 10 else 0.3
            else:
                try:
                    if self._client:
                        response = self._client.chat_postMessage(
                            channel=self._current_channel,
                            text=text,
                        )
                        new_message = {
                            "text": text,
                            "ts": response.get("ts"),
                            "channel": response.get("channel"),
                        }
                        self._message_history.append(new_message)
                        reward = 0.8 if len(text) > 10 else 0.3
                    else:
                        reward = -0.1
                        observation = {"status": "client_not_available"}
                except SlackApiError as e:
                    logger.error(f"Failed to post message: {e}")
                    reward = -0.5
                    observation = {"status": "error", "error": str(e)}
            
            if reward > 0:
                observation = {
                    "message_posted": new_message,
                    "status": "success",
                }
        
        elif action_type == "react":
            message_ts = action.get("message_ts")
            emoji = action.get("emoji", "thumbsup")
            
            if self.mock_mode:
                reward = 0.5
                observation = {"status": "reacted", "emoji": emoji}
            else:
                try:
                    if self._client and message_ts:
                        self._client.reactions_add(
                            channel=self._current_channel,
                            timestamp=message_ts,
                            name=emoji,
                        )
                        reward = 0.5
                        observation = {"status": "reacted", "emoji": emoji}
                    else:
                        reward = -0.1
                        observation = {"status": "invalid_action"}
                except SlackApiError as e:
                    logger.error(f"Failed to add reaction: {e}")
                    reward = -0.2
                    observation = {"status": "error", "error": str(e)}
        
        else:
            reward = -0.2
            observation = {"status": "unknown_action", "action_type": action_type}
        
        # Check termination
        if action_type == "complete" or self._step_count >= 20:
            terminated = True
            reward += 1.0  # Completion bonus
        
        # Track trajectory
        self._trajectory["observations"].append(observation)
        self._trajectory["actions"].append(action)
        self._trajectory["rewards"].append(reward)
        self._trajectory["dones"].append(terminated)
        
        info = {
            "step": self._step_count,
            "action_type": action_type,
            "channel_id": self._current_channel,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[Any]:
        """
        Render the current state of the environment.
        
        Returns:
            Current channel state and message history
        """
        return {
            "channel_id": self._current_channel,
            "messages": self._message_history[-10:],
            "step": self._step_count,
        }
    
    def close(self) -> None:
        """
        Clean up environment resources.
        """
        self._current_channel = None
        self._message_history = []
        self._episode_started = False
        self._step_count = 0
        self._trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
    
    @property
    def observation_space(self) -> Any:
        """
        Get the observation space specification.
        
        Returns:
            None (Slack doesn't have explicit spaces)
        """
        return None
    
    @property
    def action_space(self) -> Any:
        """
        Get the action space specification.
        
        Returns:
            List of available action types
        """
        return ["read_messages", "post_message", "react", "complete"]
    
    def get_rollout(self) -> UniversalRollout:
        """
        Convert current trajectory to UniversalRollout format.
        
        Returns:
            UniversalRollout object
        """
        if not self._episode_started or len(self._trajectory["observations"]) == 0:
            raise RuntimeError("No trajectory data available. Reset and step the environment first.")
        
        # Extract prompt (first observation)
        prompt = json.dumps(self._trajectory["observations"][0])
        
        # Extract completion (all actions)
        actions_str = [json.dumps(act) for act in self._trajectory["actions"]]
        completion = " ".join(actions_str) if len(actions_str) > 1 else (actions_str[0] if actions_str else "")
        
        # Extract reward (sum)
        total_reward = sum(self._trajectory["rewards"])
        
        # Build metadata
        metadata = {
            "num_steps": len(self._trajectory["actions"]),
            "channel_id": self._current_channel,
        }
        
        # Multi-step trajectory data
        observations = None
        actions = None
        dones = None
        
        if len(self._trajectory["observations"]) > 1:
            observations = [json.dumps(obs) for obs in self._trajectory["observations"]]
            actions = [json.dumps(act) for act in self._trajectory["actions"]]
            dones = self._trajectory["dones"]
        
        return UniversalRollout(
            prompts=[prompt],
            completions=[completion],
            rewards=[total_reward],
            observations=observations,
            actions=actions,
            dones=dones,
            metadata=metadata,
        )

