"""
CRM Support Sandbox Environment.

A reference implementation of a CRM support sandbox that simulates
customer service workflows with tickets, customers, agents, and messages.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.core.algorithms import UniversalRollout
from loguru import logger


@dataclass
class Customer:
    """Customer entity."""
    customer_id: str
    name: str
    email: str
    tier: str = "standard"  # standard, premium, enterprise
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Ticket:
    """Support ticket entity."""
    ticket_id: str
    customer_id: str
    subject: str
    description: str
    priority: str = "medium"  # low, medium, high, urgent
    status: str = "open"  # open, assigned, in_progress, resolved, closed
    assigned_agent: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """Support agent entity."""
    agent_id: str
    name: str
    specialization: List[str] = field(default_factory=list)
    workload: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CRMSupportSandbox(EnvironmentAdapter):
    """
    CRM Support Sandbox Environment.
    
    Simulates a customer service CRM system with:
    - Customers (with tiers and metadata)
    - Support tickets (with priority, status, messages)
    - Agents (with specialization and workload)
    
    Actions:
    - view_ticket: View ticket details
    - reply_to_ticket: Reply to a ticket
    - escalate_ticket: Escalate ticket priority
    - assign_ticket: Assign ticket to an agent
    - resolve_ticket: Mark ticket as resolved
    - search_customers: Search for customers
    
    Example:
        ```python
        sandbox = CRMSupportSandbox()
        obs, info = sandbox.reset()
        obs, reward, done, truncated, info = sandbox.step({
            "action_type": "view_ticket",
            "ticket_id": "ticket-1"
        })
        ```
    """
    
    def __init__(
        self,
        num_customers: int = 10,
        num_agents: int = 3,
        num_tickets: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize CRM Support Sandbox.
        
        Args:
            num_customers: Number of synthetic customers to create
            num_agents: Number of support agents
            num_tickets: Number of initial tickets
            seed: Random seed for reproducibility
        """
        self.num_customers = num_customers
        self.num_agents = num_agents
        self.num_tickets = num_tickets
        self.seed = seed
        
        # State
        self.customers: Dict[str, Customer] = {}
        self.tickets: Dict[str, Ticket] = {}
        self.agents: Dict[str, Agent] = {}
        
        # Episode state
        self._current_ticket_id: Optional[str] = None
        self._episode_started = False
        self._step_count = 0
        self._max_steps = 20
        
        # Trajectory tracking
        self._trajectory: Dict[str, list] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        # Initialize synthetic data
        self._initialize_data()
    
    def _initialize_data(self) -> None:
        """Initialize synthetic customers, agents, and tickets."""
        import random
        if self.seed is not None:
            random.seed(self.seed)
        
        # Create customers
        tiers = ["standard", "premium", "enterprise"]
        for i in range(self.num_customers):
            customer_id = f"customer-{i+1}"
            self.customers[customer_id] = Customer(
                customer_id=customer_id,
                name=f"Customer {i+1}",
                email=f"customer{i+1}@example.com",
                tier=random.choice(tiers),
            )
        
        # Create agents
        specializations = ["billing", "technical", "sales", "general"]
        for i in range(self.num_agents):
            agent_id = f"agent-{i+1}"
            self.agents[agent_id] = Agent(
                agent_id=agent_id,
                name=f"Agent {i+1}",
                specialization=random.sample(specializations, k=random.randint(1, 2)),
                workload=0,
            )
        
        # Create initial tickets
        priorities = ["low", "medium", "high", "urgent"]
        for i in range(self.num_tickets):
            ticket_id = f"ticket-{i+1}"
            customer_id = random.choice(list(self.customers.keys()))
            ticket = Ticket(
                ticket_id=ticket_id,
                customer_id=customer_id,
                subject=f"Support Request {i+1}",
                description=f"Customer needs help with issue {i+1}",
                priority=random.choice(priorities),
                status="open",
                created_at=datetime.utcnow().isoformat(),
            )
            self.tickets[ticket_id] = ticket
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the CRM sandbox to initial state.
        
        Args:
            seed: Optional random seed
            options: Optional reset options (e.g., ticket_id to focus on)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        # Reset episode state
        self._step_count = 0
        self._trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        # Select ticket to work on
        ticket_id = options.get("ticket_id") if options else None
        if not ticket_id:
            # Select random open ticket
            open_tickets = [t for t in self.tickets.values() if t.status == "open"]
            if open_tickets:
                ticket_id = open_tickets[0].ticket_id
            else:
                # Create a new ticket
                ticket_id = f"ticket-new-{uuid.uuid4().hex[:8]}"
                customer_id = list(self.customers.keys())[0]
                self.tickets[ticket_id] = Ticket(
                    ticket_id=ticket_id,
                    customer_id=customer_id,
                    subject="New Support Request",
                    description="Customer needs assistance",
                    priority="medium",
                    status="open",
                    created_at=datetime.utcnow().isoformat(),
                )
        
        self._current_ticket_id = ticket_id
        self._episode_started = True
        
        # Initial observation
        ticket = self.tickets[ticket_id]
        customer = self.customers[ticket.customer_id]
        
        observation = {
            "ticket_id": ticket_id,
            "ticket": {
                "subject": ticket.subject,
                "description": ticket.description,
                "priority": ticket.priority,
                "status": ticket.status,
                "customer_id": ticket.customer_id,
            },
            "customer": {
                "name": customer.name,
                "email": customer.email,
                "tier": customer.tier,
            },
            "available_actions": [
                "view_ticket",
                "reply_to_ticket",
                "escalate_ticket",
                "assign_ticket",
                "resolve_ticket",
                "search_customers",
            ],
        }
        
        self._trajectory["observations"].append(observation)
        
        info = {
            "ticket_id": ticket_id,
            "customer_id": ticket.customer_id,
            "environment": "crm_support_sandbox",
        }
        
        return observation, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the CRM sandbox.
        
        Args:
            action: Action dictionary with "action_type" and parameters
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict)
        """
        if not self._episode_started:
            raise RuntimeError("Environment must be reset before stepping")
        
        if self._current_ticket_id is None:
            raise RuntimeError("No ticket selected")
        
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
        
        ticket = self.tickets[self._current_ticket_id]
        
        if action_type == "view_ticket":
            reward = 0.1
            observation = {
                "ticket": {
                    "ticket_id": ticket.ticket_id,
                    "subject": ticket.subject,
                    "description": ticket.description,
                    "priority": ticket.priority,
                    "status": ticket.status,
                    "messages": ticket.messages,
                },
                "status": "viewed",
            }
        
        elif action_type == "reply_to_ticket":
            message_text = action.get("message", "")
            if len(message_text) > 10:
                ticket.messages.append({
                    "message_id": str(uuid.uuid4()),
                    "text": message_text,
                    "timestamp": datetime.utcnow().isoformat(),
                    "sender": "agent",
                })
                ticket.updated_at = datetime.utcnow().isoformat()
                reward = 0.5 if len(message_text) > 50 else 0.3
                observation = {
                    "status": "replied",
                    "message": message_text,
                }
            else:
                reward = -0.1
                observation = {"status": "message_too_short"}
        
        elif action_type == "escalate_ticket":
            new_priority = action.get("priority", "high")
            if new_priority in ["high", "urgent"]:
                ticket.priority = new_priority
                ticket.updated_at = datetime.utcnow().isoformat()
                reward = 0.4
                observation = {"status": "escalated", "new_priority": new_priority}
            else:
                reward = -0.1
                observation = {"status": "invalid_priority"}
        
        elif action_type == "assign_ticket":
            agent_id = action.get("agent_id")
            if agent_id and agent_id in self.agents:
                ticket.assigned_agent = agent_id
                ticket.status = "assigned"
                ticket.updated_at = datetime.utcnow().isoformat()
                self.agents[agent_id].workload += 1
                reward = 0.6
                observation = {"status": "assigned", "agent_id": agent_id}
            else:
                reward = -0.2
                observation = {"status": "agent_not_found"}
        
        elif action_type == "resolve_ticket":
            resolution = action.get("resolution", "")
            if len(resolution) > 10:
                ticket.status = "resolved"
                ticket.updated_at = datetime.utcnow().isoformat()
                ticket.metadata["resolution"] = resolution
                reward = 1.0  # High reward for resolution
                observation = {"status": "resolved", "resolution": resolution}
                terminated = True
            else:
                reward = -0.1
                observation = {"status": "resolution_too_short"}
        
        elif action_type == "search_customers":
            query = action.get("query", "")
            # Simple search simulation
            matching = [
                c for c in self.customers.values()
                if query.lower() in c.name.lower() or query.lower() in c.email.lower()
            ]
            reward = 0.2
            observation = {
                "status": "search_complete",
                "results": [
                    {"customer_id": c.customer_id, "name": c.name, "email": c.email}
                    for c in matching[:5]
                ],
            }
        
        else:
            reward = -0.2
            observation = {"status": "unknown_action", "action_type": action_type}
        
        # Check truncation
        if self._step_count >= self._max_steps:
            truncated = True
            reward -= 0.5  # Penalty for timeout
        
        # Track trajectory
        self._trajectory["observations"].append(observation)
        self._trajectory["actions"].append(action)
        self._trajectory["rewards"].append(reward)
        self._trajectory["dones"].append(terminated or truncated)
        
        info = {
            "step": self._step_count,
            "action_type": action_type,
            "ticket_id": self._current_ticket_id,
            "ticket_status": ticket.status,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[Any]:
        """Render current state."""
        if self._current_ticket_id:
            ticket = self.tickets[self._current_ticket_id]
            return {
                "ticket": ticket,
                "customer": self.customers.get(ticket.customer_id),
                "step": self._step_count,
            }
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        self._current_ticket_id = None
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
        """Observation space (not formally defined)."""
        return None
    
    @property
    def action_space(self) -> Any:
        """Action space."""
        return [
            "view_ticket",
            "reply_to_ticket",
            "escalate_ticket",
            "assign_ticket",
            "resolve_ticket",
            "search_customers",
        ]
    
    def get_rollout(self) -> UniversalRollout:
        """Convert trajectory to UniversalRollout."""
        if not self._episode_started or len(self._trajectory["observations"]) == 0:
            raise RuntimeError("No trajectory data available. Reset and step the environment first.")
        
        prompt = json.dumps(self._trajectory["observations"][0])
        actions_str = [json.dumps(act) for act in self._trajectory["actions"]]
        completion = " ".join(actions_str) if len(actions_str) > 1 else (actions_str[0] if actions_str else "")
        total_reward = sum(self._trajectory["rewards"])
        
        metadata = {
            "num_steps": len(self._trajectory["actions"]),
            "ticket_id": self._current_ticket_id,
        }
        
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

