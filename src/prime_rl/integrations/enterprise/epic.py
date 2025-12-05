"""
Epic (Electronic Health Record) adapter for healthcare workflows.

This module implements EpicAdapter, a mock/stub adapter for HL7/FHIR
based healthcare workflows, enabling RL training on healthcare tasks.
"""

from typing import Any, Dict, Optional, Tuple
import json
import uuid

from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.core.algorithms import UniversalRollout
from loguru import logger


class EpicAdapter(EnvironmentAdapter):
    """
    Adapter for Epic Electronic Health Record (EHR) systems.
    
    This is a mock/stub implementation for HL7/FHIR based healthcare workflows.
    In production, would connect to actual Epic APIs or simulators.
    
    The adapter simulates common EHR operations like:
    - Patient lookup
    - Chart review
    - Order entry
    - Documentation
    - Clinical decision support
    
    Example:
        ```python
        adapter = EpicAdapter(
            environment_id="epic-ehr-v1",
            mock_mode=True,
        )
        
        obs, info = adapter.reset()
        obs, reward, terminated, truncated, info = adapter.step({
            "action_type": "lookup_patient",
            "patient_id": "12345"
        })
        ```
    """
    
    def __init__(
        self,
        environment_id: str = "epic-ehr-v1",
        mock_mode: bool = True,
        mock_patients: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Epic adapter.
        
        Args:
            environment_id: Environment identifier
            mock_mode: If True, use mock data (default: True)
            mock_patients: Optional dictionary of mock patient data
        """
        self.environment_id = environment_id
        self.mock_mode = mock_mode
        self.mock_patients = mock_patients or self._create_mock_patients()
        
        # Current state
        self._current_patient: Optional[Dict[str, Any]] = None
        self._current_chart: Optional[Dict[str, Any]] = None
        self._episode_started = False
        self._step_count = 0
        
        # Track trajectory
        self._trajectory: Dict[str, list] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
    
    def _create_mock_patients(self) -> Dict[str, Any]:
        """Create mock patient data for testing."""
        return {
            "12345": {
                "patient_id": "12345",
                "name": "John Doe",
                "age": 45,
                "conditions": ["Hypertension", "Type 2 Diabetes"],
                "medications": ["Lisinopril", "Metformin"],
                "vitals": {"bp": "140/90", "hr": 72, "temp": 98.6},
            },
            "67890": {
                "patient_id": "67890",
                "name": "Jane Smith",
                "age": 32,
                "conditions": ["Asthma"],
                "medications": ["Albuterol"],
                "vitals": {"bp": "120/80", "hr": 68, "temp": 98.4},
            },
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the Epic EHR environment.
        
        Args:
            seed: Optional random seed
            options: Optional reset options (e.g., patient_id, task_type)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        # Reset state
        self._current_patient = None
        self._current_chart = None
        self._step_count = 0
        self._trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        # Get initial patient/task from options
        patient_id = options.get("patient_id") if options else None
        if not patient_id:
            # Select random patient
            import random
            if seed is not None:
                random.seed(seed)
            patient_id = random.choice(list(self.mock_patients.keys()))
        
        if patient_id in self.mock_patients:
            self._current_patient = self.mock_patients[patient_id]
        else:
            # Create new mock patient
            self._current_patient = {
                "patient_id": patient_id,
                "name": "Unknown",
                "age": 0,
                "conditions": [],
                "medications": [],
                "vitals": {},
            }
        
        self._current_chart = {
            "patient_id": patient_id,
            "encounters": [],
            "orders": [],
            "notes": [],
        }
        
        self._episode_started = True
        
        # Initial observation: patient summary
        observation = {
            "patient_id": patient_id,
            "summary": f"Patient {self._current_patient['name']}, Age {self._current_patient['age']}",
            "conditions": self._current_patient.get("conditions", []),
            "available_actions": ["lookup_patient", "review_chart", "enter_order", "document_note"],
        }
        
        self._trajectory["observations"].append(observation)
        
        info = {
            "patient_id": patient_id,
            "environment_id": self.environment_id,
        }
        
        return observation, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the Epic EHR environment.
        
        Args:
            action: Action dictionary with "action_type" and parameters
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict)
        """
        if not self._episode_started:
            raise RuntimeError("Environment must be reset before stepping")
        
        if self._current_patient is None:
            raise RuntimeError("No patient loaded")
        
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
        
        if action_type == "lookup_patient":
            patient_id = action.get("patient_id")
            if patient_id and patient_id in self.mock_patients:
                self._current_patient = self.mock_patients[patient_id]
                reward = 0.5
                observation = {
                    "patient_id": patient_id,
                    "summary": f"Patient {self._current_patient['name']}",
                    "status": "found",
                }
            else:
                reward = -0.1
                observation = {"status": "patient_not_found"}
        
        elif action_type == "review_chart":
            reward = 0.3
            observation = {
                "chart": self._current_chart,
                "patient": self._current_patient,
            }
        
        elif action_type == "enter_order":
            order_type = action.get("order_type", "unknown")
            order_details = action.get("details", {})
            
            order = {
                "order_id": str(uuid.uuid4()),
                "order_type": order_type,
                "details": order_details,
                "timestamp": str(self._step_count),
            }
            self._current_chart["orders"].append(order)
            
            # Reward based on order appropriateness
            if order_type in ["lab", "medication", "imaging"]:
                reward = 0.7
            else:
                reward = 0.2
            
            observation = {
                "order_entered": order,
                "status": "success",
            }
        
        elif action_type == "document_note":
            note_text = action.get("note", "")
            note = {
                "note_id": str(uuid.uuid4()),
                "text": note_text,
                "timestamp": str(self._step_count),
            }
            self._current_chart["notes"].append(note)
            
            reward = 0.5 if len(note_text) > 10 else 0.1
            observation = {
                "note_documented": note,
                "status": "success",
            }
        
        else:
            reward = -0.2
            observation = {"status": "unknown_action", "action_type": action_type}
        
        # Check termination (mock: terminate after 10 steps or on "complete" action)
        if action_type == "complete" or self._step_count >= 10:
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
            "patient_id": self._current_patient.get("patient_id"),
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[Any]:
        """
        Render the current state of the environment.
        
        Returns:
            Current patient/chart state
        """
        return {
            "patient": self._current_patient,
            "chart": self._current_chart,
            "step": self._step_count,
        }
    
    def close(self) -> None:
        """
        Clean up environment resources.
        """
        self._current_patient = None
        self._current_chart = None
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
            None (Epic doesn't have explicit spaces)
        """
        return None
    
    @property
    def action_space(self) -> Any:
        """
        Get the action space specification.
        
        Returns:
            List of available action types
        """
        return ["lookup_patient", "review_chart", "enter_order", "document_note", "complete"]
    
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
            "patient_id": self._current_patient.get("patient_id") if self._current_patient else None,
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

