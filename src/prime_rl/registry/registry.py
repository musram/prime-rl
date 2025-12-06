"""
Scenario registry implementation.

Provides a lightweight registry for scenarios with metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class ScenarioMetadata:
    """
    Metadata for a scenario.
    
    Attributes:
        id: Unique scenario identifier
        name: Human-readable name
        description: Scenario description
        config_path: Path to configuration file
        category: Scenario category (e.g., crm_sandbox, replication, browser)
        tags: List of tags for filtering
        env_type: Environment adapter type
        verifier_type: Verifier client type
        metadata: Additional metadata dictionary
    """
    id: str
    name: str
    description: str
    config_path: Optional[Path] = None
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    env_type: Optional[str] = None
    verifier_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "config_path": str(self.config_path) if self.config_path else None,
            "category": self.category,
            "tags": self.tags,
            "env_type": self.env_type,
            "verifier_type": self.verifier_type,
            "metadata": self.metadata,
        }


class ScenarioRegistry:
    """
    Central registry for PRIME-RL scenarios.
    
    Provides a catalog of available scenarios with metadata for discovery
    and configuration. Scenarios can be filtered by category, tags, etc.
    """
    
    def __init__(self):
        """Initialize scenario registry."""
        self._scenarios: Dict[str, ScenarioMetadata] = {}
        self._initialize_builtin_scenarios()
    
    def _initialize_builtin_scenarios(self) -> None:
        """Initialize built-in scenarios."""
        # CRM Support Sandbox
        self.register(ScenarioMetadata(
            id="crm_support_sandbox",
            name="CRM Support Sandbox",
            description="Customer service CRM sandbox with tickets, customers, and agents",
            config_path=Path("configs/crm_support_sandbox.toml"),
            category="crm_sandbox",
            tags=["enterprise", "customer_service", "sandbox"],
            env_type="crm_support",
            verifier_type="crm_rubric",
        ))
        
        # Finance Reconciliation Sandbox
        self.register(ScenarioMetadata(
            id="finance_reconciliation_sandbox",
            name="Finance Reconciliation Sandbox",
            description="Financial reconciliation sandbox with transactions and ledgers",
            config_path=Path("configs/finance_reconciliation_sandbox.toml"),
            category="finance_sandbox",
            tags=["enterprise", "finance", "reconciliation", "sandbox"],
            env_type="finance_reconciliation",
            verifier_type="finance_rubric",
        ))
        
        # Replication Benchmark - Project A
        self.register(ScenarioMetadata(
            id="replication_project_a",
            name="Replication Training - Project A",
            description="Replication training benchmark: implement calculate_sum function",
            config_path=Path("configs/replication_project_a.toml"),
            category="replication",
            tags=["replication", "coding", "pytest", "benchmark"],
            env_type="python_repo",
            verifier_type="pytest",
            metadata={
                "repo_path": "benchmarks/replication/project_a",
                "task": "Implement calculate_sum function",
            },
        ))
    
    def register(self, scenario: ScenarioMetadata) -> None:
        """
        Register a scenario.
        
        Args:
            scenario: ScenarioMetadata object
        """
        self._scenarios[scenario.id] = scenario
    
    def get(self, scenario_id: str) -> Optional[ScenarioMetadata]:
        """
        Get a scenario by ID.
        
        Args:
            scenario_id: Scenario identifier
            
        Returns:
            ScenarioMetadata or None if not found
        """
        return self._scenarios.get(scenario_id)
    
    def list_all(self) -> List[ScenarioMetadata]:
        """
        List all registered scenarios.
        
        Returns:
            List of ScenarioMetadata objects
        """
        return list(self._scenarios.values())
    
    def list_by_category(self, category: str) -> List[ScenarioMetadata]:
        """
        List scenarios by category.
        
        Args:
            category: Category name
            
        Returns:
            List of ScenarioMetadata objects
        """
        return [s for s in self._scenarios.values() if s.category == category]
    
    def list_by_tag(self, tag: str) -> List[ScenarioMetadata]:
        """
        List scenarios by tag.
        
        Args:
            tag: Tag name
            
        Returns:
            List of ScenarioMetadata objects
        """
        return [s for s in self._scenarios.values() if tag in s.tags]
    
    def search(self, query: str) -> List[ScenarioMetadata]:
        """
        Search scenarios by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of ScenarioMetadata objects matching the query
        """
        query_lower = query.lower()
        return [
            s for s in self._scenarios.values()
            if query_lower in s.name.lower() or query_lower in s.description.lower()
        ]


# Global registry instance
_registry: Optional[ScenarioRegistry] = None


def get_registry() -> ScenarioRegistry:
    """
    Get the global scenario registry instance.
    
    Returns:
        ScenarioRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ScenarioRegistry()
    return _registry

