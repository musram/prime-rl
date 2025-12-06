# Enterprise Sandbox Recipe

This document explains how to create and use enterprise sandboxes in PRIME-RL, and how to mirror your own stack.

## Overview

Enterprise sandboxes are application-level environments that simulate real-world workflows (CRM, Finance, HR, etc.) for secure offline training and RLVR-friendly evaluation.

## Architecture

Sandboxes in PRIME-RL follow this architecture:

1. **Environment Adapter**: Implements `EnvironmentAdapter` interface
   - Manages state (entities, relationships)
   - Provides `reset()` and `step()` methods
   - Converts trajectories to `UniversalRollout` format

2. **Verifier/Rubric**: Implements `VerifierClient` interface
   - Scores episodes based on rubric criteria
   - Provides reward signals for training

3. **Configuration**: TOML config files that wire everything together

## Built-in Sandboxes

### CRM Support Sandbox

Simulates customer service workflows with:
- Customers (with tiers and metadata)
- Support tickets (with priority, status, messages)
- Agents (with specialization and workload)

**Usage:**
```bash
prime-rl train --config configs/crm_support_sandbox.toml --mode online
```

**Actions:**
- `view_ticket`: View ticket details
- `reply_to_ticket`: Reply to a ticket
- `escalate_ticket`: Escalate ticket priority
- `assign_ticket`: Assign ticket to an agent
- `resolve_ticket`: Mark ticket as resolved
- `search_customers`: Search for customers

### Finance Reconciliation Sandbox

Simulates financial reconciliation workflows with:
- Transactions (from various sources)
- Ledger entries (internal accounting records)
- Reconciliation tasks (matching transactions to ledger entries)

**Usage:**
```bash
prime-rl train --config configs/finance_reconciliation_sandbox.toml --mode online
```

**Actions:**
- `view_transaction`: View transaction details
- `view_ledger_entry`: View ledger entry details
- `match_transaction`: Match transaction to ledger entry
- `reconcile_account`: Reconcile an account for a period
- `flag_exception`: Flag a reconciliation exception

## Creating Your Own Sandbox

### Step 1: Define Your Domain Model

Create dataclasses for your entities:

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class YourEntity:
    entity_id: str
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Step 2: Implement EnvironmentAdapter

Create a class that inherits from `EnvironmentAdapter`:

```python
from prime_rl.core.environment import EnvironmentAdapter

class YourSandbox(EnvironmentAdapter):
    def __init__(self, ...):
        # Initialize your state
        pass
    
    def reset(self, seed=None, options=None):
        # Reset to initial state
        return observation, info
    
    def step(self, action):
        # Execute action and return (obs, reward, done, truncated, info)
        return observation, reward, terminated, truncated, info
    
    def get_rollout(self):
        # Convert trajectory to UniversalRollout
        return UniversalRollout(...)
```

### Step 3: Create Rubric Verifier

Implement a `VerifierClient` that scores episodes:

```python
from prime_rl.core.verifier import VerifierClient

class YourRubricVerifier(VerifierClient):
    def verify(self, observation, action, trace_id=None, metadata=None):
        # Score based on rubric criteria
        reward = compute_score(...)
        return VerificationResult(reward=reward, success=...)
```

### Step 4: Register Your Sandbox

Register your adapter and verifier in `src/prime_rl/core/registry.py`:

```python
from prime_rl.sandboxes.your_sandbox import YourSandbox, YourRubricVerifier

register_environment_adapter("your_sandbox", YourSandbox)
register_verifier_client("your_rubric", YourRubricVerifier)
```

### Step 5: Create Configuration

Create a TOML config file:

```toml
[backend]
type = "torch"
mode = "online"

[algorithm]
name = "grpo"

[model]
name = "meta-llama/Llama-3.1-8B-Instruct"

[environment]
type = "your_sandbox"
# Your sandbox-specific config

[verifier]
type = "your_rubric"

[output]
output_dir = "outputs/your_sandbox"
```

### Step 6: Add to Scenario Registry

Register your scenario in `src/prime_rl/registry/registry.py`:

```python
self.register(ScenarioMetadata(
    id="your_sandbox",
    name="Your Sandbox",
    description="Description of your sandbox",
    config_path=Path("configs/your_sandbox.toml"),
    category="your_category",
    tags=["your", "tags"],
    env_type="your_sandbox",
    verifier_type="your_rubric",
))
```

## Mirroring Your Stack

To mirror your own enterprise stack:

1. **Extract Schemas**: Document your data models (customers, tickets, etc.)

2. **Create Synthetic Data**: Generate realistic but synthetic seed data:
   ```python
   def create_synthetic_customers(num=100):
       return [Customer(...) for _ in range(num)]
   ```

3. **Map Workflows**: Identify key workflows and actions:
   - What actions can agents take?
   - What are the success criteria?
   - What are the constraints?

4. **Implement State Transitions**: Model how state changes with actions:
   ```python
   def step(self, action):
       if action["type"] == "your_action":
           # Update state
           self.state.update(...)
   ```

5. **Define Rubrics**: Create scoring rubrics:
   - Correctness: Did the action achieve the goal?
   - Policy adherence: Did it follow company policies?
   - Completeness: Was the response complete?

6. **Test and Iterate**: Run training and adjust based on results.

## Best Practices

1. **Start Small**: Begin with a minimal MVP (few entities, simple workflows)

2. **Use Realistic Data**: Synthetic data should mirror real distributions

3. **Clear Success Criteria**: Define what "good" means for your domain

4. **Iterative Refinement**: Add complexity gradually

5. **Documentation**: Document your sandbox's purpose, entities, and workflows

## Examples

See `src/prime_rl/sandboxes/crm/` and `src/prime_rl/sandboxes/finance/` for reference implementations.

