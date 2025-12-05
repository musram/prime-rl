# Usage Example: Offline RL Training with PRIME-RL

This document provides a brief usage example for Phase 1 (Offline Engine) of PRIME-RL.

## Prerequisites

1. Install PRIME-RL (with JAX support):
```bash
pip install jax jaxlib
# Or install PRIME-RL with JAX dependencies
```

2. Prepare an Interaction Trace dataset in JSONL format (see schema in PRD §3.2)

## Example Configuration

Create a configuration file `config.toml`:

```toml
[backend]
type = "jax"
mode = "offline"

[dataset]
path = "data/traces.jsonl"
schema_version = "v1"
batch_size = 32
shuffle = true

[algorithm]
name = "dpo"
learning_rate = 1e-5
beta = 0.1

[model]
name = "meta-llama/Llama-3.1-8B"
trust_remote_code = false

[output]
output_dir = "./output"
max_steps = 1000
checkpoint_every = 100
eval_every = 200
```

## Example Interaction Trace Data

Create `data/traces.jsonl`:

```json
{"trace_id": "uuid-1", "environment_id": "browser-gym-v1", "schema_version": "v1", "metadata": {"source": "halluminate"}, "steps": [{"t": 0, "observation": "Navigate to google.com", "action": "click(10, 20)", "reward": 0.0, "done": false}, {"t": 1, "observation": "Google homepage", "action": "type('hello')", "reward": 1.0, "done": true}], "final_outcome": "success", "labels": {"human_score": 0.9}}
```

## Running Training

```bash
prime-rl train --mode offline --config config.toml
```

## Python API

You can also use PRIME-RL programmatically:

```python
from pathlib import Path
from prime_rl.train import train

# Run training
train(config_path=Path("config.toml"), mode="offline")
```

## Loading Interaction Traces

```python
from prime_rl.core.interaction_trace import load_traces_from_jsonl
from prime_rl.core.interaction_trace import InteractionTrace

# Load traces
traces = load_traces_from_jsonl("data/traces.jsonl")

# Convert to UniversalRollout
for trace in traces:
    rollout = trace.to_universal_rollout()
    print(f"Prompt: {rollout.prompts[0]}")
    print(f"Completion: {rollout.completions[0]}")
    print(f"Reward: {rollout.rewards[0]}")
```

## Using JAX Backend Directly

```python
from prime_rl.backends.jax import JaxTrainer, JaxDPO
from prime_rl.backends.jax.data_loader import JaxDataLoader

# Initialize algorithm
algorithm = JaxDPO(config={"learning_rate": 1e-5, "beta": 0.1})

# Create data loader
data_loader = JaxDataLoader(
    file_path="data/traces.jsonl",
    batch_size=32,
    shuffle=True,
)

# Initialize trainer
trainer = JaxTrainer(
    algorithm=algorithm,
    config={"max_steps": 1000},
)

# Run training
trainer.run_training_loop()
```

## Next Steps

- Phase 1 provides stub implementations. Full DPO algorithm and training loop will be implemented in subsequent phases.
- See `docs/arch_core.md` for architecture details.
- See `docs/product_spec.md` for the full PRD.

