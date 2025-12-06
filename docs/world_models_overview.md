# World Models Overview

This document describes the generic world-model framework in PRIME-RL, which enables learning and using environment models for simulation, data augmentation, and model-based RL.

## Overview

World models learn to predict environment dynamics from `InteractionTrace` data, allowing PRIME-RL to:

- **Roll out agents cheaply** in simulation ("model-imagined world")
- **Mix real and simulated experience** (Dyna-style model-based RL)
- **Quantify sim-to-real gaps** using existing verifiers
- **Augment offline datasets** with synthetic trajectories

## Architecture

### Core Abstractions

The framework is built around three core abstractions:

1. **WorldModel** (`src/prime_rl/world_models/base.py`): Abstract interface for world models
   - `predict_next()`: Predict next state, reward, done from (state, action)
   - `train_step()`: Perform one training step
   - `save()` / `load()`: Checkpoint management
   - `sample_initial_state()`: Sample initial states

2. **StateEncoder** / **ActionEncoder** (`src/prime_rl/world_models/data.py`): Convert states/actions to tensors
   - `CRMStateEncoder`: Encodes structured CRM states
   - `TextObsEncoder`: Encodes text observations
   - `DictActionEncoder`: Encodes dictionary actions

3. **WorldModelEnvAdapter** (`src/prime_rl/world_models/env_adapter.py`): Wraps any WorldModel as EnvironmentAdapter
   - Implements standard `reset()` and `step()` interface
   - Can be used anywhere a real environment is used

### Registry System

World models are registered by algorithm name:

```python
from prime_rl.world_models.base import register_world_model, create_world_model

# Register a new world model
register_world_model("my_algorithm", MyWorldModel)

# Create from config
config = WorldModelConfig(
    algorithm="my_algorithm",
    state_representation="my_rep",
)
model = create_world_model(config)
```

## Available World Models

### CRM MLP World Model (`crm_mlp`)

A simple MLP-based world model for CRM sandbox:

- **State Representation**: `crm_structured` (structured CRM state)
- **Architecture**: Multi-layer perceptron
- **Predicts**: Next state, reward, done flag
- **Use Case**: CRM sandbox simulation

**Configuration:**
```toml
[world_model]
algorithm = "crm_mlp"
state_representation = "crm_structured"
algorithm_config = {
    hidden_dim = 256,
    learning_rate = 0.001
}
```

### Text Transformer World Model (`text_transformer`)

A transformer-based world model for text observations:

- **State Representation**: `text` (text observations)
- **Architecture**: Transformer encoder
- **Predicts**: Next text observation, reward, done flag
- **Use Case**: Text-based environments

**Configuration:**
```toml
[world_model]
algorithm = "text_transformer"
state_representation = "text"
algorithm_config = {
    max_length = 512,
    d_model = 128,
    nhead = 4,
    num_layers = 2,
    learning_rate = 0.0001
}
```

## Training World Models

### From Configuration File

```bash
python -m prime_rl.world_models.train --config configs/world_model_crm_mlp.toml
```

### From Python API

```python
from prime_rl.world_models.train import train_world_model
from prime_rl.world_models.base import WorldModelConfig

config = WorldModelConfig(
    algorithm="crm_mlp",
    state_representation="crm_structured",
    algorithm_config={"hidden_dim": 256, "learning_rate": 0.001},
)

results = train_world_model(
    config=config,
    data_path=Path("data/crm_traces.jsonl"),
    output_dir=Path("outputs/world_models/crm_mlp"),
    num_epochs=20,
    batch_size=32,
)
```

## Using World Models as Environments

### In Training Config

```toml
# configs/train_with_world_model.toml
[backend]
type = "torch"
mode = "online"

[algorithm]
name = "grpo"

[model]
name = "meta-llama/Llama-3.1-8B-Instruct"

# Use world model instead of real environment
[world_model]
algorithm = "crm_mlp"
state_representation = "crm_structured"
checkpoint_path = "outputs/world_models/crm_mlp/checkpoint_best"
```

### From Python API

```python
from prime_rl.world_models.env_adapter import WorldModelEnvAdapter
from prime_rl.world_models.base import WorldModelConfig, create_world_model

config = WorldModelConfig(
    algorithm="crm_mlp",
    state_representation="crm_structured",
    checkpoint_path=Path("checkpoints/crm_world_model"),
)

env = WorldModelEnvAdapter(config=config)

obs, info = env.reset()
obs, reward, done, truncated, info = env.step({"action_type": "reply_to_ticket"})
```

## Use Cases

### 1. Pure Offline Data Augmentation

Generate synthetic trajectories to augment offline datasets:

```python
world_model = create_world_model(config)
env = WorldModelEnvAdapter(world_model=world_model, config=config)

# Generate synthetic trajectories
traces = []
for _ in range(1000):
    obs, info = env.reset()
    episode_steps = []
    done = False
    while not done:
        action = sample_action()  # From policy or random
        obs, reward, done, truncated, info = env.step(action)
        episode_steps.append((obs, action, reward, done))
    traces.append(episode_steps)

# Combine with real traces and train offline RL
```

### 2. Dyna-Style Imagination Steps

Mix real and simulated experience during online training:

```python
# For each real environment step:
real_obs, real_reward, real_done, _, _ = real_env.step(action)

# Run k imaginary steps in world model
for _ in range(k):
    sim_obs, sim_reward, sim_done, _, _ = world_model_env.step(action)
    # Add to replay buffer with lower weight
    replay_buffer.add(sim_obs, action, sim_reward, sim_done, weight=0.5)
```

### 3. Sim-to-Real Gap Analysis

Compare policy performance in real vs. simulated environments:

```python
# Train policy in world model
policy = train_in_world_model(world_model_env)

# Evaluate in real environment
real_metrics = evaluate_in_real_env(real_env, policy)

# Evaluate in world model
sim_metrics = evaluate_in_world_model(world_model_env, policy)

# Compare using verifiers
gap = compute_sim_to_real_gap(real_metrics, sim_metrics)
```

## Creating Custom World Models

### Step 1: Implement WorldModel Interface

```python
from prime_rl.world_models.base import WorldModel, WorldModelConfig, register_world_model

class MyWorldModel(WorldModel):
    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        # Initialize your model architecture
    
    def predict_next(self, state_batch, action_batch):
        # Predict next state, reward, done
        return next_states, rewards, dones, info
    
    def train_step(self, batch):
        # Training step
        return {"loss": loss_value}
    
    def save(self, path):
        # Save model checkpoint
        pass
    
    def load(self, path):
        # Load model checkpoint
        pass
    
    def sample_initial_state(self, num_samples=1):
        # Sample initial states
        return initial_states

# Register
register_world_model("my_algorithm", MyWorldModel)
```

### Step 2: Create State/Action Encoders (if needed)

```python
from prime_rl.world_models.data import StateEncoder

class MyStateEncoder(StateEncoder):
    def encode(self, state):
        # Convert state to tensor
        return encoded_state
    
    def decode(self, encoded):
        # Convert tensor back to state
        return state
    
    @property
    def state_dim(self):
        return dimension
```

### Step 3: Add Configuration Support

Update `src/prime_rl/core/config.py` if needed, or use existing `algorithm_config` dict.

## Best Practices

1. **Start Simple**: Begin with simple models (MLP) before complex architectures
2. **Validate on Held-Out Data**: Always evaluate world model accuracy on held-out traces
3. **Use Verifiers**: Leverage existing verifiers to evaluate world model realism
4. **Monitor Sim-to-Real Gap**: Track how well simulated performance predicts real performance
5. **Iterative Refinement**: Start with basic predictions, then add complexity as needed

## Limitations & Future Work

### Current Limitations

- Simplified encoders (character-level for text, basic features for structured)
- No support for multi-modal states
- Limited to single-step prediction (no long-horizon rollouts yet)
- No uncertainty quantification

### Future Enhancements

- **Dreamer-style latent models**: Learn latent state representations
- **Long-horizon prediction**: Multi-step rollouts with uncertainty
- **Hybrid modeling**: Mix learned models with real tool calls
- **Better encoders**: Proper tokenizers, learned embeddings
- **Uncertainty estimation**: Quantify prediction confidence

## Examples

See:
- `configs/world_model_crm_mlp.toml`: Training CRM MLP world model
- `configs/world_model_text_transformer.toml`: Training text transformer world model
- `configs/train_with_world_model.toml`: Using world model in RL training

## References

- Original design doc: `docs/world-model-sim.md`
- Gap analysis: `docs/gap_analysis_rl_take1.md` (World Models section)

