# Core Architecture: PRIME-RL Phase 1

## Overview

This document summarizes the current state of PRIME-RL and the planned core abstractions for Phase 1 (Foundation: Offline Engine) as outlined in the Product Requirement Document.

## Current Components

### Existing Structure

PRIME-RL currently has a PyTorch-based online RL training system with the following key components:

1. **Orchestrator** (`src/prime_rl/orchestrator/`):
   - Manages environment stepping and rollout collection
   - Handles communication between trainer and inference services
   - Uses `verifiers` library for environment abstraction
   - Supports async environment stepping and batch preparation

2. **Trainer** (`src/prime_rl/trainer/`):
   - PyTorch-based RL trainer using FSDP2
   - Supports algorithms like GRPO, GSPO, OPO, RLOO, CISPO
   - Handles model loading, checkpointing, and distributed training
   - Integrates with HuggingFace models

3. **Inference** (`src/prime_rl/inference/`):
   - OpenAI-compatible API server with vLLM backend
   - Supports weight updates and reloading
   - Handles rollout generation

4. **Evaluation** (`src/prime_rl/eval/`):
   - Evaluation framework for environments
   - Registry system for managing eval tasks

5. **Core** (`src/prime_rl/core/`):
   - Currently minimal: `UniversalRollout`, `RLAlgorithm`, `OfflineRLAlgorithm`, `TrainingMetrics`
   - Needs expansion to match PRD requirements

### Current Environment Integration

- Uses `verifiers` library for environment abstraction
- Environments are loaded via `vf.load_environment(env_id, **args)`
- No standardized `EnvironmentAdapter` interface yet
- No `VerifierClient` abstraction for remote verification

### Current Data Flow

1. **Online RL**: Orchestrator → Environment → Inference → Trainer (closed loop)
2. **Offline RL**: Not yet implemented (Phase 1 goal)

## Planned Core Abstractions (Phase 1)

### 1. Core Module (`src/prime_rl/core/`)

#### `UniversalRollout`
- Framework-agnostic trajectory data structure
- Supports conversion to PyTorch tensors and JAX arrays
- Handles batched time-major and batch-major layouts
- Currently exists but needs enhancement for JAX compatibility

#### `RLAlgorithm` Interface
- Backend-agnostic algorithm interface
- Methods:
  - `init_state(rng, config) -> state`
  - `train_step(state, batch) -> (new_state, metrics)`
  - `compute_loss(rollouts) -> loss_dict`
- Currently exists but needs refinement

#### `Trainer` Interface
- Abstract base class for training loops
- Methods:
  - `run_training_loop()` with hooks for logging, checkpointing, evaluation
- Not yet implemented as abstract interface

#### `EnvironmentAdapter` Base Class
- Standardized interface for environments
- Methods: `reset()`, `step(action)`, `render()`, `close()`
- Will wrap existing `verifiers` environments
- Not yet implemented

#### `VerifierClient` Base Class
- Interface for remote reward/grading APIs
- Supports async verification with idempotency, retries, timeouts
- Not yet implemented

### 2. JAX Backend (`src/prime_rl/backends/jax/`)

#### Structure
```
src/prime_rl/backends/jax/
├── __init__.py
├── trainer.py          # JaxTrainer
├── algorithms/
│   ├── __init__.py
│   ├── dpo.py          # JaxDPO
│   └── cql.py          # JaxCQL (future)
└── data_loader.py      # JaxDataLoader
```

#### `JaxTrainer`
- Manages JAX training loop with `pmap`/`pjit`/`shard_map`
- Handles device mesh, mixed precision, gradient accumulation
- Implements `Trainer` interface

#### `JaxDPO`
- Direct Preference Optimization implementation in JAX
- Implements `RLAlgorithm` interface
- Processes preference-style interaction traces

#### `JaxDataLoader`
- High-performance JSONL/Parquet reader for Interaction Traces
- Shards traces across devices
- Prefetches to device memory
- Converts traces to `UniversalRollout`

### 3. Interaction Trace Standard

#### Schema (JSONL)
- `trace_id`: UUID identifier
- `environment_id`: Environment identifier
- `schema_version`: Version string
- `metadata`: Source, timestamps, etc.
- `steps`: Array of (observation, action, reward, done, logprobs, metadata)
- `final_outcome`: "success" | "failure" | enum
- `labels`: Optional supervision (human_score, preference_group_id, etc.)

#### Reader Implementation
- Validates required fields
- Tolerates partial traces
- Converts to `UniversalRollout` instances
- Supports sharding and prefetching

### 4. Unified Configuration Schema

#### Structure
```toml
[backend]
type = "jax"  # or "torch"
mode = "offline"  # or "online"
algorithm = "dpo"  # or "cql", "ppo", "grpo", etc.

[dataset]
path = "path/to/traces.jsonl"
schema_version = "v1"
sharding = { ... }

[model]
name = "meta-llama/Llama-3.1-8B"
...
```

### 5. CLI Entrypoint

#### Command
```bash
prime-rl train --mode offline --config path/to/config.toml
```

#### Implementation
- Parses unified config schema
- Selects backend (JAX for offline)
- Loads dataset via Interaction Trace reader
- Initializes `JaxTrainer` with `JaxDPO`
- Runs training loop

## Migration Strategy

### Phase 1 (Current)
- Build new JAX backend alongside existing PyTorch code
- Create core abstractions that both backends can use
- Implement Interaction Trace reader
- Add CLI entrypoint

### Phase 2 (Future)
- Refactor existing PyTorch trainer to use `Trainer` interface
- Refactor orchestrator to use `EnvironmentAdapter`
- Add `VerifierClient` for remote verification
- Unified config schema for both backends

### Phase 3 (Future)
- RLaaS control plane (`TrainingRun`, `project_id`/`run_id`)
- Multi-tenancy hooks
- Eval-to-prod dashboards

## Design Principles

1. **Backend Extensibility**: New backends can be added by implementing `RLAlgorithm` and `Trainer` interfaces
2. **Environment Plug-ins**: New environments integrate via `EnvironmentAdapter`
3. **Verifier Plug-ins**: New verifiers integrate via `VerifierClient`
4. **Multi-Modal Support**: Observations/actions are opaque to PRIME-RL core, allowing future multi-modal support
5. **Hosted RLaaS Ready**: Architecture supports future remote orchestration

## Testing Strategy

- Unit tests for all core abstractions
- Integration tests for JAX backend
- Tests for Interaction Trace reader (validation, conversion)
- CLI tests for end-to-end offline training flow

