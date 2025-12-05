# Product Requirement Document: PRIME-RL as the RLaaS Engine

## 1. Overview & Vision

**Objective**: Transform `prime-rl` from a research repository into a production-grade **RL-as-a-Service (RLaaS) Engine** for verifiable work.  
**Thesis**: As outlined in *The TAM Is All of Human Labor*, the future of AI is models learning to work via **Offline Interaction Data** and **Online Verifiable Environments**. `prime-rl` is the **engine layer** that turns those environments and traces into trained capabilities.

The product must:

*   **Support both Offline and Online RL** for complex, long-horizon tasks.
*   **Expose a clean RLaaS surface area** (APIs/CLIs) for higher-level platforms.
*   **Integrate with an ecosystem of environments and data vendors** via simple, stable contracts.
*   **Prioritize verifiability, observability, and determinism** over ad-hoc experimentation.

## 2. Core Architecture: Dual Engines + RLaaS Layer

The system supports two distinct but compatible modes of operation, abstracting away the underlying framework (PyTorch/JAX) and deployment environment.

### 2.0 Mapping to TAM Layers (L0–L4)

PRIME-RL is explicitly aligned with the functional RL environment layers described in `TAM-doc.md`:

*   **Layer 0 – Orchestration**  
    *Handled by*: the Orchestrator and RLaaS control plane.  
    *Where in PRD*: §2.3 (RLaaS Control Plane), §3.5 (Orchestration, Scaling & Observability), §4.2 (Distributed Orchestrator).  
    *What PRIME-RL does*: spin up and reset thousands of runs, route environment steps across workers, manage `run_id`/`trace_id`, and expose metrics/logs for determinism and observability.

*   **Layer 1 – Environment**  
    *Handled by*: `EnvironmentAdapter` implementations.  
    *Where in PRD*: §2.2 (Environment Adapters), §3.1 (Environment & Verifier Contracts).  
    *What PRIME-RL does*: provide a stable interface over concrete worlds (browser envs, internal tools, simulators) so that engines and algorithms are decoupled from specific UIs/SDKs.

*   **Layer 2 – Data**  
    *Handled by*: the Interaction Trace standard, dataset registry, and loaders.  
    *Where in PRD*: §2.1 (Dataset Registry & Validation), §3.2 (Interaction Trace), §3.3 (`JaxDataLoader`).  
    *What PRIME-RL does*: normalize real/synthetic interaction data into a common schema and `UniversalRollout`, with validation and efficient sharding/prefetching for large-scale training.

*   **Layer 3 – Tasks & Rewards**  
    *Handled by*: algorithms + env/verifier integration.  
    *Where in PRD*: §2.1/§2.2 (algorithms for offline/online), §3.1 (RLAlgorithm/Trainer), §3.2 (labels/final_outcome).  
    *What PRIME-RL does*: encode “what to do” and “how to score it” through algorithm configs, environment setups, and reward signals from envs or verifiers, for both offline (DPO/CQL) and online (PPO/GRPO) modes.

*   **Layer 4 – Verification**  
    *Handled by*: `VerifierClient` and safety/alignment hooks.  
    *Where in PRD*: §2.2 (Remote Verification), §3.1 (`VerifierClient`), §3.6 (Safety, Alignment & Governance).  
    *What PRIME-RL does*: call out to remote graders, record their decisions in traces/metrics, provide auditability of rewards, and expose hooks to detect reward hacking or misaligned behavior.

This mapping ensures that anyone familiar with the TAM layers can directly see how PRIME-RL implements each layer in code and configuration.

### 2.1 Engine A: The Offline Engine (Data-First)

**Use Case**: Learning from large-scale logs of human or agent interaction traces (e.g., "reasoning traces", browser/computer use logs, coding traces).  
**Primary Backend**: **JAX** (for high-throughput batch processing and TPU/GPU scaling).

**Key Requirements**:

*   **Universal Data Ingestion**: A standardized JSONL format for "Interaction Traces" (see §3.2).
*   **High-Performance Offline RL Algorithms**:
    *   DPO (Direct Preference Optimization) for preference-style data.
    *   CQL (Conservative Q-Learning) / related offline RL for trajectory data.
*   **Dataset Registry & Validation**:
    *   Declarative dataset configs (paths, schema version, sharding).
    *   Lightweight schema validation and stats logging at load time.
*   **Scale**: Capable of processing terabyte-scale datasets using TPUs/GPUs with efficient sharding and prefetching.

### 2.2 Engine B: The Online Engine (Environment-First)

**Use Case**: Fine-tuning agents against live, verifiable environments (e.g., simulators, browser/computer-use envs, task-specific sandboxes).  
**Primary Backend**: **PyTorch** (for compatibility with vLLM and broader ecosystem tools).

**Key Requirements**:

*   **Massive Orchestration**: A horizontally scalable Orchestrator capable of managing **10,000+** concurrent environment steps across workers.
*   **Environment Adapters**:
    *   Pluggable `EnvironmentAdapter` interface for different env providers (browser sims, internal tools, custom simulators).
    *   Support for both synchronous and asynchronous step APIs.
*   **Remote Verification**:
    *   First-class support for asynchronous, remote reward signals via a `VerifierClient` (HTTP/gRPC).
    *   Clear contracts for idempotency, retries, and timeouts.
*   **Determinism**:
    *   Strict reproducibility guarantees (seed handling, logged configs, pinned versions).
    *   Ability to **replay** an interaction trace in the same environment configuration.

### 2.3 RLaaS Control Plane (Logical Layer)

Above the engines, PRIME-RL exposes a thin RLaaS control plane:

*   **Run Abstraction**: A `TrainingRun` object (config + code revision + dataset/environment reference).
*   **Multi-Tenancy Ready**: All state scoped by `project_id` / `run_id` to allow future hosted RLaaS.
*   **APIs & CLI**:
    *   Python SDK for library-style use.
    *   CLI for batch and local development (see §4.3).

Implementation of the control plane can start minimal (local-only metadata) but must not preclude later remote orchestration.

## 3. Technical Specifications

### 3.1 Core Abstraction Layer (`src/prime_rl/core`)

To support dual backends without code duplication, we implement a rigorous interface layer.

*   **`UniversalRollout`**: A framework-agnostic data structure that holds trajectory data.
    *   Must support efficient conversion to both Torch Tensors and JAX Arrays (zero/low copy where feasible).
    *   Handles batched time-major and batch-major layouts.
*   **`RLAlgorithm` Interface** (backend-agnostic):
    *   `init_state(rng, config) -> state`
    *   `train_step(state, batch) -> (new_state, metrics)`
    *   `compute_loss(rollouts) -> loss_dict`
*   **`Trainer` Interface**:
    *   `run_training_loop()` with hooks for logging, checkpointing, and evaluation.
*   **Environment & Verifier Contracts**:
    *   `EnvironmentAdapter` base class: `reset()`, `step(action)`, `render()`, `close()`.
    *   `VerifierClient` base class for remote reward / grading APIs.
*   **Configuration Schema**:
    *   Unified TOML/YAML configuration that selects:
        *   backend (`jax` / `torch`),
        *   mode (`offline` / `online`),
        *   algorithm (`dpo`, `cql`, `ppo`, `grpo`, etc.),
        *   environment or dataset.

### 3.2 The "Interaction Trace" Data Standard

We define a standard schema for offline data to serve as the integration point for data vendors and environment providers. The minimal trace unit is a **trajectory**.

```json
{
  "trace_id": "uuid",
  "environment_id": "browser-gym-v1",
  "schema_version": "v1",
  "metadata": {
    "source": "halluminate",
    "created_at": "2025-11-30T12:34:56Z"
  },
  "steps": [
    {
      "t": 0,
      "observation": "...",              // opaque to PRIME-RL, env-specific
      "action": "click(10, 20)",         // can be structured or stringified
      "reward": 0.0,
      "done": false,
      "logprobs": null,                  // optional: policy logprobs if available
      "metadata": {"url": "https://google.com"}
    },
    {
      "t": 1,
      "observation": "...",
      "action": "type('hello')",
      "reward": 1.0,
      "done": true,
      "logprobs": [/* optional */],
      "metadata": {}
    }
  ],
  "final_outcome": "success",            // or "failure", enum-like
  "labels": {                            // optional supervision
    "human_score": 0.9,
    "preference_group_id": "pref-123"    // for DPO-style comparisons
  }
}
```

**Notes**:

*   PRIME-RL must tolerate partial traces (missing fields) and provide clear validation/logging.
*   The reader should expose a typed Python object (e.g., `InteractionTrace`) that can be mapped into `UniversalRollout`.

### 3.3 JAX Integration Spec (`src/prime_rl/backends/jax`)

**Location**: `src/prime_rl/backends/jax`

**Components**:

*   `JaxTrainer`:
    *   Manages the training loop, checkpointing, and device mesh (`pmap` / `pjit` / `shard_map`).
    *   Handles mixed precision and gradient accumulation for large-batch training.
*   `JaxAlgorithm` implementations:
    *   `JaxDPO`, `JaxCQL`, etc., implementing the `RLAlgorithm` interface.
*   `JaxDataLoader`:
    *   Efficient multi-threaded data loader for the "Interaction Trace" format (JSONL/Parquet).
    *   Shards traces across devices and prefetches to device memory.

JAX code should be written with **functional, pure-style APIs** and minimize host-device transfers.

### 3.4 PyTorch Online Integration Spec (`src/prime_rl/backends/torch`)

**Location**: `src/prime_rl/backends/torch`

**Components**:

*   `TorchTrainer` for online RL (PPO/GRPO-style algorithms).
*   Integrations with:
    *   the Orchestrator (for environment stepping),
    *   `EnvironmentAdapter` instances,
    *   remote `VerifierClient`s.
*   Support for model loading from standard LLM engines (e.g., vLLM, Hugging Face) via simple adapters.

### 3.5 Orchestration, Scaling & Observability

**Orchestrator**:

*   Refactor `orchestrator.py` into a reusable module that:
    *   Schedules environment steps across local/remote workers.
    *   Supports backpressure and rate limiting for verifier calls.
    *   Exposes a simple gRPC/HTTP control plane (future-ready; can start as in-process).

**Observability & Telemetry**:

*   Structured logging with trace IDs (`trace_id`, `run_id`, `env_instance_id`).
*   Metrics for:
    *   reward distributions,
    *   environment step latency and error rates,
    *   token usage (if applicable),
    *   eval-to-prod correlation hooks (integration point for W&B, Prometheus, etc.).

### 3.6 Safety, Alignment & Governance (Foundational Hooks)

The engine must include **hooks** (not full policies) for:

*   **Reward Hacking Detection**: Simple metrics and logging to detect anomalous reward patterns.
*   **Policy Gating**: Optional pre-/post-processing hooks to integrate external safety filters.
*   **Auditability**: Ability to reconstruct why a given reward was assigned (link to verifier inputs/outputs).

These are initially "extension points" for downstream products, not fully built safety systems.

## 4. Execution Roadmap

### 4.1 Phase 1: Foundation (The Offline Engine)

**Goal**: Ship a minimal, high-quality Offline Engine in JAX with clean abstractions and tests.

1.  **Refactor Core**:
    *   Implement `src/prime_rl/core` abstractions: `UniversalRollout`, `RLAlgorithm`, `Trainer`, `EnvironmentAdapter`, `VerifierClient`.
    *   Define and document the unified configuration schema.
2.  **Implement JAX Backend (MVP)**:
    *   Create `src/prime_rl/backends/jax` with `JaxTrainer` and a basic `JaxDPO` implementation.
3.  **Data Loader**:
    *   Build a high-performance reader for Interaction Traces (JSONL) that produces `UniversalRollout`s.
4.  **Basic CLI**:
    *   `prime-rl train --mode offline --config path/to/config.toml` running a local JAX job.

### 4.2 Phase 2: Scale (The Online Engine)

**Goal**: Add PyTorch-based Online Engine and scalable orchestration.

1.  **Distributed Orchestrator**:
    *   Refactor `orchestrator.py` to support worker pools (local first; remote workers later).
    *   Implement basic retry/backoff semantics for env/verifier failures.
2.  **PyTorch Backend**:
    *   Implement `TorchTrainer` and a reference PPO/GRPO algorithm using `EnvironmentAdapter`.
3.  **Verifier API**:
    *   Standardize the protocol for remote environment verification (request/response schema + error model).

### 4.3 Phase 3: Productization & RLaaS Surface

**Goal**: Turn the engine into a productizable RLaaS core.

1.  **Unified CLI & Python SDK**:
    *   `prime-rl train --mode offline/online --config ...`
    *   Python API for embedding PRIME-RL in other systems.
2.  **Eval-to-Prod Dashboards**:
    *   Pre-configured dashboards (e.g., via W&B or similar) for "eval-to-prod" correlation tracking.
3.  **Run Metadata & Multi-Tenancy Hooks**:
    *   Simple `run_id`/`project_id` abstractions and local metadata store (upgradeable to remote DB).

## 5. Prompt for AI Developer

To the AI Developer:

> Using the plan above, please begin execution of **Phase 1 (Foundation: Offline Engine)**.  
>  
> 1. Analyze the existing `src/prime_rl` structure and document a short architecture note (`docs/arch_core.md`) summarizing current components and planned core abstractions.  
> 2. Create the `src/prime_rl/core` directory and implement the `UniversalRollout`, `RLAlgorithm`, and `Trainer` base classes as strictly typed, well-documented interfaces.  
> 3. Define the `EnvironmentAdapter` and `VerifierClient` abstract base classes with clear method contracts and docstrings.  
> 4. Draft the `src/prime_rl/backends/jax` directory structure (including stubs for `JaxTrainer` and `JaxDPO`) and ensure they conform to the `RLAlgorithm`/`Trainer` interfaces.  
> 5. Implement an initial Interaction Trace reader that:  
>    * reads JSONL traces in the schema of §3.2,  
>    * validates required fields, and  
>    * converts them into `UniversalRollout` instances.  
> 6. Add a minimal CLI entrypoint `prime-rl train --mode offline --config path/to/config.toml` that wires together config → data loader → `JaxTrainer` stub (even if the algorithm is initially a no-op).  
> 7. Ensure all new code is type-hinted, covered by basic unit tests, and documented (docstrings + brief usage examples in `docs/`).

## 6. Future Extensions Alignment

`future-extension.md` describes longer-term goals such as **multi-modal agents**, **hosted RLaaS**, and a rich **environment/data marketplace**. The architecture in this PRD must stay compatible with those directions:

*   **Backend Extensibility**:
    *   New backends (e.g., additional JAX/PyTorch variants or other frameworks) can be added by implementing the `RLAlgorithm` and `Trainer` interfaces and wiring into the unified config, without changing callers.
*   **Environment & Verifier Plug-ins**:
    *   New environment types (e.g., VLM-based browser agents, desktop sims) and verifiers (reward models, human-in-the-loop graders) integrate by implementing `EnvironmentAdapter` and `VerifierClient`, preserving the same orchestrator and training loops.
*   **Multi-Modal Support**:
    *   Observations and actions in the Interaction Trace schema are intentionally opaque to PRIME-RL, allowing future support for text, images, video, and structured tool calls as long as envs/adapters know how to encode/decode them.
*   **Hosted RLaaS & Marketplace**:
    *   The RLaaS control plane (`TrainingRun`, `project_id`/`run_id`, config-driven runs) and structured telemetry are designed so they can later be backed by a remote service (DB/queue/API) rather than just local files, enabling a hosted PRIME-RL and an environment/data marketplace on top.

These principles should guide implementation decisions so near-term code does not block the future product roadmap.
