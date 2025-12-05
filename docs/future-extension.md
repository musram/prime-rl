# Future Vision: The Infrastructure for Verifiable Work

> *Based on the thesis "The TAM Is All of Human Labor"*
https://docs.google.com/document/d/1WZzkanbd3thyi8OtiOoit7mDgH9RXC4YDEBmbqYp2ro/edit?pli=1&tab=t.0

## Mission

PRIME-RL aims to be the fundamental infrastructure layer for the next generation of AI capability: **models that learn to work**. 

As the industry shifts from static pre-training to dynamic mid/post-training, the bottleneck is no longer just data volume, but **verifiability**—the ability to define "correctness" in complex, multi-step workflows. PRIME-RL provides the high-performance, scalable training stack needed to turn these verifiable environments into trained model capabilities.

## Strategic Pillars

### 1. The "RL as a Service" (RLaaS) Engine
The market is moving towards managed platforms where enterprises can upload a process (simulated as an environment) and train a custom agent. PRIME-RL serves as the low-level engine for these platforms.

*   **Goal**: Make spinning up a training run on a custom business environment as easy as spinning up a web server.
*   **Key Features**:
    *   **Massive Parallelism**: Orchestrating thousands of concurrent environment steps (inference + verification) efficiently.
    *   **Determinism**: Ensuring reproducible training runs despite asynchronous execution.
    *   **Cost Efficiency**: Optimizing the ratio of inference tokens to training updates.

### 2. Offline RL & Interaction Data
Before models can interact live, they often need to learn from logs of human or expert agent interactions ("reasoning traces").

*   **Goal**: Enable learning from static "interaction datasets" before deploying to live environments.
*   **Key Features**:
    *   **JAX Backend**: Leveraging JAX for high-performance Offline RL (DPO, CQL, etc.) to process massive interaction logs (see `docs/jax_integration.md`).
    *   **Data Standardization**: Ingesting traces from various sources (browser usage, coding sessions, tool use logs).

### 3. The Environment Marketplace
PRIME-RL integrates natively with the ecosystem of verifiable environments.

*   **Goal**: Be the universal "player" for any verifiable environment.
*   **Key Features**:
    *   **Standardized Interfaces**: Deep integration with `verifiers` and support for emerging standards in "browser use" or "computer use" environments.
    *   **Verifiability First**: First-class support for complex reward signals, grading DSLs, and multi-modal feedback.

## Roadmap

### Phase 1: Foundation (Current)
*   Stable PyTorch implementation for Online RL (PPO/GRPO).
*   Integration with `verifiers`.
*   Scalable Orchestrator/Trainer architecture.

### Phase 2: Offline & Scale (Near Term)
*   **JAX Integration**: Implement high-performance offline RL algorithms.
*   **Orchestrator Scaling**: Optimize for 10k+ concurrent environments.
*   **Evaluation Suite**: robust "eval-to-prod" correlation tracking.

### Phase 3: The Universal Stack (Long Term)
*   **Multi-Modal Agents**: Native support for VLM-based agents (browser/computer use).
*   **Hosted RLaaS**: API-driven training for enterprise environments.
*   **Ecosystem integration**: Plug-and-play support for third-party data and environment providers (e.g., specialized medical or legal envs).

