## Using Verifiers and Environments with PRIME-RL

This document explains how to plug external **environments** and **verifiers** (including rubric-based verifiers such as RaR-style rewards [Rubrics as Rewards](https://arxiv.org/pdf/2507.17746) and BetterEvaluation-style rubrics [BetterEvaluation rubrics](https://www.betterevaluation.org/methods-approaches/methods/rubrics)) into PRIME-RL in just a few steps.

The goal is that any external provider (e.g. `PrimeIntellect-ai/verifiers`, PrimeIntellect dashboard environments) can be integrated via thin adapters and configuration, without changing the core training loops.

---

## 1. Concepts

- **Environment**: The world where the agent acts (browser, app, simulator, etc.), accessed through an `EnvironmentAdapter`.
- **Verifier**: A grading service that turns a prompt + model output (+ optional rubric) into a scalar reward, accessed through a `VerifierClient`.
- **Interaction Trace**: Standard JSONL schema used by PRIME-RL to store offline trajectories and labels (see `docs/product_spec.md` §3.2).

PRIME-RL already defines:

- `EnvironmentAdapter` and `AsyncEnvironmentAdapter` in `src/prime_rl/core/environment.py`
- `VerifierClient` and `VerificationRequest` in `src/prime_rl/core/verifier.py`

These are the only interfaces most integrations need to implement.

---

## 2. Integrating a New Environment Provider

### 2.1 Expected Environment Contract

An `EnvironmentAdapter` exposes a minimal, gym-like interface:

- `reset() -> observation`
- `step(action) -> (observation, reward, done, info)`
- `render()` and `close()` (optional)

Internally, the adapter can call:

- HTTP APIs,
- a Python SDK,
- local simulators,
- or any other mechanism.

PRIME-RL does not inspect or constrain the **content** of `observation` or `action` beyond being serializable and consistent with your verifier/env logic.

### 2.2 Example: PrimeIntellect Environments

For environments listed in the PrimeIntellect dashboard  
`https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars`

you can create an adapter in `src/prime_rl/core/adapters/prime_intellect_env_adapter.py` that:

- stores configuration (e.g. `environment_id`, endpoint URL, API key),
- implements `reset()` by calling the provider’s "reset environment" endpoint or SDK function,
- implements `step(action)` by calling the provider’s "step" endpoint or SDK, returning `(observation, reward, done, info)`.

Configuration example (TOML):

```toml
[environment]
type = "prime_intellect"
environment_id = "browser-gym-v1"
endpoint = "https://api.primeintellect.ai/environments"
api_key = "env:PRIME_INTELLECT_API_KEY"
```

The PRIME-RL trainer will construct this adapter based on `type` and route all environment calls through it.

---

## 3. Integrating a New Verifier (Rubrics, RLVR, etc.)

### 3.1 Verifier Concept

A verifier is any service that maps:

- `(prompt x, model output ŷ, optional rubric/metadata)` → **scalar reward** + optional per-criterion scores.

This aligns with:

- **RLVR** (Reinforcement Learning with Verifiable Rewards): binary or test-based correctness.
- **Rubrics as Rewards (RaR)** [Rubrics as Rewards](https://arxiv.org/pdf/2507.17746): multi-criteria rubrics aggregated into a reward.
- **BetterEvaluation rubrics** [BetterEvaluation rubrics](https://www.betterevaluation.org/methods-approaches/methods/rubrics): structured checklists turned into scoring functions.

In PRIME-RL, all of these are implemented as subclasses of `VerifierClient`.

### 3.2 VerifierClient Contract

`VerifierClient` handles:

- constructing a `VerificationRequest` (trace IDs, prompt, response, rubric, etc.),
- sending it to the underlying service (HTTP/gRPC/SDK),
- returning a scalar reward and optional metadata,
- handling retries/backoff (via `retry_with_backoff` utilities).

### 3.3 Example: Rubric-Based Verifier via PrimeIntellect

Suppose you have:

- a rubric specification per task (following RaR or BetterEvaluation patterns),
- a verifier service (from `https://github.com/PrimeIntellect-ai/verifiers` or your own) that implements:
  - `(x, ŷ, rubric) → reward` using explicit aggregation (Section 2.2 in [Rubrics as Rewards](https://arxiv.org/pdf/2507.17746)) or implicit aggregation via an LLM-as-judge.

You can implement `PrimeIntellectVerifierClient` that:

- packages an input as `VerificationRequest`,
- calls the verifier endpoint,
- parses the scalar reward and attaches any rubric-level scores to metadata.

Configuration example:

```toml
[verifier]
type = "prime_intellect_rar"
endpoint = "https://api.primeintellect.ai/verifier/rubric"
api_key = "env:PRIME_INTELLECT_API_KEY"
```

The online trainer will then:

- generate trajectories via `EnvironmentAdapter`,
- call `VerifierClient` to obtain rubric-based rewards,
- train with PPO/GRPO using those rewards.

---

## 4. Online RL: Environment + Verifier Together

For **online RL** with PyTorch (PPO/GRPO), you typically need only:

1. An `EnvironmentAdapter` implementation.
2. A `VerifierClient` implementation if rewards are not provided directly by the environment.
3. A small config file.

### 4.1 Minimal Online Config Example

```toml
[backend]
framework = "torch"
mode = "online"

[algorithm]
name = "grpo"            # or "ppo"

[environment]
type = "prime_intellect"
environment_id = "browser-gym-v1"
endpoint = "https://api.primeintellect.ai/environments"

[verifier]
type = "prime_intellect_rar"
endpoint = "https://api.primeintellect.ai/verifier/rubric"

[output]
output_dir = "outputs/prime_intellect_rar"
max_steps = 10000
checkpoint_every = 500
```

Run:

```bash
prime-rl train --mode online --config configs/online_prime_intellect.toml
```

PRIME-RL will:

- spin up the environment through `EnvironmentAdapter`,
- query the verifier through `VerifierClient` for each completion (using rubrics as needed),
- update the policy with GRPO/PPO.

---

## 5. Offline RL: Using Verifiers to Build Interaction Traces

For **offline RL** with JAX (DPO/CQL), environments and verifiers are used *before* training to create datasets of Interaction Traces.

### 5.1 Data Collection

1. Run agents in your external environment (via your adapter or provider UI).
2. For each episode/trace:
   - record steps: observations, actions, rewards,
   - call your verifier to obtain final rubric-based scores or labels,
   - store them in the Interaction Trace JSONL schema (see `docs/product_spec.md`).

Example trace (abridged):

```json
{
  "trace_id": "uuid",
  "environment_id": "browser-gym-v1",
  "steps": [...],
  "final_outcome": "success",
  "labels": {
    "rubric_score": 0.87,
    "preference_group_id": "group-42"
  }
}
```

### 5.2 Offline Config Example

```toml
[backend]
framework = "jax"
mode = "offline"

[dataset]
path = "data/prime_intellect_rar_traces.jsonl"
schema_version = "v1"

[algorithm]
name = "dpo"

[output]
output_dir = "outputs/offline_prime_intellect_rar"
max_steps = 2000
checkpoint_every = 200
```

Run:

```bash
prime-rl train --mode offline --config configs/offline_prime_intellect.toml
```

The JAX Offline Engine will then:

- load Interaction Traces,
- create preference pairs (reward- or `preference_group_id`-based),
- train a DPO policy using those offline rubrics.

---

## 6. Relation to Rubrics-as-Rewards & DR Tulu

The design in this document is compatible with recent rubric-based and deep-research RL work:

- **Rubrics as Rewards (RaR)** [Rubrics as Rewards](https://arxiv.org/pdf/2507.17746):  
  - PRIME-RL’s `VerifierClient` can directly implement RaR-style **explicit** or **implicit** aggregation (Section 2.2 in the paper) by treating rubrics as part of the verifier input and returning the normalized scalar reward.
- **DR Tulu / RLER** [DR Tulu: DR Tulu blog](https://allenai.org/blog/dr-tulu):  
  - Evolving, instance-specific rubrics and positive/negative rubric buffers can live entirely inside the verifier service.  
  - From PRIME-RL’s perspective, the verifier still just exposes `(x, ŷ, rubric_state) → reward`, and the evolving rubric buffer is an implementation detail of the verifier.  
  - MCP-style multi-tool research stacks (like `dr-agent-lib` in DR Tulu) map naturally onto PRIME-RL’s `EnvironmentAdapter` and integrations layer (see `docs/task_integrations_layer.md`), where each research tool is a callable action within the environment.

In other words, PRIME-RL focuses on **how to consume rewards and trajectories**, while RaR/RLER/DR-Tulu-style systems focus on **how to generate high-quality rubric-based rewards**. Connecting the two only requires a suitable `VerifierClient`.

---

## 7. “Few Steps” Integration Checklist

To integrate a new environment + verifier stack:

1. **Environment**  
   - Implement an `EnvironmentAdapter` (or `AsyncEnvironmentAdapter`) that forwards `reset`/`step` to the provider’s API/SDK.
2. **Verifier**  
   - Implement a `VerifierClient` subclass that calls your chosen verifier (RLVR or rubric-based) and returns a scalar reward.
3. **Config**  
   - Add a TOML config that wires together:
     - `backend` (jax/torch, offline/online),
     - `algorithm` (dpo/cql/ppo/grpo),
     - `environment` (type, IDs, endpoints),
     - `verifier` (type, endpoints),
     - `output` (logging/checkpointing).
4. **Run**  
   - Use `prime-rl train --mode online` or `--mode offline` with that config.

No changes to PRIME-RL’s core algorithms are required—everything is done through these thin adapter layers and the standardized Interaction Trace format.


