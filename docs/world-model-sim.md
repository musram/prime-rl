# World Model Simulation

## 1. Clarify the target

The gap is:

> Environment–World Model (learned simulators): No world‑model component; all envs are hand‑coded or tool‑based.

Concretely, we want a learned environment that can be used through the same `EnvironmentAdapter` interface, trained from `InteractionTrace` logs, so PRIME‑RL can:

- Roll out agents cheaply in simulation ("model‑imagined world").
- Mix real and simulated experience (Dyna / model‑based RL).
- Quantify the sim‑to‑real gap using existing verifiers.

Think of it as: "a `WorldModelEnvAdapter` that pretends to be CRM / browser / finance env, but is backed by a learned model."

## 2. Start narrow: one sandbox, one world model

Pick one environment you control end‑to‑end, ideally:

- CRM sandbox (`src/prime_rl/sandboxes/crm/`) or
- Finance sandbox (`src/prime_rl/sandboxes/finance/`),

because:

- State is structured (tickets, ledgers), not arbitrary web.
- You can generate lots of trajectories.
- You already have verifiers and rubrics to evaluate behavior.

### Define a clean modeling target:

Let $s_t$ be a structured state (ticket fields, queue state, etc.).

Let $a_t$ be the agent action (e.g., "reply(template_id=3)", "post_credit(amount=100)").

Let $o_t$ be the observation shown to the agent (usually a rendering of $s_t$).

Let $r_t$ and $\text{info}_t$ come from the verifier.

We want a model $p_\theta(s_{t+1}, r_t, \text{info}_t \mid s_t, a_t)$ or, more pragmatically:

- **Option A (structured)**: model next structured state $s_{t+1}$ and reward.
- **Option B (text‑heavy)**: model next observation $o_{t+1}$ + reward directly as text/JSON.

For your CRM/finance sandboxes, Option A (structured) is easier and more stable.

## 3. Data & logging: extend InteractionTrace minimally

You already have:

- `InteractionTrace` with steps: `[{observation, action, reward, done, metadata}]`.

To train a world model:

- Ensure metadata contains a canonical state snapshot for each step:
  - e.g. `metadata["state"] = { tickets: [...], user_status: ... }` for CRM.
- For tool‑rich envs, log tool calls & results in a structured way so they can be part of state or action.

This gives you supervised pairs:

- **Input**: $(s_t, a_t)$
- **Target**: $(s_{t+1}, r_t, \text{done}_t, \text{verifier\_labels}_t)$

## 4. Model architecture (first pass, not fancy)

For the first Phase WM‑1:

Implement a simple JAX or PyTorch MLP/Transformer that:

1. Encodes $s_t$ to a vector.
2. Encodes $a_t$ (categorical or parameterized).
3. Concatenates and predicts:
   - next state diff $\Delta s = s_{t+1} - s_t$ or directly $s_{t+1}$,
   - reward $r_t$,
   - $\text{done}_t$.

### Concretely:

For CRM:

- One‑hot / embedding for ticket statuses, agent id, etc.
- Dense layers → outputs logits for categorical fields, real values for numeric fields.
- Loss: sum of cross‑entropy (categorical) + MSE (numeric) + BCE (done).

Optionally add an auxiliary head to predict verifier scores (rubric scores) from $s_{t+1}$, so the world model learns what matters for rewards.

Keep this in a new module, e.g.:

- `src/prime_rl/world_models/base.py` – interface.
- `src/prime_rl/world_models/crm_world_model.py` – first concrete implementation.

## 5. Wrap it in a WorldModelEnvAdapter

Define `WorldModelEnvAdapter(EnvironmentAdapter)`:

### Holds:

- A trained world model `wm`.
- A current latent or structured state $s_t$.
- A cheap internal verifier head (optional) to produce $r_t$, or you just predict $r_t$ directly.

### Implements:

**`reset()`:**

- Sample an initial state $s_0$ from empirical distribution (from logs) or a small generative prior.
- Return a rendered observation derived from $s_0$ (same format as real env).

**`step(action)`:**

- Map incoming action to the action encoding used by `wm`.
- Predict $s_{t+1}, r_t, \text{done}_t$ via the world model.
- Update internal state.
- Return $(\text{observation}_{t+1}, r_t, \text{done}_t, \text{info}_t)$.

The key: from the trainer's POV, this is just another environment; you can switch between:

- Real CRM env adapter.
- `WorldModelEnvAdapter`.

via config.

## 6. Use it safely: model‑based RL patterns

Once `WorldModelEnvAdapter` exists, there are three incremental usages:

### Pure offline data augmentation (no closed loop)

- Sample synthetic trajectories from the world model to create extra `InteractionTrace` data.
- Train offline (DPO/CQL) on real + synthetic traces; use verifiers to check if performance improves.

### Dyna‑style imagination steps

In online training:

- For each real env step, run $k$ imaginary steps in the world model with the current policy.
- Mix real and synthetic transitions in the replay buffer (weight real transitions higher).

### Full model‑based control in sandbox

- Train a policy purely in the world model, then fine‑tune or validate in the real sandbox.
- Use your existing verifiers to measure the sim‑to‑real gap:
  - Run the same policy in real vs. simulated env and compare reward distributions and rubric scores.

At each stage, use your verifiers (esp. rubric‑based ones) as a diagnostic: they're ideal for catching world‑model failure modes (reward hacking, unrealistic sequences).

## 7. Evaluation & research questions

As a research scientist, the key questions you can study:

### Accuracy vs utility:

How accurate does the world model need to be (per‑step, multi‑step) before synthetic rollouts are actually helpful?

### Verifier‑aligned metrics:

Instead of only predicting low‑level transitions, can we predict verifier outputs (rubric scores, success probabilities) as an auxiliary task, to bias the model towards RL‑relevant aspects?

### Hybrid modeling of tools:

For expensive tools (web search, DB queries), can the world model approximate their distribution coarsely while still calling the real tool occasionally to stay grounded?

You can design small, controlled experiments in CRM/finance sandboxes to answer these, using:

- Held‑out real trajectories as ground truth.
- Existing verifiers as judges of both world‑model realism and policy performance.

## 8. Minimal concrete roadmap

### Phase WM‑1 (MVP)

- Add `world_models/` module and `WorldModelEnvAdapter`.
- Train a simple next‑state + reward predictor for CRM sandbox from existing traces.
- Support offline synthetic data generation; evaluate with verifiers.

### Phase WM‑2 (Model‑based RL in sandbox)

- Integrate `WorldModelEnvAdapter` with Torch trainer:
  - Dyna‑style imagination steps,
  - pure simulated training + real fine‑tune comparison.
- Log sim‑to‑real performance gaps per verifier metric.

### Phase WM‑3 (Generalization)

- Extend to second domain (finance sandbox or browser‑gym).
- Explore more sophisticated latent world models (Dreamer‑style) if gains justify complexity.

---

This path turns the current "❌ High gap" into a bounded, staged research program that fits cleanly into PRIME‑RL's abstractions and uses your existing verifiers as both reward engines and scientific instruments.
