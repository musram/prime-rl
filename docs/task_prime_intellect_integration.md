# Implementation Task: Prime Intellect Integrations

**Context**: 
We have confirmed that the strategy in `docs/verifiers_and_environments_integration.md` is the correct way to integrate with the Prime Intellect ecosystem. We now need to implement the concrete adapters that connect `prime-rl` to the Prime Intellect Dashboard and Verifier APIs.

**Objective**: 
Implement the `prime_intellect` integration package containing an Environment Adapter and a Verifier Client.

**Requirements**:

1.  **Directory Structure**:
    *   Create `src/prime_rl/integrations/prime_intellect/`
    *   Include `__init__.py`, `adapter.py` (Environment), and `client.py` (Verifier).

2.  **Environment Adapter (`adapter.py`)**:
    *   Implement `PrimeIntellectEnvAdapter` inheriting from `EnvironmentAdapter`.
    *   **Config**: Should take `environment_id`, `endpoint`, and `api_key`.
    *   **Behavior**:
        *   `reset()`: POST to `{endpoint}/reset` with `{env_id}`. Return observation.
        *   `step(action)`: POST to `{endpoint}/step` with `{action}`. Return `(obs, reward, done, info)`.
    *   *Note*: Use `httpx` or `requests` for API calls. Handle JSON serialization.

3.  **Verifier Client (`client.py`)**:
    *   Implement `PrimeIntellectVerifierClient` inheriting from `VerifierClient`.
    *   **Config**: Should take `endpoint` (e.g. `.../verifier/rubric`) and `api_key`.
    *   **Behavior**:
        *   `verify(rollout)`: Construct a payload matching the PI Verifier API (prompt, completion, rubric metadata).
        *   POST to the endpoint.
        *   Extract the scalar `reward` and any `metadata` (e.g., rubric scores) from the response.

4.  **Configuration Factory**:
    *   Update `src/prime_rl/core/registry.py` (or create it if missing) to map the string `"prime_intellect"` in `config.toml` to these new classes.

**Deliverables**:
*   The python module `src/prime_rl/integrations/prime_intellect/`.
*   Unit tests in `tests/unit/integrations/test_prime_intellect.py` (mocking the HTTP calls).

**Instruction to AI**:
> "Please execute the plan above. 
> 1. Create the folder `src/prime_rl/integrations/prime_intellect/`.
> 2. Implement the `PrimeIntellectEnvAdapter` in `adapter.py` with robust error handling for HTTP requests.
> 3. Implement the `PrimeIntellectVerifierClient` in `client.py` supporting rubric metadata passing.
> 4. Ensure strict typing and docstrings matching the `docs/verifiers_and_environments_integration.md` spec."

