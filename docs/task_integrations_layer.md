# Implementation Task: Build the "Integrations Layer" for RLaaS

**Context**: 
We have built the core `prime-rl` engine (Offline JAX + Online PyTorch) with abstract interfaces (`EnvironmentAdapter`, `VerifierClient`). To fulfill the product vision of being an "RLaaS Engine" that integrates with the verifiable work ecosystem (as described in `TAM-doc.md`), we now need to build the concrete **Integrations Layer**.

**Objective**: 
Create the `src/prime_rl/integrations/` module to provide "batteries-included" adapters for common environment standards and remote verification services. This makes `prime-rl` ready to plug into the "browser use" and "remote grading" ecosystem immediately.

**Requirements**:

1.  **Directory Structure**:
    *   Create `src/prime_rl/integrations/`
    *   Create `src/prime_rl/integrations/browser_gym/` (for browser environments)
    *   Create `src/prime_rl/integrations/remote/` (for generic HTTP/gRPC verifiers)

2.  **BrowserGym Integration**:
    *   Implement `BrowserGymAdapter` inheriting from `EnvironmentAdapter`.
    *   It should wrap a standard Gym/Gymnasium environment (specifically targeting `browsergym` interfaces if available, or generic Gymnasium).
    *   Map observations (e.g., DOM snapshot, accessibility tree) to the `UniversalRollout` format.

3.  **Remote Verifier Integration**:
    *   Implement `HttpVerifierClient` inheriting from `VerifierClient`.
    *   It should take an endpoint URL and API key.
    *   Implement `verify(rollout)` by sending a POST request with the interaction trace and returning the reward/metadata from the response.
    *   Include basic retry logic (exponential backoff) for robustness.

4.  **Testing**:
    *   Add unit tests mocking the HTTP calls for `HttpVerifierClient`.
    *   Add a simple test for `BrowserGymAdapter` using a dummy Gym environment.

**Deliverables**:
*   Python code in `src/prime_rl/integrations/`
*   Unit tests in `tests/unit/integrations/`

**Instruction to AI**:
> "Please execute the plan above. 
> 1. Create the folder structure.
> 2. Implement `HttpVerifierClient` first (as it's critical for the 'Remote Verification' requirement).
> 3. Implement `BrowserGymAdapter` (wrapping `gymnasium.Env`).
> 4. Ensure all code is strictly typed and documented."

