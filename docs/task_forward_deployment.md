# Implementation Task: Forward Deployment (Client-Server) Architecture

**Context**: 
To support "Forward Deployment Execution" (where environments run on a user's laptop or inside a secure enterprise VPC while training happens on a centralized GPU cluster), we need to evolve PRIME-RL from a monolithic architecture to a **Client-Server Architecture**. This matches the capability of tools like ART and enables "Laptop-to-Cluster" training flows.

**Objective**: 
Implement a lightweight **Remote Worker Protocol** that allows an `EnvironmentAdapter` running anywhere to stream data to the PRIME-RL Orchestrator and receive policy updates.

## 1. Architecture Overview

### The "Forward Deployed" Worker (Client)
*   Runs on the user's infrastructure (Laptop, On-prem, specialized hardware).
*   Executes the Environment (e.g., a local browser, a proprietary simulator).
*   Uses the **PRIME-RL Client SDK** to communicate with the server.

### The PRIME-RL Cluster (Server)
*   Runs the Trainer (GPUs) and Orchestrator.
*   Exposes a **Worker API** (HTTP/gRPC) to accept traces and serve policy weights/actions.

## 2. Technical Requirements

### 2.1 The Worker API (Server-Side)
Implement a new API module in the Orchestrator (`src/prime_rl/orchestrator/api.py`):
*   `POST /v1/workers/register`: Handshake to register a new worker session.
*   `POST /v1/traces`: Endpoint to receive compressed `UniversalRollout` objects (Interaction Traces).
*   `GET /v1/policy/latest`: Endpoint for workers to download the latest policy weights (or LoRA adapters).
*   *(Optional Phase 2)* `POST /v1/actions`: For "Inference-as-a-Service" mode where the server decides the action.

### 2.2 The Remote Worker Client (Client-Side)
Create a new module `src/prime_rl/client/worker.py`:
*   **`RemoteEnvironmentWorker`**: A class that wraps a local `EnvironmentAdapter`.
*   **Loop**:
    1.  Poll server for latest Policy (if running local inference) OR send observation to server (if remote inference).
    2.  Step local environment.
    3.  Buffer trajectory.
    4.  Async push `UniversalRollout` to `POST /v1/traces`.

### 2.3 Security & Auth
*   Add simple Token-based authentication (`PRIME_RL_API_KEY`) to the Worker API.

## 3. Implementation Plan

1.  **Define the Protocol**: Create the Pydantic models for the API requests/responses in `src/prime_rl/core/protocol.py`.
2.  **Build the Server API**: Add a lightweight FastAPI router to the Orchestrator.
3.  **Build the Client SDK**: Implement the `RemoteEnvironmentWorker` loop.
4.  **Example**: Create a `examples/remote_browser/` demo showing a browser running locally training a model on the cluster.

**Instruction to AI**:
> "Please execute the plan for the **Forward Deployment Architecture**.
> 1. Create `src/prime_rl/core/protocol.py` defining the data exchange format.
> 2. Implement the `RemoteEnvironmentWorker` client in `src/prime_rl/client/`.
> 3. Stub out the Server API endpoints in the Orchestrator.
> 4. Ensure the client handles network failures gracefully (retry logic)."

