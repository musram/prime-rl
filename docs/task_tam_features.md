# Implementation Task: Massive Orchestration & Marketplace Integrations

**Context**:
To fully realize the vision outlined in `TAM-doc.md`, PRIME-RL needs to scale beyond a single-node orchestrator and integrate with the broader ecosystem of data marketplaces and enterprise tools. We are currently missing:
1.  **Massive Orchestration**: Support for 10,000+ concurrent environments via distributed worker pools.
2.  **Enterprise Adapters**: Integrations for Epic (healthcare), Slack, and other SaaS tools.
3.  **Data Marketplace**: Connectors for Surge AI / Mercor to ingest labeled data and "Interaction Traces".

**Objective**:
Implement the `prime_rl.orchestrator.distributed` module and expand the `prime_rl.integrations` package with enterprise and marketplace adapters.

**Requirements**:

### 1. Distributed Orchestrator (`src/prime_rl/orchestrator/distributed/`)
*   **`WorkerPool` Interface**: Abstract base class for managing remote workers (e.g., `KubernetesWorkerPool`, `RayWorkerPool`).
*   **`RedisQueue` Backend**: Use Redis for a high-throughput task queue (pushing environment steps, popping observations) to decouple the Orchestrator loop from worker execution.
*   **Scaling Goal**: Must support >10k concurrent environment instances by sharding the `EnvGroup` across multiple nodes.

### 2. Enterprise Adapters (`src/prime_rl/integrations/enterprise/`)
*   **`EpicAdapter`**: A mock/stub adapter for HL7/FHIR based healthcare workflows (or a generic "Electronic Health Record" simulator wrapper).
*   **`SlackAdapter`**: An adapter that wraps the Slack API to allow an agent to read messages and post replies as actions, converting them to the `UniversalRollout` format.

### 3. Data Marketplace Connectors (`src/prime_rl/integrations/marketplace/`)
*   **`SurgeAIClient`**: A client to fetch completed labeling tasks from Surge AI and convert them into `InteractionTrace` JSONL format for Offline RL.
*   **`MercorClient`**: Similar adapter for ingesting "expert demonstration" logs from Mercor.

**Deliverables**:
*   Python modules for `distributed`, `enterprise`, and `marketplace`.
*   Unit tests for the new adapters.
*   A Docker-compose setup for the Redis-backed distributed orchestrator.

**Instruction to AI**:
> "Please execute the plan above to close the TAM gaps.
> 1. Implement `RedisWorkerPool` in `src/prime_rl/orchestrator/distributed/pool.py` using `redis-py`.
> 2. Implement `SlackAdapter` in `src/prime_rl/integrations/enterprise/slack.py` (using `slack_sdk`).
> 3. Implement `SurgeAIClient` in `src/prime_rl/integrations/marketplace/surge.py` that downloads CSVs and outputs `InteractionTrace` objects.
> 4. Ensure all new components are typed and documented."

