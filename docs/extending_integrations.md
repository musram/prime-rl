
To extend the `prime-rl` Integration Layer, you simply need to implement new adapters for specific environment frameworks or verifier protocols. The architecture is designed to be pluggable.

Here is the guide on how to extend it:

### 1. Add a New Environment Framework
Create a new directory in `src/prime_rl/integrations/` (e.g., `langgraph/` or `mcp/`).

**Steps:**
1.  Inherit from `EnvironmentAdapter` (defined in `src/prime_rl/core/environment.py`).
2.  Implement the required methods:
    *   `reset()`: Initialize the environment.
    *   `step(action)`: Execute an action and return `(observation, reward, done, info)`.
    *   **Crucial**: Map the framework-specific observation (e.g., LangGraph state) to our `UniversalRollout` format.

**Example (LangGraph):**
```python
from prime_rl.core.environment import EnvironmentAdapter

class LangGraphAdapter(EnvironmentAdapter):
    def __init__(self, graph):
        self.graph = graph

    def step(self, action):
        # ... convert action to LangGraph input ...
        output = self.graph.invoke(...)
        # ... convert output to UniversalRollout observation ...
        return obs, reward, done, info
```

### 2. Add a New Remote Verifier Protocol
If you need to support a new verification protocol (e.g., a specialized secure enclave verifier or a human-in-the-loop service), add it to `src/prime_rl/integrations/remote/`.

**Steps:**
1.  Inherit from `VerifierClient` (defined in `src/prime_rl/core/verifier.py`).
2.  Implement `verify(rollout)`.

### 3. Add a Data Source Integration
To support ingesting data from a new source (e.g., "Salesforce Logs" or "MCP Server Logs"), implement a **Data Converter**.

**Steps:**
1.  Create a script or module that reads the source format.
2.  Convert the data into the **Interaction Trace JSONL** standard (defined in `docs/product_spec.md`).
3.  Use the existing `JaxDataLoader` to train on it.

---

**Immediate Next Step**:
We are currently building the `BrowserGym` and `HttpVerifier` integrations as per `docs/task_integrations_layer.md`. Once those are done, you can follow the pattern above to add LangGraph or MCP support.

