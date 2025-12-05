# Extending Integrations Implementation Summary

This document summarizes the implementation of extension examples for the PRIME-RL Integrations Layer, demonstrating how to add new environment frameworks, verifier protocols, and data source integrations.

## Overview

Following the guide in `docs/extending_integrations.md`, we have implemented example extensions that demonstrate the pluggable architecture of PRIME-RL's integrations layer.

## Implemented Extensions

### 1. LangGraphAdapter

**Location**: `src/prime_rl/integrations/langgraph/adapter.py`

**Purpose**: Adapter for LangGraph state machines as RL environments.

**Features**:
- Wraps LangGraph `StateGraph` instances
- Maps LangGraph state (messages, agent state) to string format
- Converts actions (messages) to string format
- Tracks trajectory for `UniversalRollout` conversion
- Supports custom reward functions
- Handles graph invocation and state transitions

**Usage Example**:
```python
from langgraph.graph import StateGraph
from prime_rl.integrations import LangGraphAdapter

graph = StateGraph(...)
adapter = LangGraphAdapter(
    graph,
    reward_function=lambda state, action: 1.0 if state.get("success") else 0.0
)

obs, info = adapter.reset()
obs, reward, terminated, truncated, info = adapter.step("user_message")
rollout = adapter.get_rollout()
```

**Key Implementation Details**:
- Compiles graph if needed
- Tracks state transitions through graph nodes
- Converts LangGraph state (typically dict with messages) to string observations
- Supports termination detection based on state signals (done, error, success)

**Dependencies**: `langgraph`

### 2. MCPAdapter

**Location**: `src/prime_rl/integrations/mcp/adapter.py`

**Purpose**: Adapter for MCP (Model Context Protocol) servers as RL environments.

**Features**:
- Wraps MCP server connections
- Maps MCP tool calls and responses to `UniversalRollout` format
- Supports async MCP operations
- Tracks tool call sequences as trajectories
- Handles tool listing and invocation

**Usage Example**:
```python
from mcp import StdioServerParameters
from prime_rl.integrations import MCPAdapter

server_params = StdioServerParameters(...)
adapter = MCPAdapter(server_params)

obs, info = adapter.reset()
obs, reward, terminated, truncated, info = adapter.step({
    "tool": "read_file",
    "arguments": {"path": "/tmp/file.txt"}
})
rollout = adapter.get_rollout()
```

**Key Implementation Details**:
- Initializes MCP session on reset
- Lists available tools from server
- Converts tool calls to actions and tool results to observations
- Supports both sync and async MCP operations (with async context handling)

**Dependencies**: `mcp`

### 3. GrpcVerifierClient

**Location**: `src/prime_rl/integrations/remote/grpc_verifier.py`

**Purpose**: gRPC-based verifier client for remote reward/grading APIs.

**Features**:
- Sends verification requests via gRPC
- Supports both sync and async verification
- Batch verification for parallel processing
- Retry logic with exponential backoff
- Proper error handling for gRPC errors

**Usage Example**:
```python
from prime_rl.integrations import GrpcVerifierClient
# Note: Requires generated gRPC stubs from .proto file

client = GrpcVerifierClient(
    endpoint="localhost:50051",
    service_stub=VerificationServiceStub,
)

result = await client.verify_async(
    observation="user clicked button",
    action="click(button_id)",
)
```

**Key Implementation Details**:
- Stub implementation (requires generated gRPC stubs from .proto)
- Handles gRPC channel creation and management
- Converts gRPC errors to `VerifierError` hierarchy
- Supports both insecure and secure channels (via channel_options)

**Dependencies**: `grpcio`

**Note**: This is a template/stub implementation. In production, you would:
1. Define a `.proto` file for your verification service
2. Generate Python stubs using `protoc`
3. Implement the `_build_request()` and `_parse_response()` methods using generated message classes

### 4. Data Converters

**Location**: `src/prime_rl/integrations/data_converters/`

**Purpose**: Convert external data sources to Interaction Trace JSONL format.

#### Base DataConverter

**Location**: `src/prime_rl/integrations/data_converters/base.py`

Abstract base class defining the interface for data converters:
- `convert()`: Convert source data to `InteractionTrace` objects
- `convert_to_jsonl()`: Convert and write directly to JSONL file
- `validate_trace()`: Validate trace objects

#### SalesforceLogConverter

**Location**: `src/prime_rl/integrations/data_converters/salesforce.py`

**Features**:
- Converts Salesforce API logs, user interaction logs, or workflow logs
- Supports JSON and CSV input formats
- Extracts trace IDs, timestamps, actions, and observations
- Maps Salesforce records to `InteractionTrace` format

**Usage Example**:
```python
from prime_rl.integrations import SalesforceLogConverter
from pathlib import Path

converter = SalesforceLogConverter(environment_id="salesforce-v1")
converter.convert_to_jsonl(
    source_path=Path("salesforce_logs.json"),
    output_path=Path("traces.jsonl"),
)
```

**Key Implementation Details**:
- Extracts trace ID from record ID field
- Converts Salesforce timestamps
- Maps Salesforce actions/operations to trace steps
- Uses success indicators for rewards

#### MCPLogConverter

**Location**: `src/prime_rl/integrations/data_converters/mcp_logs.py`

**Features**:
- Converts MCP server logs (tool calls, resource access)
- Groups logs by request_id/session_id
- Creates multi-step traces from tool call sequences
- Maps tool calls to actions and results to observations

**Usage Example**:
```python
from prime_rl.integrations import MCPLogConverter
from pathlib import Path

converter = MCPLogConverter(environment_id="mcp-server-v1")
converter.convert_to_jsonl(
    source_path=Path("mcp_logs.json"),
    output_path=Path("traces.jsonl"),
)
```

**Key Implementation Details**:
- Groups logs by trace_id (request_id or session_id)
- Creates one trace per request/session
- Maps tool calls to steps with rewards based on success/error
- Preserves tool call sequences as multi-step trajectories

## Architecture Patterns

All extensions follow consistent patterns:

### Environment Adapters

1. **Inherit from `EnvironmentAdapter`**
2. **Implement required methods**:
   - `reset()`: Initialize environment
   - `step()`: Execute action
   - `render()`: Render state
   - `close()`: Cleanup
   - `observation_space` / `action_space`: Space properties
3. **Track trajectory** for `UniversalRollout` conversion
4. **Provide `get_rollout()`** method (optional but recommended)

### Verifier Clients

1. **Inherit from `VerifierClient`**
2. **Implement required methods**:
   - `verify()`: Synchronous verification
   - `verify_async()`: Asynchronous verification
   - `verify_batch()`: Batch verification
   - `close()`: Cleanup
3. **Use retry logic** from `prime_rl.orchestrator.retry`
4. **Handle errors** using `VerifierError` hierarchy

### Data Converters

1. **Inherit from `DataConverter`**
2. **Implement required methods**:
   - `convert()`: Convert to `InteractionTrace` objects
   - `convert_to_jsonl()`: Convert and write to JSONL
3. **Validate traces** using base class `validate_trace()` method
4. **Handle errors gracefully** (skip invalid records)

## Usage in Training Pipeline

Once data is converted to Interaction Trace JSONL format, it can be used directly with existing training infrastructure:

```python
from prime_rl.integrations import SalesforceLogConverter
from prime_rl.backends.jax.data_loader import JaxDataLoader
from prime_rl.train import train_offline_jax
from prime_rl.core.config import UnifiedConfig, load_config

# Convert external data
converter = SalesforceLogConverter()
converter.convert_to_jsonl(
    source_path=Path("salesforce_logs.json"),
    output_path=Path("data/traces.jsonl"),
)

# Train with converted data
config = load_config(Path("configs/offline_example.toml"))
# Update config.dataset.path to point to converted traces
train_offline_jax(config)
```

## Testing

Basic test structure for extensions:

```python
# tests/unit/integrations/test_langgraph_adapter.py
@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not available")
def test_langgraph_adapter_reset():
    # Create mock graph
    graph = Mock()
    adapter = LangGraphAdapter(graph)
    obs, info = adapter.reset()
    assert obs is not None
```

## Future Extensions

The architecture supports easy addition of:

1. **More Environment Frameworks**:
   - SeleniumAdapter
   - PlaywrightAdapter
   - Custom simulator adapters

2. **More Verifier Protocols**:
   - WebSocketVerifierClient
   - GraphQLVerifierClient
   - Custom protocol clients

3. **More Data Sources**:
   - Database log converters
   - API log converters
   - Custom format converters

## Summary

The extension examples demonstrate:

✅ **Pluggable Architecture**: Easy to add new adapters and converters
✅ **Consistent Interfaces**: All extensions follow the same patterns
✅ **Type Safety**: Fully type-hinted implementations
✅ **Error Handling**: Robust error handling and validation
✅ **Documentation**: Comprehensive docstrings and examples

These extensions make PRIME-RL ready to integrate with:
- LangGraph workflows
- MCP servers and tools
- gRPC verification services
- External data sources (Salesforce, MCP logs, etc.)

All code follows the same patterns established in the base integrations (`BrowserGymAdapter`, `HttpVerifierClient`), making it easy for users to extend PRIME-RL with their own integrations.

