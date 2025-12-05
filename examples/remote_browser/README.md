# Remote Browser Demo

This example demonstrates Forward Deployment Architecture where a browser environment
runs locally on a laptop while training happens on a remote PRIME-RL cluster.

## Architecture

```
┌─────────────────┐         HTTP/gRPC          ┌──────────────────┐
│  Local Laptop   │ ─────────────────────────> │  PRIME-RL Cluster│
│                 │                             │                  │
│  Browser Env    │ <────────────────────────── │  Orchestrator    │
│  Worker Client  │      Policy Updates         │  Trainer (GPUs)  │
└─────────────────┘                             └──────────────────┘
```

## Setup

1. **Start the PRIME-RL Orchestrator API**:
   ```bash
   export PRIME_RL_API_KEY="your-secret-key"
   uvicorn prime_rl.orchestrator.api:app --host 0.0.0.0 --port 8000
   ```

2. **Run the remote browser worker**:
   ```bash
   python examples/remote_browser/remote_browser_worker.py \
       --server-url http://cluster.example.com:8000 \
       --api-key your-secret-key
   ```

## How It Works

1. Worker registers with the server (`POST /v1/workers/register`)
2. Worker runs browser episodes locally
3. Worker buffers traces and submits them to server (`POST /v1/traces`)
4. Worker polls for policy updates (`GET /v1/policy/latest`)
5. Server collects traces and trains the model
6. Updated policy is served to workers

## Configuration

- `PRIME_RL_API_KEY`: API key for authentication (set on server)
- `server_url`: Base URL of PRIME-RL Orchestrator API
- `trace_buffer_size`: Number of traces to buffer before sending
- `policy_poll_interval`: Interval to poll for policy updates (seconds)

