# Replication Training

This document explains Mechanize-style replication training in PRIME-RL, where agents learn to reproduce existing software functionality by passing test suites.

## Overview

Replication training is a form of RL where:
- The agent edits code in a repository
- Success is measured by passing tests (e.g., `pytest`)
- The reward signal comes from test results

This enables training agents to:
- Implement missing functions
- Fix bugs
- Refactor code
- Meet specifications

## Architecture

### Components

1. **Coding Environment**: Provides code editing capabilities
   - Can use `verifiers.ToolEnv` or `verifiers.PythonEnv`
   - Agent actions are code edits (file path + content)

2. **PytestVerifierClient**: Runs tests and converts results to rewards
   - Executes `pytest` (or configurable command)
   - Parses test output
   - Maps pass/fail to scalar rewards

3. **Replication Benchmark**: Repository + task definition
   - Source code with missing/faulty implementations
   - Test suite defining correctness
   - Task description

## Usage

### Running Replication Training

```bash
prime-rl train --config configs/replication_project_a.toml --mode online
```

### Evaluating a Trained Model

```bash
prime-rl eval --config configs/replication_project_a.toml --checkpoint outputs/replication_project_a/checkpoints/checkpoint-500
```

## Benchmark Structure

Each replication benchmark includes:

```
benchmarks/replication/project_a/
├── src/
│   └── module.py          # Source code (with TODOs)
├── tests/
│   └── test_module.py     # Test suite
├── TASK.md                 # Task description
└── pyproject.toml          # Project configuration
```

## Creating a New Benchmark

### Step 1: Create Repository Structure

```bash
mkdir -p benchmarks/replication/your_project/{src,tests}
```

### Step 2: Add Source Code

Create `src/module.py` with incomplete implementations:

```python
def target_function(x: int) -> int:
    # TODO: Implement this function
    # Expected: return x * 2
    pass
```

### Step 3: Add Tests

Create `tests/test_module.py`:

```python
import pytest
from src.module import target_function

def test_target_function():
    assert target_function(2) == 4
    assert target_function(0) == 0
    assert target_function(-1) == -2
```

### Step 4: Define Task

Create `TASK.md`:

```markdown
# Replication Task: Target Function

## Objective
Implement `target_function()` to pass all tests.

## Success Criteria
All tests in `tests/test_module.py` must pass.
```

### Step 5: Create Configuration

Create `configs/replication_your_project.toml`:

```toml
[backend]
type = "torch"
mode = "online"

[algorithm]
name = "grpo"

[model]
name = "meta-llama/Llama-3.1-8B-Instruct"

[environment]
type = "python_repo"
repo_path = "benchmarks/replication/your_project"

[verifier]
type = "pytest"
command = "pytest -q"
timeout = 60

[output]
output_dir = "outputs/replication_your_project"
```

### Step 6: Register Scenario

Add to `src/prime_rl/registry/registry.py`:

```python
self.register(ScenarioMetadata(
    id="replication_your_project",
    name="Replication Training - Your Project",
    description="Implement target_function",
    config_path=Path("configs/replication_your_project.toml"),
    category="replication",
    tags=["replication", "coding", "pytest"],
    env_type="python_repo",
    verifier_type="pytest",
))
```

## PytestVerifierClient

The `PytestVerifierClient`:

1. **Copies Repository**: Creates a temporary copy
2. **Applies Code Changes**: Writes agent's code edits
3. **Runs Tests**: Executes `pytest` (or configurable command)
4. **Parses Results**: Extracts pass/fail counts
5. **Computes Reward**: Maps test results to [0, 1] reward

**Reward Computation:**
- All tests pass → reward = 1.0
- Some tests pass → reward = (passed / total)
- Timeout/error → reward = 0.0

## Example: Project A

The included `project_a` benchmark:

- **Task**: Implement `calculate_sum()` function
- **Tests**: 4 test cases covering edge cases
- **Success**: All tests pass

**Running:**
```bash
cd benchmarks/replication/project_a
pytest -q  # Should fail initially (function not implemented)
```

## Advanced Usage

### Custom Test Commands

You can use different test runners:

```toml
[verifier]
type = "pytest"
command = "python -m unittest discover"  # Use unittest
timeout = 120
```

### Multi-File Tasks

For tasks spanning multiple files:

```python
# Agent action format
{
    "file": "src/module.py",
    "content": "..."
}
```

The verifier applies changes to the specified file.

### Coverage-Based Rewards

Extend `PytestVerifierClient` to include coverage:

```python
command = "pytest --cov=src --cov-report=term"
# Parse coverage from output
```

## Best Practices

1. **Start Simple**: Begin with single-function tasks
2. **Clear Tests**: Tests should clearly define correctness
3. **Incremental Complexity**: Add complexity gradually
4. **Realistic Tasks**: Use real-world coding patterns
5. **Documentation**: Document task requirements clearly

## Limitations

- Currently supports Python projects only
- Requires `pytest` (or compatible test runner)
- Test execution happens synchronously
- No support for multi-repo tasks yet

## Future Enhancements

- Support for other languages (JavaScript, Go, etc.)
- Multi-file editing in single action
- Coverage-based rewards
- Multi-repo benchmarks
- Integration with CI/CD systems

