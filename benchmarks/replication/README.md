# Replication Training Benchmarks

This directory contains replication training benchmarks for Mechanize-style training.

## Structure

Each benchmark includes:
- Source code repository (or synthetic repo)
- Test suite
- Task definition (what to replicate/fix)
- Configuration files

## Benchmarks

### project_a (Example)

A simple Python project with:
- `src/module.py`: Main module with functions
- `tests/test_module.py`: Test suite
- Task: Implement missing function `calculate_sum()`

## Usage

See `docs/replication_training.md` for detailed usage instructions.

## Adding New Benchmarks

1. Create a new directory under `benchmarks/replication/`
2. Add source code and tests
3. Create a `TASK.md` file describing the replication task
4. Add configuration to scenario registry (see `src/prime_rl/registry/`)

