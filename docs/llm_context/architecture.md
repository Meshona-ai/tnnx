# Architecture

## Scope

`tnnx` is a Python package and example suite for ONNX-based transpilation to readable JAX and MLX Python artifacts. The retained product surface is package code under `src/tnnx`, the `tnnx` CLI, and examples/tests that prove named model paths.

Removed or unsupported surfaces: FPGA, HLS, RTL, C/C++ native runtime/codegen, hardware toolchains, web servers, and web APIs.

## Package Layout

- `src/tnnx/api.py`: public orchestration via `transpile_onnx`.
- `src/tnnx/cli.py`: `tnnx transpile` argparse CLI.
- `src/tnnx/config.py`: `CompileConfig`, `ResourceBudget`, `ArtifactManifest`, and default pass tuple.
- `src/tnnx/ingest/`: ONNX loading, initializer extraction, dtype mapping, shape hints, and ONNX-to-semantic op map.
- `src/tnnx/ir/`: dataclasses for `TensorRef`, `OpNode`, `GraphIR`, schema validation, and JSON serde.
- `src/tnnx/passes/`: conservative pass surface: prune, normalize, shape propagation.
- `src/tnnx/codegen/`: JAX and MLX render/emit functions for generated modules.
- `src/tnnx/runtime/`: `.npz` weight save/load helpers.
- `examples/`: source-backed model demos and model-zoo smoke catalog.
- `tests/`: unit, integration, and snapshot gates.

## Flow

ONNX model -> `load_onnx_to_ir` -> `GraphIR` -> configured passes -> backend render/emit -> `graph_ir.json`, `compile_metadata.json`, `model_jax.py` or `model_mlx.py`, `weights.npz`.

## Ownership Boundaries

Repo-local owners from `AGENTS.md` apply:

- `tnnx-ingest-ir`: `src/tnnx/ingest`, `src/tnnx/ir`.
- `tnnx-graph-passes`: `src/tnnx/passes`.
- `tnnx-jax-codegen`: JAX backend and shared codegen when JAX-led.
- `tnnx-mlx-codegen`: MLX backend.
- `tnnx-examples-model-zoo`: `examples`.
- `tnnx-test-quality`: tests, validation gates, and this context pack.

## Current Architecture Summary

The codebase is intentionally direct: dataclasses, explicit operator dispatch, generated-source string assembly, and simple pass functions. There is no plugin framework and no registry beyond dictionaries such as `ONNX_TO_SEMANTIC` and `SEMANTIC_SCHEMAS`.

The main architectural risks are environment-heavy model paths and keeping public claims synchronized with tests. Scheduler drift, prune no-op behavior, and metadata-only config ambiguity were addressed in the execution branch.
