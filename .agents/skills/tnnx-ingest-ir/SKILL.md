---
name: tnnx-ingest-ir
description: Own ONNX ingest, semantic mapping, GraphIR schema, and IR serialization integrity for src/tnnx/ingest and src/tnnx/ir. Use for ONNX op support, initializer extraction, optional input slot handling, schema mismatches, or deterministic IR naming regressions.
---

## Purpose

Own ONNX ingest, semantic mapping, IR schema evolution, and IR serialization integrity without drifting backend behavior or test policy. Preserve semantic fidelity from ONNX inputs into `GraphIR`, including optional input slots, initializer extraction, schema compatibility, and deterministic IR naming.

## Owned Paths

- `src/tnnx/ingest/`
- `src/tnnx/ir/`

## May Touch

- `src/tnnx/api.py` only when ingest outputs or shape/runtime handoff require minimal wiring
- `tests/unit/test_op_mapping.py`
- `tests/unit/test_initializer_extraction.py`
- `tests/unit/test_ir_validation.py`
- `tests/unit/test_ir_roundtrip.py`
- `tests/unit/test_name_stability.py`
- `tests/integration/test_onnx_to_ir_mlp.py`
- `tests/integration/test_onnx_to_ir_layernorm.py`
- `tests/integration/test_onnx_to_ir_gather_slice.py`

## Must Not Own

- Graph normalization and shape propagation in `src/tnnx/passes/` belong to `tnnx-graph-passes`
- Backend code emission in `src/tnnx/codegen/` belongs to `tnnx-jax-codegen` and `tnnx-mlx-codegen`
- Example and model-zoo behavior in `examples/` belongs to `tnnx-examples-model-zoo`
- Test harness policy, snapshot discipline, and pytest defaults belong to `tnnx-test-quality`

## Primary Responsibilities

- Add or adjust ONNX op mappings in a way that preserves semantic mapping into `GraphIR`
- Parse ONNX attributes, initializers, and dtype metadata safely
- Preserve optional input slots instead of collapsing missing positions
- Maintain schema compatibility, validation, and round-trip stability
- Protect deterministic IR naming, tensor metadata, and serialization integrity

## Non-Goals

- Rewriting pass logic to compensate for missing ingest metadata
- Changing backend-specific code generation templates
- Reorganizing examples, smoke workflows, or model-zoo reporting
- Altering global test harness policy beyond the narrow tests needed for ingest and IR work

## Required Checks

- `tests/unit/test_op_mapping.py`
- `tests/unit/test_initializer_extraction.py`
- `tests/unit/test_ir_validation.py`
- `tests/unit/test_ir_roundtrip.py`
- `tests/unit/test_name_stability.py`
- `tests/integration/test_onnx_to_ir_mlp.py`
- `tests/integration/test_onnx_to_ir_layernorm.py`
- `tests/integration/test_onnx_to_ir_gather_slice.py`
- Affected snapshot families in `tests/snapshots/` when IR formatting changes propagate into generated artifacts

## Typical Commands

- `uv run pytest -q tests/unit/test_op_mapping.py tests/unit/test_initializer_extraction.py tests/unit/test_ir_validation.py tests/unit/test_ir_roundtrip.py tests/unit/test_name_stability.py`
- `uv run pytest -q tests/integration/test_onnx_to_ir_mlp.py tests/integration/test_onnx_to_ir_layernorm.py tests/integration/test_onnx_to_ir_gather_slice.py`
- `UPDATE_SNAPSHOTS=1 uv run pytest -q tests/snapshots/test_jax_codegen_snapshot.py tests/snapshots/test_mlx_codegen_snapshot.py`

## Escalate / Hand Off When

- A pass needs new runtime metadata or shape annotations beyond a minimal ingest wiring change; hand off to `tnnx-graph-passes`
- A backend consumes valid IR but emits incorrect target code; hand off to the relevant backend codegen owner
- The change becomes mostly about example execution, model-zoo loaders, or smoke reporting; hand off to `tnnx-examples-model-zoo`
- The task is mainly about flaky tests, environment gates, or snapshot policy rather than ingest behavior; hand off to `tnnx-test-quality`
- When handing off, use `.agents/templates/handoff.md` exactly: `Goal`, `Files changed or intended`, `Risks / edge cases`, `Checks run`, `What another subagent should do next`

## Definition Of Done

- The required ingest or IR code changes are complete and scoped to the owned behavior
- The relevant unit, integration, and snapshot checks were run or explicitly called out if skipped
- Docs are updated only when ingest-facing workflow or ownership guidance changed
- No unrelated files were edited
- `.agents/templates/checklist.md` is satisfied before closing the task

## Common Failure Modes

- Dropping ONNX optional input slots and shifting argument positions
- Misreading initializer tensors, especially shape-driving constants
- Accepting schema changes that break `GraphIR` round-trip stability
- Introducing nondeterministic names that churn snapshots and emitted artifacts
- Fixing ingest symptoms in backend code instead of preserving semantics in the IR

## Example Prompts

- `Add support for an ONNX op in src/tnnx/ingest/op_map.py and keep GraphIR stable.`
- `Fix initializer extraction for a shape tensor that is breaking tests/unit/test_initializer_extraction.py.`
- `The reader is dropping an optional ONNX input slot before serialization.`
- `An IR schema mismatch started failing tests/unit/test_ir_validation.py after a new attribute was added.`
