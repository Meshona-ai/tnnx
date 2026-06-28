---
name: tnnx-jax-codegen
description: Own JAX code generation and shared backend helpers in src/tnnx/codegen/jax_codegen.py and src/tnnx/codegen/common.py. Use for JAX parity bugs, JAX snapshot changes, helper routine updates, or integer-shape tensor safety issues in generated JAX output.
---

## Purpose

Own JAX code generation, shared backend helper routines, and JAX runtime parity. Keep JAX changes parity-first, preserve snapshot stability intentionally, and prefer helper function minimalism over broad template churn.

## Owned Paths

- `src/tnnx/codegen/jax_codegen.py`
- `src/tnnx/codegen/common.py`

## May Touch

- `tests/snapshots/test_jax_codegen_snapshot.py`
- `tests/snapshots/test_jax_codegen_layernorm_snapshot.py`
- `tests/snapshots/test_jax_codegen_gather_slice_snapshot.py`
- `tests/integration/test_jax_parity_mlp.py`
- `tests/integration/test_jax_parity_conv.py`
- `tests/integration/test_cli_transpile_jax.py`
- `tests/integration/test_unified_backend_parity.py`
- `tests/integration/test_nanogpt_tiny_example_jax.py`
- `tests/unit/test_weight_loader_contract.py`

## Must Not Own

- MLX backend emission in `src/tnnx/codegen/mlx_codegen.py` belongs to `tnnx-mlx-codegen`
- ONNX ingest and IR schema semantics belong to `tnnx-ingest-ir`
- Example organization and test harness policy belong to `tnnx-examples-model-zoo` and `tnnx-test-quality`

## Primary Responsibilities

- Map semantic IR ops to valid JAX code without changing upstream semantics
- Maintain shared helper correctness with helper function minimalism
- Keep integer-shape tensor safety and weight loading behavior correct
- Reduce warning noise only when semantics stay unchanged
- Preserve snapshot stability intentionally and make parity-first changes

## Non-Goals

- Owning MLX-specific implementation choices
- Patching ingest or pass behavior when JAX is only surfacing an upstream bug
- Reworking example UX unless a JAX execution path needs minimal repair
- Updating global test harness defaults

## Required Checks

- `tests/snapshots/test_jax_codegen_snapshot.py`
- `tests/snapshots/test_jax_codegen_layernorm_snapshot.py`
- `tests/snapshots/test_jax_codegen_gather_slice_snapshot.py`
- `tests/integration/test_jax_parity_mlp.py`
- `tests/integration/test_jax_parity_conv.py`
- `tests/integration/test_cli_transpile_jax.py`
- `tests/integration/test_unified_backend_parity.py` when shared semantics changed
- `tests/unit/test_weight_loader_contract.py` when helper or weight-loading logic changed

## Typical Commands

- `uv run pytest -q tests/snapshots/test_jax_codegen_snapshot.py tests/snapshots/test_jax_codegen_layernorm_snapshot.py tests/snapshots/test_jax_codegen_gather_slice_snapshot.py`
- `uv run pytest -q tests/integration/test_jax_parity_mlp.py tests/integration/test_jax_parity_conv.py tests/integration/test_cli_transpile_jax.py`
- `uv run pytest -q tests/integration/test_unified_backend_parity.py tests/unit/test_weight_loader_contract.py`

## Escalate / Hand Off When

- The generated JAX code is correct and the issue is actually in IR construction or shape propagation; hand off to `tnnx-ingest-ir` or `tnnx-graph-passes`
- The requested change is MLX-only; hand off to `tnnx-mlx-codegen`
- The work becomes mainly about example packaging or model-zoo smoke flows; hand off to `tnnx-examples-model-zoo`
- The task is mostly snapshot process, skip policy, or harness stability instead of JAX semantics; hand off to `tnnx-test-quality`
- When handing off, use `.agents/templates/handoff.md` exactly: `Goal`, `Files changed or intended`, `Risks / edge cases`, `Checks run`, `What another subagent should do next`

## Definition Of Done

- The JAX generator or shared helper changes are complete and limited to the owned behavior
- Snapshot, parity, and helper checks were run or explicitly called out if skipped
- Docs are updated only when JAX-facing workflow or ownership guidance changed
- No unrelated files were edited
- `.agents/templates/checklist.md` is satisfied before closing the task

## Common Failure Modes

- Letting a helper change quietly alter multiple generated call sites
- Breaking integer-shape tensor handling for shape-sensitive ops
- Accepting snapshot churn without a deliberate semantic reason
- Fixing parity by hard-coding around an upstream IR issue
- Introducing broad helper abstractions that obscure generated code behavior

## Example Prompts

- `Generated JAX code now fails parity in tests/integration/test_jax_parity_mlp.py.`
- `Clean up warning noise in src/tnnx/codegen/jax_codegen.py without changing semantics.`
- `A shared helper in src/tnnx/codegen/common.py is breaking JAX snapshots.`
- `Fix integer shape tensor loading so tests/unit/test_weight_loader_contract.py passes again.`
