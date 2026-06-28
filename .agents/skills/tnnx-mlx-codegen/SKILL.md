---
name: tnnx-mlx-codegen
description: Own MLX code generation in src/tnnx/codegen/mlx_codegen.py. Use for MLX parity bugs, MLX snapshot changes, layout handling issues, or MLX-specific numerical tolerance and runtime constraints.
---

## Purpose

Own MLX code generation and MLX runtime parity. Keep layout handling correct for ONNX exports, preserve parity symmetry with JAX when semantics match, and respect MLX-specific numerical tolerance expectations and runtime constraints.

## Owned Paths

- `src/tnnx/codegen/mlx_codegen.py`

## May Touch

- `src/tnnx/codegen/common.py` only when shared helper semantics must stay aligned with JAX
- `tests/snapshots/test_mlx_codegen_snapshot.py`
- `tests/snapshots/test_mlx_codegen_layernorm_snapshot.py`
- `tests/snapshots/test_mlx_codegen_gather_slice_snapshot.py`
- `tests/integration/test_mlx_parity_mlp.py`
- `tests/integration/test_mlx_parity_conv.py`
- `tests/integration/test_cli_transpile_mlx.py`
- `tests/integration/test_unified_backend_parity.py`
- `tests/integration/test_whisper_tiny_example_mlx.py`
- `tests/integration/test_whisper_audio_real_mlx.py`
- `tests/unit/test_mlx_conv_padding_validation.py`

## Must Not Own

- JAX backend emission and shared-helper primary ownership belong to `tnnx-jax-codegen`
- ONNX ingest and pass semantics belong to `tnnx-ingest-ir` and `tnnx-graph-passes`
- Example organization and suite policy belong to `tnnx-examples-model-zoo` and `tnnx-test-quality`

## Primary Responsibilities

- Map semantic IR ops to valid MLX code
- Keep layout handling correct for ONNX exports and convolution-heavy graphs
- Maintain parity symmetry with JAX when semantics should match
- Respect MLX-specific runtime constraints and numerical tolerance expectations
- Own MLX-specific runtime parity and generated source regressions

## Non-Goals

- Owning JAX backend implementation choices
- Fixing upstream semantic IR issues in the emitter layer
- Reorganizing example UX unless a minimal MLX execution-path change is required
- Changing broad test harness policy

## Required Checks

- `tests/snapshots/test_mlx_codegen_snapshot.py`
- `tests/snapshots/test_mlx_codegen_layernorm_snapshot.py`
- `tests/snapshots/test_mlx_codegen_gather_slice_snapshot.py`
- `tests/integration/test_mlx_parity_mlp.py`
- `tests/integration/test_mlx_parity_conv.py`
- `tests/integration/test_cli_transpile_mlx.py`
- `tests/integration/test_unified_backend_parity.py` when shared semantics changed
- `tests/integration/test_whisper_tiny_example_mlx.py` and `tests/integration/test_whisper_audio_real_mlx.py` when Whisper or audio execution paths are touched
- `tests/unit/test_mlx_conv_padding_validation.py` when layout-sensitive logic changed

## Typical Commands

- `uv run pytest -q tests/snapshots/test_mlx_codegen_snapshot.py tests/snapshots/test_mlx_codegen_layernorm_snapshot.py tests/snapshots/test_mlx_codegen_gather_slice_snapshot.py`
- `uv run pytest -q tests/integration/test_mlx_parity_mlp.py tests/integration/test_mlx_parity_conv.py tests/integration/test_cli_transpile_mlx.py`
- `uv run pytest -q tests/integration/test_unified_backend_parity.py tests/unit/test_mlx_conv_padding_validation.py`
- `RUN_MLX_E2E=1 uv run pytest -q tests/integration/test_whisper_audio_real_mlx.py tests/integration/test_whisper_audio_transpile_source.py`

## Escalate / Hand Off When

- The issue comes from wrong IR semantics or incorrect shape propagation rather than MLX emission; hand off to `tnnx-ingest-ir` or `tnnx-graph-passes`
- The change is really a shared-helper primary decision or a JAX-only regression; bring in `tnnx-jax-codegen` as reviewer or hand off
- The work is primarily about example orchestration, model-zoo loaders, or runtime environment setup; hand off to `tnnx-examples-model-zoo`
- The task is mostly about flaky MLX tests, skip policy, or snapshot process; hand off to `tnnx-test-quality`
- When handing off, use `.agents/templates/handoff.md` exactly: `Goal`, `Files changed or intended`, `Risks / edge cases`, `Checks run`, `What another subagent should do next`

## Definition Of Done

- The MLX generator change is complete and limited to the owned behavior
- Snapshot, parity, and MLX-specific checks were run or explicitly called out if skipped
- Docs are updated only when MLX-facing workflow or ownership guidance changed
- No unrelated files were edited
- `.agents/templates/checklist.md` is satisfied before closing the task

## Common Failure Modes

- Breaking layout handling for ONNX-exported conv graphs
- Letting MLX diverge from JAX where semantics should match
- Treating numerical tolerance drift as acceptable without proving it is MLX-specific
- Hiding an upstream IR bug with backend-specific special cases
- Forgetting to exercise Whisper/audio flows after touching MLX runtime-sensitive code

## Example Prompts

- `Generated MLX code regressed in tests/integration/test_mlx_parity_conv.py.`
- `Fix layout handling in src/tnnx/codegen/mlx_codegen.py for an ONNX-exported conv model.`
- `Whisper MLX output broke after a codegen change and needs a minimal fix.`
- `Align MLX output with JAX for a parity mismatch without changing IR semantics.`
