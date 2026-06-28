# Known Limits

## Validation Limits

- Default `uv run pytest -q` passes after T01/T02.
- Real Whisper audio MLX E2E runs by default in tests. The loader contains a Homebrew-specific fallback for the observed `ffmpeg` 8.0.1_4 to x265 4.1/4.2 dylib mismatch; if neither matching x265 keg nor a repaired `ffmpeg` exists, audio loading still fails clearly.

## Unsupported Or Unvalidated Paths

- FLUX synthetic/default lanes require `accelerate`, `diffusers`, and `onnxscript`, which are now in the `dev` group. Real FLUX checkpoint lanes still require local snapshot setup and env gates.
- Full Qwen real JAX/MLX lanes require opt-in env gates and cached/downloaded assets.
- Model-zoo planned/experimental entries are not stable support claims.
- `RELU6` and `SILU` are semantic/codegen-supported but have no direct ONNX map entry at review base.

## Backend Limits

- MLX FLUX parity is not a stable current claim.
- Generated JAX/MLX import/compile/runtime smoke is covered by `scripts/check_generated_code.py`; broader named-model parity still depends on model-specific gates.

## Correctness Uncertainties

- Public config fields `deterministic`, `emit_shape_asserts`, and `opset` are intentionally metadata-only in this release.
- Unknown ONNX ops fail before IR pruning; prune only acts on supported GraphIR nodes.

## Packaging Limits

- Python is restricted to `>=3.14,<3.15`.
- Wheel and sdist package contents are checked by `scripts/check_package_contents.py`.
