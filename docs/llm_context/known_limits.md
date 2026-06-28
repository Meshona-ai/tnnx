# Known Limits

## Baseline Failures

- Default `uv run pytest -q`: 15 failures before examples deps, mainly missing `onnxscript`/`diffusers` for FLUX paths plus host `ffmpeg` failures.
- After `uv sync --dev --group examples`: 2 failures remain, both Whisper real-audio MLX tests blocked by host `ffmpeg` missing x265 library.

## Unsupported Or Unvalidated Paths

- No FPGA/HLS/RTL/C/C++/native-runtime/web/API-server targets.
- Real FLUX checkpoint lanes require local snapshot setup and env gates.
- Full Qwen real JAX/MLX lanes require opt-in env gates and cached/downloaded assets.
- Model-zoo planned/experimental entries are not stable support claims.
- `RELU6` and `SILU` are semantic/codegen-supported but have no direct ONNX map entry at review base.

## Backend Limits

- MLX lacks the JAX topological scheduler path.
- MLX FLUX parity is not a stable current claim.
- Generated code import/compile gates should be expanded for both backends.

## Correctness Uncertainties

- ONNX shape-inference failures are swallowed.
- ONNX external-data trust boundary is not characterized.
- Public config fields `deterministic`, `emit_shape_asserts`, and `opset` are metadata-only or unused until task T05 resolves them.
- Prune pass is advertised but currently no-op.

## Packaging Limits

- Python is restricted to `>=3.14,<3.15`.
- sdist includes examples, research scripts, tests, and agent/process files; wheel is package-only.
