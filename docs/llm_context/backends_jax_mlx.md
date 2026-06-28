# JAX And MLX Backends

## Shared Contract

Both backends render readable Python modules with:

- backend imports (`jax.numpy` or `mlx.core` plus `numpy` helpers when needed)
- generated ONNX helper functions only for ops present in the graph
- `load_weights(path)`
- `forward(params, inputs)` returning a dict keyed by GraphIR outputs
- optional alias when `entrypoint != "forward"`

Both backend dispatch surfaces cover all semantic schemas at review base.

## JAX Codegen

`src/tnnx/codegen/jax_codegen.py` includes static metadata analysis for shape-like tensors, shared topological emission ordering, generated helper functions, and optional `forward_jit` support when static metadata parameters exist.

Key tests: JAX snapshots, JAX runtime parity tests, topology scheduler contract, bfloat16/FP8/instance norm/reshape/gather-slice tests.

## MLX Codegen

`src/tnnx/codegen/mlx_codegen.py` emits MLX helpers, prepackages convolution weights for layout, uses the shared topological emission ordering, and supports the same semantic dispatch set as JAX.

## Common Code

`src/tnnx/codegen/common.py` contains the shared topological emission order and the small shared attr parsers used by both backends. The old test-only `NameGenerator` helper was removed.

## Backend-Specific Limits

- MLX real FLUX work is explicitly behind JAX-first notes and not a stable current claim.
- MLX audio Whisper gates depend on local MLX and `ffmpeg`; the Whisper loader retries the observed Homebrew x265 dylib mismatch before failing.
- JAX warnings show x64 disabled by default for some paths; do not claim float64 parity unless tested.

## Generated-Code Acceptance

At minimum, generated code should parse/compile/import, have no unresolved placeholders, and pass the tiny JAX/MLX runtime smoke in `scripts/check_generated_code.py`. Runtime parity is required for retained behavior changes.
