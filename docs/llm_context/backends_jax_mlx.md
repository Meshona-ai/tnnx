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

`src/tnnx/codegen/jax_codegen.py` includes static metadata analysis for shape-like tensors, an internal topological scheduler, generated helper functions, and optional `forward_jit` support when static metadata parameters exist.

Key tests: JAX snapshots, JAX runtime parity tests, topology scheduler contract, bfloat16/FP8/instance norm/reshape/gather-slice tests.

## MLX Codegen

`src/tnnx/codegen/mlx_codegen.py` emits MLX helpers, prepackages convolution weights for layout, and supports the same semantic dispatch set as JAX.

Known parity gap: MLX currently iterates `ir.nodes` directly while JAX uses `_scheduled_nodes(ir)`. Add MLX out-of-order topology characterization before changing code.

## Common Code

`src/tnnx/codegen/common.py` exports `NameGenerator` and `order_nodes_for_emission`, but current backend emitters do not use them. Treat this as a verification/shrink candidate, not proven dead code.

## Backend-Specific Limits

- MLX real FLUX work is explicitly behind JAX-first notes and not a stable current claim.
- MLX audio Whisper gates depend on local MLX and working `ffmpeg`.
- JAX warnings show x64 disabled by default for some paths; do not claim float64 parity unless tested.

## Generated-Code Acceptance

At minimum, generated code should parse/compile/import and have no unresolved placeholders. Runtime parity is required for retained behavior changes.
