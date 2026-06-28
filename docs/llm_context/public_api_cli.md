# Public API And CLI

## Top-Level Python Exports

`src/tnnx/__init__.py` exports:

- `ArtifactManifest`
- `CompileConfig`
- `DEFAULT_PASSES`
- `ResourceBudget`
- `transpile_onnx`

Version: `0.1.0`.

## Console Entry Point

`pyproject.toml` defines `tnnx = "tnnx.cli:main"`.

## CLI Surface

Command: `tnnx transpile`

Required flags:

- `--onnx <path>`
- `--target {jax,mlx}`
- `--out <dir>`

Optional flags:

- `--weights <filename>` default `weights.npz`
- `--entry <name>` default `forward`
- `--graph-ir` / `--no-graph-ir`, default enabled
- `--passes prune,normalize,shape_prop` or `none`
- `--target-hardware <tag>`
- `--preferred-dtype <dtype>`
- `--memory-budget-mb <int>`
- `--latency-priority balanced|low_latency|low_memory|auditability`
- `--resource-note <text>`

## Config Keys

`CompileConfig` fields: `deterministic`, `infer_shapes`, `emit_shape_asserts`, `emit_graph_ir`, `opset`, `entrypoint`, `weights_filename`, `enabled_passes`, `resource_budget`.

Known uncertainty: `deterministic`, `emit_shape_asserts`, and `opset` are currently recorded in metadata but do not clearly alter behavior. Treat as `VERIFY` until task T05 resolves the public contract.

## Stable Behavior

- `transpile_onnx` writes artifacts and returns `ArtifactManifest`.
- Unsupported target raises `ValueError`.
- Unsupported compile pass raises `ValueError` or argparse type error.
- Generated modules expose `load_weights` and `forward`.

## Internal/Unstable Behavior

- Private helpers such as `_runtime_values_from_weights` are tested but not top-level exports.
- Backend helper functions inside generated source are not public API.
- Example module APIs are demo-facing and often environment-dependent.

## Removed Behavior That Must Not Return

No FPGA/HLS/RTL/C/C++ native runtime/codegen, hardware vendor toolchain, web server, or web API CLI surface should be reintroduced during cleanup.
