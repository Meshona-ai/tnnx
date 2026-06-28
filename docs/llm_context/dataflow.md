# Dataflow

## ONNX Input Assumptions

`load_onnx_to_ir` accepts an ONNX path and calls `onnx.load`. It optionally runs ONNX shape inference. It assumes graph initializers and value info are enough to build tensor refs. Shape inference errors are currently swallowed and must be planned as a diagnostic improvement.

## Graph Construction

- Initializers are extracted with `onnx.numpy_helper.to_array` and sorted by name.
- Graph inputs exclude initializers.
- Graph outputs come from ONNX graph outputs.
- `Constant` nodes with a `value` attribute are folded into initializers.
- Non-constant nodes are mapped through `ONNX_TO_SEMANTIC`; unknown ops raise `UnsupportedOpError`.
- Optional ONNX input slots are filtered, with `input_slots` recorded for GroupQueryAttention, Pad, and Resize.

## Constants And External Data

Weights are saved to `weights.npz`. Small initializer values up to 64 scalars are passed into shape propagation as runtime values. External-data safety has not been characterized; it is a planned security task.

## Shape And Dtype Handling

Dtypes are read from ONNX value info or initializer arrays. Missing tensor info defaults to `float32` and empty shape. Shape propagation covers all current semantic ops, but several ops depend on runtime constants such as reshape shapes, slice bounds, pad values, split sizes, and resize scales/sizes.

## Operator Handling

Operator support has four surfaces that must stay aligned:

1. ONNX spelling in `src/tnnx/ingest/op_map.py`.
2. Semantic schema in `src/tnnx/ir/schema.py`.
3. Shape propagation in `src/tnnx/passes/shape_prop.py`.
4. JAX and MLX emitters in `src/tnnx/codegen/`.

## Generated Artifact Contract

Every transpile emits `weights.npz`, `compile_metadata.json`, a backend module, and optionally `graph_ir.json`. Generated modules expose `load_weights(path)` and `forward(params, inputs)`. JAX may additionally emit `forward_jit` when static metadata exists.

## Validation Checkpoints

- Ingest: unsupported op tests, initializer extraction, IR roundtrip, IR validation.
- Passes: shape propagation unit tests and runtime parity integration tests.
- Codegen: snapshots, parse/compile/import smoke, JAX/MLX numerical parity.
- CLI/API: manifest, metadata, generated artifacts, help text.
- Models: model-zoo smoke, Qwen, Whisper, FLUX, NanoGPT/GPT-2 lanes by tier.
