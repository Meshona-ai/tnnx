# Decisions

## Current Decisions

- Start at ONNX. Source framework export is an example concern; core transpilation starts from ONNX.
- Keep GraphIR explicit. Dataclasses and JSON serde are preferred over a speculative compiler framework.
- Keep passes conservative. Correctness and validation come before performance optimization.
- Emit readable Python for JAX and MLX. Generated code is an artifact users can inspect.
- Treat examples as proof lanes, not hidden core product code.
- Keep optional performance-only optimization out unless needed for correctness or retained named-model viability.

## Removed Scope

FPGA/HLS/RTL/C/C++ native runtime/codegen, hardware vendor toolchains, web servers, and web APIs are out of scope. The second-pass residue audit found no true tracked residue.

## Backend Parity

JAX and MLX should share semantic coverage and retained behavior gates. They do not need identical implementation, but backend-specific differences must be explicit and tested.

## Public API

The public API should stay small: `transpile_onnx`, config dataclasses, manifest, and CLI. Metadata-only public knobs need explicit docs or removal.

## Future Work Not Done Now

This audit does not implement product fixes, new operators, new model support, new backend support, or dependency changes. It creates the plan and context needed to do those safely later.
