# Decisions

## Current Decisions

- Start at ONNX. Source framework export is an example concern; core transpilation starts from ONNX.
- Keep GraphIR explicit. Dataclasses and JSON serde are preferred over a speculative compiler framework.
- Keep passes conservative. Correctness and validation come before performance optimization.
- Emit readable Python for JAX and MLX. Generated code is an artifact users can inspect.
- Treat examples as proof lanes, not hidden core product code.
- Keep optional performance-only optimization out unless needed for correctness or retained named-model viability.
- Keep Python support restricted to 3.14 for this branch; CI and package metadata both say `>=3.14,<3.15`.
- Keep `deterministic`, `emit_shape_asserts`, and `opset` in `CompileConfig` as metadata-only fields for this release.

## Removed Scope

Retired low-level and web/server surfaces are out of scope. The reusable residue guard found no true tracked residue.

## Backend Parity

JAX and MLX share semantic coverage and topological emission ordering. They do not need identical implementation, but backend-specific differences must be explicit and tested.

## Public API

The public API should stay small: `transpile_onnx`, config dataclasses, manifest, and CLI. Metadata-only public knobs must stay explicitly documented.

## Future Work Not Done Now

This branch does not add new model/backend support. It improves validation reliability, public claims, invariants, packaging, and context-pack maintenance around the retained behavior.
