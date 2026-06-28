# tnnx Demo Docs

This folder is the operator pack for a live `tnnx` walkthrough.

Use it when you need to explain the project quickly, show a reliable demo, and
connect the current implementation to the larger compiler thesis.

## One-Liner

`tnnx` is an open compiler pipeline that starts at the ONNX boundary, lowers a
model into a deterministic graph IR, runs normalization and shape-aware passes,
and emits readable artifacts for JAX and MLX targets.

## Demo Goal

In a short session, prove five things:

1. `tnnx` is a compiler pipeline, not a one-off conversion script.
2. The ONNX boundary gives one portable input surface for multiple targets.
3. Generated artifacts are inspectable and usable, not opaque blobs.
4. The project already handles real model families beyond toy layers.
5. There is a credible next step: deeper optimization and broader model coverage.

## Doc Map

- `demo-runbook.md`: exact commands, talk track, fallbacks, and timing.
- `architecture.md`: pipeline, artifacts, and design choices worth explaining.
- `model_support.md`: named model tiers, gates, assets, and blockers.
- `operators.md`: generated operator matrix from schema, ingest map, and backend dispatch.
- `roadmap.md`: optimization story for the next 12 months.
- `testing.md`: local gate, CI gate, pytest markers, and environment gates.

## What Exists Today

- ONNX ingest with deterministic weight extraction.
- Internal graph IR with schema validation.
- Conservative compiler passes: prune, normalize, shape propagation.
- Backend codegen for `jax` and `mlx`.
- Example and runtime lanes for NanoGPT, Whisper, GPT-2, Qwen 3.5, and FLUX.

## Recommended Demo Shape

- Show a small model live to avoid dependency or latency risk.
- Open the generated `graph_ir.json` and backend artifact immediately after.
- Use Qwen, Whisper, or FLUX as proof of scale, not as the live happy path.
- Close by explaining the optimization layer you want to build next.
