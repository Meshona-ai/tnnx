# Extension Guide

## Add An Operator

Use the smallest path that proves behavior: mapping, schema, shape propagation, backend emission, one focused runtime parity test, snapshots only when generated source intentionally changes, and context-pack updates.

## Add A Model Path

1. Add or update source-backed example code under `examples/`.
2. Define required assets and whether they are tracked, generated, cached, or downloaded.
3. Add a cheap always-on gate first.
4. Put expensive/network/snapshot paths behind explicit env vars or markers.
5. Update `model_zoo.md`, `validation.md`, and machine indexes.

## Modify Backend Codegen

1. Add a tiny GraphIR or ONNX characterization test.
2. Update JAX and MLX separately unless the bug is truly shared.
3. Keep generated helpers explicit and boring.
4. Run backend snapshots plus runtime parity for the touched op.
5. Update `backends_jax_mlx.md`, `operators.md`, and indexes.

## Common Mistakes

- Do not claim support from docs or catalog entries alone.
- Do not weaken tests to hide environmental failures.
- Do not add a plugin framework for one new operator or model.
- Do not centralize backend code until snapshots/parity make drift visible.
- Do not delete scripts/examples from grep alone; verify dynamic imports, docs, CLI paths, tests, and user-facing commands.
