# Demo Runbook

Use the small-model path for live demos. Keep real checkpoint paths as evidence, not the live
critical path.

## Setup

```bash
uv sync --dev
uv run tnnx --help
uv run tnnx transpile --help
```

## Fast Compiler Demo

```bash
uv run python -m examples.model_mlp --target jax --output-dir /tmp/tnnx-demo/mlp
sed -n '1,80p' /tmp/tnnx-demo/mlp/generated_tiny_mlp_jax/model_jax.py
sed -n '1,80p' /tmp/tnnx-demo/mlp/generated_tiny_mlp_jax/graph_ir.json
```

This shows ONNX export, GraphIR emission, JAX source generation, and weights in one quick path.

## Backend Parity Demo

```bash
uv run pytest -q tests/integration/test_unified_backend_parity.py
```

This proves the same supported IR paths run through JAX and MLX.

## Model Evidence

```bash
uv run python -m examples.model_zoo.transpile_smoke --target jax --model ResNet-18
uv run pytest -q tests/integration/test_whisper_audio_real_mlx.py
```

Use `model_support.md` for the current model support tiers and blockers.

## Fallbacks

- If a live export is slow, switch to `uv run pytest -q tests/unit/test_api_manifest.py`.
- If a checkpoint path is missing assets, show `model_support.md` instead of downloading live.
- Do not claim FLUX/Qwen real checkpoint support without running their environment-gated commands.
