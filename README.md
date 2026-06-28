# tnnx

Minimal ONNX-based transpiler that emits JAX and MLX artifacts.

## Demo Docs

If you are preparing a walkthrough, start here:

- `docs/README.md`
- `docs/architecture.md`
- `docs/roadmap.md`

## Quick Start

Requires Python 3.14.

```bash
uv sync --dev
uv run tnnx transpile --onnx model.onnx --target jax --out generated/jax
uv run tnnx transpile --onnx model.onnx --target jax --out generated/jax --no-graph-ir
```

Targets:
- `jax`: emits `model_jax.py`
- `mlx`: emits `model_mlx.py`

All targets emit:
- `weights.npz`

By default, targets also emit:
- `graph_ir.json` (disable with `--no-graph-ir`)

## Export behavior note

The current PyTorch export path is example-input driven. When you export a model
to ONNX with the provided examples, `torch.onnx.export(...)` runs the model with
dummy or sample inputs to trace the graph.

For source-code transpilation, the exporter attempts
`torch.onnx.export(..., dynamo=True)` first and falls back to the legacy exporter
when needed.

This means:

- Exporting a large model can be expensive because the traced forward path is
  executed during export.
- Python-side control flow can be frozen to the path taken by those sample
  inputs. In practice, `if` / `else` branches that depend on Python values,
  shape-derived checks, or data-dependent conditions may be hardcoded into the
  exported ONNX graph based on the branch taken at export time.

Once an ONNX file exists, `tnnx` does not execute the original model again.
`tnnx` reads the ONNX graph and transpiles it into JAX or MLX artifacts.

## Real Model Example

NanoGPT-tiny (single-block) demo:
```bash
uv run python -m examples.model_nanogpt_tiny
uv run python -m examples.run_nanogpt_tiny_jax
```

Whisper-tiny-style (Conv1D front-end + encoder/decoder) MLX demo:
```bash
uv run python -m examples.model_whisper_tiny
uv run python -m examples.run_whisper_tiny_mlx
```

Real audio transcription through the transpiler (source Whisper -> ONNX -> generated MLX):
```bash
uv run python -m examples.whisper_audio.transpile_and_transcribe \
  --audio examples/terminator.mp3 \
  --output-dir examples/whisper_audio/out \
  --decode-tokens 64 \
  --max-new-tokens 48
```

Ordered source-backed model catalog (canonical upstream implementations, shared by family):
```bash
uv run python -m examples.model_zoo.list_models
uv run python -m examples.model_zoo.check_compatibility
uv run python -m examples.model_zoo.transpile_smoke --list
uv run python -m examples.model_zoo.transpile_smoke --target jax --model ResNet-18
```

Explicit prompt-observation demo (all steps under `examples/nanogpt/`):
```bash
uv run python -m examples.nanogpt.train_nanogpt_tiny \
  --prompt "hello jax" \
  --max-new-tokens 32
```
The script trains (or reloads) a tiny toy checkpoint before transpiling, so output is meaningful.

Pretrained GPT-2 inference demo:
```bash
uv run python -m examples.nanogpt.infer_gpt2_from_transformers_jax \
  --model-id sshleifer/tiny-gpt2 \
  --prompt "Hello from tnnx"
```

## Generated Python Contract

For `jax` and `mlx`, generated modules expose:
- `load_weights(path)`
- `forward(params, inputs)`

## Snapshot Tests

Codegen snapshot tests compare full generated source against checked-in files under
`tests/snapshots/expected/`.

Run snapshots:
```bash
uv run pytest -q tests/snapshots
```

Update snapshots intentionally:
```bash
UPDATE_SNAPSHOTS=1 uv run pytest -q tests/snapshots
```

## Support Matrices

- `docs/operators.md`: generated supported-operator matrix.
- `docs/model_support.md`: named model tiers, gates, assets, and blockers.
- `docs/testing.md`: local/CI gates and environment-gated lanes.
