# Examples

This folder now follows one artifact contract across the shipped examples:

- Most examples need optional runtimes such as `torch`, `jax`, or `mlx`.
  Run `uv sync --dev` once before using the examples.
- Use `uv run python -m ...` from the repo root.
- Pass `--output-dir <DIR>` to control where generated artifacts are written.
- Model export examples write:
  - `<DIR>/<model>.onnx`
  - `<DIR>/generated_<model>_<target>/model_<target>.py` for `jax` or `mlx`
  - `<DIR>/generated_<model>_<target>/weights.npz`
  - `<DIR>/generated_<model>_<target>/graph_ir.json` by default (opt-out via `--no-graph-ir` on CLI transpile)

If you do not pass `--output-dir`, each example uses its documented default directory.

## Important export note

PyTorch -> ONNX export in these examples is not a pure shape-only conversion.
The current exporter path calls `torch.onnx.export(...)` with sample inputs, so it
executes the model code once during export.

That has two practical consequences:

- Export cost scales with the model path exercised by the sample inputs. Large
  models can be expensive to export even before transpilation starts.
- Python-side conditionals taken during export can be specialized into the
  exported graph. If your model has `if` / `else` branches based on Python
  values, tensor shapes, or data-dependent checks, the branch chosen by the
  sample inputs may be the one that gets baked into the ONNX graph.

After the ONNX file exists, `tnnx` only reads and transpiles that graph. The
ONNX -> JAX / MLX step does not execute the model again.

## Core model export examples

These examples export a small PyTorch model to ONNX and immediately transpile it.
All of them accept `--output-dir` and `--target {jax,mlx}`.

```bash
uv run python -m examples.model_mlp --output-dir /tmp/examples/mlp
uv run python -m examples.model_conv --output-dir /tmp/examples/conv
uv run python -m examples.model_residual --output-dir /tmp/examples/residual
uv run python -m examples.model_nanogpt_tiny --output-dir /tmp/examples/nanogpt
uv run python -m examples.model_whisper_tiny --output-dir /tmp/examples/whisper
```

Useful variants:

```bash
uv run python -m examples.model_whisper_tiny --target mlx --output-dir /tmp/examples/whisper-mlx
```

## End-to-end runtime demos

These examples export ONNX, transpile, then execute the generated backend code.

NanoGPT tiny through generated JAX:

```bash
uv run python -m examples.run_nanogpt_tiny_jax --output-dir /tmp/examples/nanogpt-jax
```

Whisper tiny through generated MLX:

```bash
uv run python -m examples.run_whisper_tiny_mlx --output-dir /tmp/examples/whisper-mlx
```

Toy NanoGPT training + generated JAX demo:

```bash
uv run python -m examples.nanogpt.train_nanogpt_tiny \
  --prompt "hello jax" \
  --max-new-tokens 32 \
  --output-dir /tmp/examples/nanogpt-demo
```

Pretrained GPT-2 inference through generated JAX:

```bash
uv run python -m examples.nanogpt.infer_gpt2_from_transformers_jax \
  --model-id sshleifer/tiny-gpt2 \
  --prompt "Hello from tnnx" \
  --output-dir /tmp/examples/gpt2-jax
```

Pretrained Qwen3.5-0.8B inference through generated JAX:

```bash
uv run python -m examples.qwen.infer_qwen3_5_from_transformers_jax \
  --prompt "Write one short sentence about compilers." \
  --output-dir /tmp/examples/qwen3_5_jax
```

Pretrained Qwen3.5-0.8B inference through generated MLX:

```bash
uv run python -m examples.qwen.infer_qwen3_5_from_transformers_mlx \
  --prompt "Write one short sentence about compilers." \
  --output-dir /tmp/examples/qwen3_5_mlx
```

Real `openai/whisper-tiny` speech-to-text through generated MLX:

```bash
uv run python -m examples.whisper_audio.transpile_and_transcribe \
  --audio examples/terminator.mp3 \
  --decode-tokens 64 \
  --max-new-tokens 48 \
  --output-dir /tmp/examples/whisper-audio
```

## Model zoo

The model-zoo examples stay behind one shared smoke runner instead of one file per model.
It also accepts `--output-dir`.

```bash
uv run python -m examples.model_zoo.transpile_smoke --list
uv run python -m examples.model_zoo.transpile_smoke \
  --target jax \
  --model ResNet-18 \
  --output-dir /tmp/examples/model-zoo
```

Artifacts land under the directory you pass, with one ONNX file plus one
`generated_<target>_<slug>/` folder per job.

See `../docs/model_support.md` for model tiers, required assets, and exact validation gates.

## Viewing transpiled code

For any generated example, inspect these files:

- `graph_ir.json`: normalized graph IR after ONNX ingest and graph passes (default artifact, optional).
- `model_jax.py` or `model_mlx.py`: the generated backend source.
- `weights.npz`: parameter bundle loaded by the generated runtime.

Typical inspection commands:

```bash
sed -n '1,200p' /tmp/examples/mlp/generated_tiny_mlp_jax/model_jax.py
sed -n '1,120p' /tmp/examples/mlp/generated_tiny_mlp_jax/graph_ir.json
```

## Layout

- `examples/model_*.py`: small self-contained export + transpile examples.
- `examples/run_*`: export + transpile + runtime parity demos.
- `examples/nanogpt/`: toy training demo plus pretrained GPT-2 inference demos.
- `examples/qwen/`: standalone Qwen3.5 JAX/MLX export + runtime generation demos.
- `examples/whisper_audio/`: source-backed Whisper transcription example.
- `examples/model_zoo/`: shared smoke runner for larger upstream-backed model families.
