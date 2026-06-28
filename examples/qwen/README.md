# Qwen3.5 JAX / MLX Examples

These standalone examples use two lanes:

- The always-on synthetic test lane exports a tiny forced-`full_attention`
  Qwen3.5 text model from Transformers, transpiles it to JAX/MLX, and checks
  prompt-window parity.
- The real checkpoint lane uses the prebuilt ONNX-community
  `onnx-community/Qwen3.5-0.8B-ONNX` `fp16` weights, specializes the merged
  decoder to decode-only (`sequence_length == 1`), transpiles that graph to
  JAX/MLX, and then generates decoded text through the generated runtime module.

Run the real checkpoint JAX path:

```bash
uv run python -m examples.qwen.infer_qwen3_5_from_transformers_jax \
  --prompt "Write one short sentence about compilers." \
  --output-dir /tmp/examples/qwen3_5_jax
```

Run the real checkpoint MLX path:

```bash
uv run python -m examples.qwen.infer_qwen3_5_from_transformers_mlx \
  --prompt "Write one short sentence about compilers." \
  --output-dir /tmp/examples/qwen3_5_mlx
```

If the ONNX-community snapshot is not cached locally, the example downloads the
required ONNX and tokenizer assets from Hugging Face on the first run. Use
`--local-files-only` to force offline loading.

The integration test suite also uses a tiny forced-`full_attention` synthetic
config to keep the always-on path deterministic and fast while still exercising
the generated JAX runtime.
