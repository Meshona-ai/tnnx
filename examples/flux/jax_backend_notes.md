# FLUX Backend Notes

This page keeps only the current backend facts needed by future FLUX work. The old chronological
handoff log was consolidated because runnable tests now carry the detailed regression history.

## Implemented JAX Support

- ONNX float8 dtype ingest is recognized in `src/tnnx/ingest/dtypes.py`.
- JAX `CAST` handles supported ONNX float8 enum ids when the installed JAX exposes those dtypes.
- `RECIPROCAL`, `SQUEEZE`, and `INSTANCENORM` are mapped, shaped, emitted, and covered by focused
  JAX runtime parity tests.
- FLUX reduced/synthetic component tests cover `vae_decoder`, `transformer`, `text_encoder`, and a
  reduced prompt-to-image path.

## Validation

Run the focused FLUX JAX gate with:

```bash
uv run pytest -q \
  tests/integration/test_flux_vae_decoder_source.py \
  tests/integration/test_flux_vae_decoder_jax_parity.py \
  tests/integration/test_flux_transformer_source.py \
  tests/integration/test_flux_transformer_jax_parity.py \
  tests/integration/test_flux_text_encoder_source.py \
  tests/integration/test_flux_text_encoder_jax_parity.py \
  tests/integration/test_flux_jax_prompt_to_image_smoke.py
```

Checkpoint-backed tests require:

```bash
export TNNX_FLUX_SNAPSHOT=/abs/path/to/flux/snapshot
export RUN_FLUX_E2E=1
```

## Future MLX Work

Do not infer MLX FLUX support from JAX. A future MLX FLUX pass should reuse the same component
order and gates from `examples/flux/submodule_plan.md`, then add MLX-specific parity evidence.
