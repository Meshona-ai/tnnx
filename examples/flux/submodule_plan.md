# FLUX Submodule Status

This page is the current FLUX lane ledger. Historical burn-down notes were removed; use tests and
`examples/flux/snapshot_setup.md` as the source of truth for runnable gates.

## Active Gates

| Lane | Status | Gate |
| --- | --- | --- |
| `vae_decoder` synthetic JAX | ready | `uv run pytest -q tests/integration/test_flux_vae_decoder_source.py tests/integration/test_flux_vae_decoder_jax_parity.py` |
| `transformer` synthetic/reduced JAX | ready | `uv run pytest -q tests/integration/test_flux_transformer_source.py tests/integration/test_flux_transformer_jax_parity.py` |
| `text_encoder` synthetic/reduced JAX | ready | `uv run pytest -q tests/integration/test_flux_text_encoder_source.py tests/integration/test_flux_text_encoder_jax_parity.py` |
| reduced prompt-to-image JAX | ready | `uv run pytest -q tests/integration/test_flux_jax_prompt_to_image_smoke.py` |
| checkpoint-backed FLUX | environment-gated | `RUN_FLUX_E2E=1` plus `TNNX_FLUX_SNAPSHOT=/abs/path/to/snapshot` |

## Snapshot Requirements

The real checkpoint topology is fixed to `vae_decoder`, `transformer`, and `text_encoder`.
Checkpoint-backed tests use `tests/_support/flux.py` for centralized gate handling. A valid local
snapshot must contain component configs and weights; see `examples/flux/snapshot_setup.md`.

## Current Boundaries

- JAX is the active FLUX backend.
- MLX FLUX is not claimed.
- Full checkpoint prompt-to-image remains gated by local assets and `RUN_FLUX_E2E=1`.
- The default suite covers reduced or synthetic FLUX paths only.
