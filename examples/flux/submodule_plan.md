# FLUX Submodule Plan

Active blocker ledger:

- Real `transformer` export now has checkpoint-specific sample input wiring, but backend support for
  the exported ONNX op surface is still expected to be incomplete.
- Checkpoint-backed `transformer` now has a snapshot-gated JAX transpile smoke lane, but that lane
  is still expected to `xfail` until the real unsupported-op set and remaining backend blockers are
  worked down.
- Current local asset status: the loader-free inspection path now confirms the local
  `black-forest-labs/FLUX.2-klein-4B` snapshot has config + weights for `vae_decoder`,
  `transformer`, and `text_encoder`. The current remaining real blocker is no longer missing
  assets; it is the transformer component load itself still being killed in this environment during
  the real checkpoint prep lane. The default `black-forest-labs/FLUX.2-klein-4b-fp8` target still
  needs separate single-file loader support before it can replace the current real path. The
  operator checklist for the supported real snapshot lives in `examples/flux/snapshot_setup.md`.
- The real checkpoint topology is fixed to `vae_decoder` + `transformer` + `text_encoder`.
- `examples/model_zoo` integration is deferred until the standalone JAX path is stable.

Exact export order:

1. `vae_decoder`
2. `transformer`
3. reduced transformer + VAE image path
4. `text_encoder`
5. full prompt-to-image JAX path

Real checkpoint order note:

- The checkpoint-prep report and the gated real E2E contract use the fixed real export order:
  `vae_decoder`, `transformer`, `text_encoder`.
- The checkpoint-prep CLI can now isolate one real lane with `--checkpoint-submodule`; use that
  for the active transformer blocker burn-down before widening back to a full sweep.
- The checkpoint-prep CLI now also has a cheaper proof path with
  `--checkpoint-graph-only --checkpoint-reduced-shapes`; use that first when the full checkpoint
  transformer export is too slow or too disk-heavy.

Complexity controls:

- One backend at a time: JAX only for this iteration
- One reduced fixture set shared across tests
- One source export test and one parity test per active submodule lane
- No duplicated blocker lists in test files
- Snapshot-gated setup now stays centralized in `tests/_support/flux.py`

Current implementation status:

- `vae_decoder`: active and covered by source export + JAX parity tests
- `vae_decoder` (checkpoint-backed, reduced graph-only smoke): active and now reaches `ready`
  against the real `black-forest-labs/FLUX.2-klein-4B` snapshot in the same reduced checkpoint
  prep flow
- `vae_decoder` (checkpoint-backed, weighted JAX transpile): active and now passes as a dedicated
  snapshot-gated integration test against the real `black-forest-labs/FLUX.2-klein-4B` snapshot
- `vae_decoder` (checkpoint-backed, weighted JAX runtime smoke): active and now passes on tiny
  real latents; this lane also forced the JAX backend to honor ONNX `Reshape` zero-copy semantics
- `transformer`: active reduced synthetic lane covered by source export + JAX parity tests
- `transformer` (checkpoint-backed): source export lane is wired behind `RUN_FLUX_E2E=1`; the
  checkpoint-backed source test is allowed to `xfail` on exporter instability or the unsupported-op
  list until backend support catches up. The reduced random-weight checkpoint-backed JAX
  transpile/runtime smoke tests now pass on reduced shapes, so this lane has a bounded real proof
- `transformer` (checkpoint-backed, reduced graph-only smoke): active as the preferred cheap
  blocker-burn-down lane when run with
  `--checkpoint-submodule transformer --checkpoint-graph-only --checkpoint-reduced-shapes`
  because that mode now uses a reduced config derived from the real transformer config with random
  weights instead of loading the full checkpoint weights
- `text_encoder` (checkpoint-backed, graph-only smoke): active as a cheap real export lane when
  run with `--checkpoint-submodule text_encoder --checkpoint-graph-only`, because that mode now
  uses a reduced config derived from the real text-encoder config with random weights instead of
  loading the full checkpoint weights
- `text_encoder` (checkpoint-backed, reduced JAX runtime smoke): active as an opt-in test on the
  same reduced random-weight config, so the real text-encoder lane now covers export, JAX
  transpilation, and a cheap runtime parity check without pulling full checkpoint weights
- reduced checkpoint bridge (`text_encoder` -> `transformer` -> `vae_decoder`): active as a
  snapshot-gated JAX test on the reduced real topology; it runs the PyTorch reference first, then
  checks the generated JAX path against that same reduced real bridge
- reduced checkpoint bridge example: active as `run_flux_jax_reduced_checkpoint_bridge_demo()`;
  writes a PyTorch reference PNG and a generated JAX PNG for the same reduced real bridge
- reduced image path: active JAX smoke path, emits a PNG
- synthetic FLUX.2 single-encoder proof lane: active; runs a tiny random-weight PyTorch
  reference first, then transpiles the same tiny `text_encoder` + `transformer` + `vae_decoder`
  path to JAX as a cheap workflow proof before paying real-checkpoint cost
- deterministic tiny-config proof lane: active; derives a tiny config from the real FLUX.2
  component configs, fills every floating-point weight with constants, runs the PyTorch reference
  first, then runs the generated JAX path on the same bounded topology; the PyTorch preflight is
  also exposed separately via `--use-tiny-config-torch`
- full prompt-to-image (reduced synthetic): active JAX smoke path using the single transpiled
  `text_encoder`, then bridging that output into the reduced transformer path
- `text_encoder`: snapshot-gated real source export test behind `RUN_FLUX_E2E=1`; parity stays on
  the tiny synthetic FLUX.2 lane to avoid heavyweight real-checkpoint runs
- full prompt-to-image (checkpoint-backed): still deferred behind `RUN_FLUX_E2E=1`
  but the gated reduced-checkpoint artifact contract now passes by running the real submodule prep
  flow with `checkpoint_reduced_shapes=True`; full real prompt-to-image assembly is still deferred
  surface before reporting the remaining assembly blocker

Future backend note:

- When MLX work begins later, reuse the exact same model ledger from `required_models.md` and the
  same submodule order. Do not branch into a separate FLUX process for MLX.
