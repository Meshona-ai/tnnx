# FLUX (JAX First)

This example area is the standalone staging ground for FLUX-family support before any
`examples/model_zoo` integration.

Current scope:

- JAX only
- local cached snapshots only
- reduced synthetic transformer + VAE smoke path for end-to-end validation
- synthetic tiny FLUX.2-style single-text-encoder proof path with random weights and tensors
- deterministic tiny FLUX.2 config proof path with constant weights and shrunk real config shapes
- real FLUX snapshot loading hooks for `vae_decoder`, `transformer`, and `text_encoder`
- checkpoint-backed `transformer` export + transpile smoke lane behind the snapshot gate

Environment contract:

- `TNNX_FLUX_SNAPSHOT=/abs/path/to/local/flux/snapshot`
- `RUN_FLUX_E2E=1` for expensive checkpoint-backed tests
- Detailed operator steps for supplying a real snapshot live in `examples/flux/snapshot_setup.md`.
- `HUGGING_FACE_TOKEN` can live in the shell environment or in `.env`; the checkpoint download path
  now reads both.

Fast smoke path:

```bash
uv run python -m examples.run_flux_jax --out /tmp/flux_jax_smoke
```

This exports reduced FLUX-like submodules, transpiles both to JAX, and writes a PNG.
The same JAX demo path now also accepts injected `prompt_embeddings` + `pooled_prompt`, which is
the handoff seam for the future real text-encoder path.
You can also provide those arrays as a fixture file with:
`uv run python -m examples.run_flux_jax --out /tmp/flux_jax_smoke --prompt-npz /tmp/prompt_fixture.npz`
The reduced full prompt-to-image path is now active too:
`uv run python -m examples.run_flux_jax --out /tmp/flux_jax_smoke --use-text-encoders`
That prompt-to-image lane now follows the single-text-encoder FLUX.2 shape and can also take
token fixtures:
`uv run python -m examples.run_flux_jax --out /tmp/flux_jax_smoke --use-text-encoders --token-ids-npz /tmp/token_fixture.npz`
That JAX path now runs a standalone PyTorch prompt-to-image reference first and records the
reference image before executing the generated JAX path.
For the fastest topology-faithful proof, there is also a synthetic tiny FLUX.2 lane exposed as
both as Python entrypoints and from the CLI:
- `run_flux_pytorch_dummy_flux2_demo(...)`
- `run_flux_jax_dummy_flux2_demo(...)`
- `uv run python -m examples.run_flux_jax --out /tmp/flux_dummy_flux2 --use-dummy-flux2`
That lane keeps a single `text_encoder` like the real FLUX.2 topology, but uses tiny random
weights and tiny random inputs so the PyTorch reference and generated JAX path stay cheap.
There is now also a deterministic tiny-config lane derived from the real FLUX.2 config shapes,
but with every expensive dimension capped so it stays in a safe 16 GB envelope:
- `run_flux_pytorch_tiny_config_e2e_demo(...)`
- `run_flux_jax_tiny_config_e2e_demo(...)`
- `uv run python -m examples.run_flux_jax --out /tmp/flux_tiny_config_torch --use-tiny-config-torch`
- `uv run python -m examples.run_flux_jax --out /tmp/flux_tiny_config --use-tiny-config`
That lane uses constant weights plus zero-valued token/latent inputs, runs the PyTorch reference
first, then runs the generated JAX path on the same tiny topology and compares the final image.
The checkpoint-backed prompt-to-image phase now also has an explicit gated contract test in
`tests/integration/test_flux_prompt_to_image_jax_e2e.py`; it will skip without local assets and
`RUN_FLUX_E2E=1`, and it will `xfail` with the current real blocker once enabled. That gate is
now graph-contract-only: it exports real ONNX graphs without weights and checks the unsupported-op
surface instead of running a heavyweight multi-submodule transpile sweep.
There is now also a reusable checkpoint prep command:
`uv run python -m examples.run_flux_jax --out /tmp/flux_jax_real --prepare-checkpoint-artifacts`
It exports every real FLUX submodule it can, and in the default mode it also transpiles the
supported lanes to JAX before it writes
`flux_checkpoint_jax_report.json` with ready, blocked, and missing counts.
That report now also records whether each submodule used a reduced random-weight config.
There is also a loader-free asset inspection command when full model loading is still too heavy:
`uv run python -m examples.run_flux_jax --out /tmp/flux_jax_real --inspect-checkpoint-assets`
That reads the snapshot layout plus config metadata without calling `from_pretrained`.
It no longer crashes when the target snapshot is missing; it writes a report with the missing
reason instead. It now defaults to the real-loader target `black-forest-labs/FLUX.2-klein-4B`.
It also runs the existing PyTorch submodule forward first for every available real lane before any
ONNX export or JAX transpile work starts.
You can also target a different local cache key with
`--model-id black-forest-labs/FLUX.2-klein-base-4B`. The real checkpoint topology is fixed to
`vae_decoder`, `transformer`, and `text_encoder`.
For the current blocker-burn-down loop, you can narrow that report to one real lane:
`uv run python -m examples.run_flux_jax --out /tmp/flux_jax_real --prepare-checkpoint-artifacts --checkpoint-submodule transformer`
That keeps the real checkpoint iteration linear instead of sweeping every available submodule.
For a much cheaper real-checkpoint smoke pass, add:
`--checkpoint-graph-only --checkpoint-reduced-shapes`
That skips serializing checkpoint weights into the ONNX artifact. For the heavy real lanes, that
mode also instantiates reduced configs derived from the checkpoint metadata with random weights
instead of loading the full checkpoint weights. The transformer lane also uses tiny sample shapes
in that mode, which is useful for proving the export/op-inventory path before paying the full
checkpoint load cost.
There is also a download command now:
`uv run python -m examples.run_flux_jax --download-checkpoint-assets`

To unblock the real checkpoint-backed path, the next operator step is:

- Provide a full local snapshot with weights.
- For the current diffusers-style real loader, prefer `black-forest-labs/FLUX.2-klein-4B`.
- The `black-forest-labs/FLUX.2-klein-4b-fp8` repo is reachable, but it is a single-file
  checkpoint and needs separate loader support before it can be used as the main real path.
- Then set `TNNX_FLUX_SNAPSHOT` to that snapshot directory.
- Then run the checkpoint-prep command above.

See `examples/flux/snapshot_setup.md` for the full checklist.

Real snapshot path:

- `resolve_flux_snapshot()` checks `TNNX_FLUX_SNAPSHOT` first.
- If the env var is unset, it falls back to the local Hugging Face cache for
  `black-forest-labs/FLUX.2-klein-4B`.
- `export_flux_submodule_onnx()` can export the real FLUX.2 `vae_decoder`, `transformer`, and
  `text_encoder` path when the snapshot and `diffusers` are available.
- To enable the real snapshot export path, install the example-only dependency set:
  `uv sync --dev --group examples`
  That group now includes `accelerate`, which avoids the slower low-memory fallback during real
  FLUX.2 component loading.
- The real `text_encoder` lane now has a snapshot-gated source export test behind `RUN_FLUX_E2E=1`.
  That source export test, plus the snapshot-gated JAX transpile/runtime smoke test for the same
  lane, now use a reduced config with random weights rather than the full checkpoint weights. JAX
  parity for text-encoder behavior stays on the tiny synthetic FLUX.2 lane so test cost stays
  bounded.
- The real `transformer` lane now has checkpoint-specific sample input wiring and an opt-in source
  export test plus snapshot-gated JAX transpile/runtime smoke tests. In reduced mode those tests
  now use a reduced config with random weights plus reduced sequence lengths, so the real
  transformer lane stays in a bounded cost envelope.
- The real `vae_decoder` lane now also has snapshot-gated weighted JAX transpile and runtime smoke
  tests, and that weighted path currently passes against the local
  `black-forest-labs/FLUX.2-klein-4B` snapshot. That runtime proof required a JAX backend fix for
  ONNX `Reshape` zero-copy semantics (`0` means “copy the corresponding input dimension”).
- The transformer export path now uses `dynamo=True`, which is why `onnxscript` belongs in the
  `examples` dependency group.
- The gated checkpoint contract now uses `prepare_flux_jax_checkpoint_artifacts(
  checkpoint_reduced_shapes=True)`, which means the heavy submodules (`transformer` and
  `text_encoder`) use reduced random-weight configs during export and JAX transpile while the real
  `vae_decoder` still uses checkpoint weights. That reduced checkpoint artifact gate now passes on
  the local `black-forest-labs/FLUX.2-klein-4B` snapshot.
- The reduced transformer config now caps `in_channels` / `out_channels` at `32`, which matches
  the real FLUX.2 VAE latent-channel count and allows a bounded three-submodule bridge test:
  reduced `text_encoder` -> reduced `transformer` -> real `vae_decoder`.
- That bridge now has a snapshot-gated JAX test which runs the PyTorch reference path first, then
  runs the generated JAX modules on the same reduced real topology and compares the final image.
- The same reduced real bridge is now exposed as `run_flux_jax_reduced_checkpoint_bridge_demo()`,
  which writes both the PyTorch reference PNG and the generated JAX PNG from that reduced real
  topology.
- The CLI exposes the same path via `uv run python -m examples.run_flux_jax --use-reduced-
  checkpoint-bridge`, which reuses the current `TNNX_FLUX_SNAPSHOT` / local cache resolution.
- The JAX backend now topologically schedules nodes before emission, which matters for the reduced
  real transformer bridge because some exported graphs place `Split` producers after their
  consumers in the raw IR order.

The reduced synthetic transformer remains the always-on JAX validation path. The checkpoint-backed
transformer lane is now wired and has a bounded reduced-runtime proof path; full real prompt-to-
image assembly is still a separate follow-on task.
