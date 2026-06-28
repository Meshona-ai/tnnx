# FLUX Snapshot Setup

This is the operator checklist for the real checkpoint-backed FLUX path. It covers the steps that
must happen outside the repo before the real JAX checkpoint flow can progress.

Preferred target:

- Provide a full local snapshot with weights for `black-forest-labs/FLUX.2-klein-4B`.

Alternate supported target:

- If that exact snapshot is not available, you can point the checkpoint-prep command at another
  complete FLUX.2 local snapshot with `--model-id`, but it still must contain the same required
  weights.

Important format note:

- `black-forest-labs/FLUX.2-klein-4B` uses the diffusers-style folder layout that the current
  real loader can consume.
- `black-forest-labs/FLUX.2-klein-4b-fp8` is currently a single-file checkpoint repo. It is a
  valid upstream asset, but it needs separate single-file loader support before it can replace the
  diffusers-style real path in this repo.

What "full snapshot with weights" means:

- Use the snapshot directory itself, not just the model cache root.
- For the default `black-forest-labs/FLUX.2-klein-4B` target, the snapshot should contain these
  subfolders:
  - `transformer/`
  - `vae/`
  - `text_encoder/`
- The subfolders must include actual weight files, not config-only metadata:
  - `transformer/` and `vae/`: `diffusion_pytorch_model.safetensors` or
    `diffusion_pytorch_model.bin`
  - `text_encoder/`: either `model.safetensors` /
    `pytorch_model.bin`, or a sharded `model.safetensors.index.json` plus `model-*.safetensors`

How to provide it to this repo:

1. Obtain the full snapshot outside the repo.
2. Point `TNNX_FLUX_SNAPSHOT` at the absolute snapshot directory:
   `export TNNX_FLUX_SNAPSHOT=/abs/path/to/flux/snapshot`
3. Install the example-only dependencies if needed:
   `uv sync --dev --group examples`
4. If you want the repo to download the expected asset set for you, store
   `HUGGING_FACE_TOKEN` in the shell environment or in `.env`, then run:
   `uv run python -m examples.run_flux_jax --download-checkpoint-assets`

How to validate the snapshot:

1. Start with the loader-free asset inspection when the full model load is still too expensive:
   `uv run python -m examples.run_flux_jax --out /tmp/flux_jax_real --inspect-checkpoint-assets`
   That validates the snapshot folder layout and config metadata without loading the checkpoint.
2. Run the checkpoint-prep report:
   `uv run python -m examples.run_flux_jax --out /tmp/flux_jax_real --prepare-checkpoint-artifacts`
   This now defaults to the real-loader target `black-forest-labs/FLUX.2-klein-4B`.
3. For iterative blocker work, narrow the report to the active lane first:
   `uv run python -m examples.run_flux_jax --out /tmp/flux_jax_real --prepare-checkpoint-artifacts --checkpoint-submodule transformer`
   That keeps the checkpoint burn-down loop focused on the real transformer before sweeping the
   rest of the topology.
   For the fastest checkpoint smoke run, add:
   `--checkpoint-graph-only --checkpoint-reduced-shapes`
   That skips serializing model weights into the ONNX file. For the heavy real lanes, it also
   builds reduced configs derived from the real checkpoint metadata with random weights instead of
   loading the full checkpoint weights. For the transformer lane specifically, it also reduces the
   sample shapes, so you can prove the real export path and inspect unsupported ops before doing
   the full heavyweight load.
4. If you want to use a different cached model id:
   `uv run python -m examples.run_flux_jax --out /tmp/flux_jax_real --prepare-checkpoint-artifacts --model-id black-forest-labs/FLUX.2-klein-base-4B`
   The real checkpoint topology stays fixed to `vae_decoder`, `transformer`, and `text_encoder`.
5. Once the report shows real submodules moving past `missing`, run the gated contract test:
   `RUN_FLUX_E2E=1 uv run pytest -q tests/integration/test_flux_prompt_to_image_jax_e2e.py`

Expected outcome:

- If the snapshot is complete, the report will move from `missing` to either `ready` or `blocked`.
- `blocked` means the next remaining work is a real export/transpile backend issue.
- `missing` means the snapshot still does not contain the required weights.
