# Required FLUX Models

This file is the canonical model inventory for the standalone FLUX example. It exists so the
later MLX port can reuse the same model ledger instead of rediscovering the checkpoint surface.

Primary target:

- `black-forest-labs/FLUX.2-klein-4b-fp8`

Current real-loader target:

- `black-forest-labs/FLUX.2-klein-4B`

Other practical 4B-class candidates for a 16 GB Mac:

- `black-forest-labs/FLUX.2-klein-4B`
- `black-forest-labs/FLUX.2-klein-base-4B`
- `black-forest-labs/FLUX.2-klein-base-4b-nvfp4`

Expected snapshot subfolders:

- `transformer/`
- `vae/`
- `text_encoder/`

Milestone mapping:

- Milestone A (FP8 preflight): no real checkpoint required
- Milestone B (`vae_decoder` parity): `vae/`
- Milestone C (reduced transformer parity): synthetic transformer is sufficient, real
  `transformer/` becomes useful once checkpoint-specific sample inputs are wired
- Milestone D (reduced JAX image smoke): no real checkpoint required
- Milestone E (`text_encoder` parity): `text_encoder/`
- Milestone F (reduced prompt harness on the synthetic single encoder): no real checkpoint
  subfolder required
- Milestone G (full prompt-to-image): the supported real checkpoint path uses `transformer/`,
  `vae/`, and `text_encoder/`

Notes:

- This iteration intentionally uses a synthetic reduced transformer + VAE path as the always-on
  JAX smoke target.
- Real checkpoint-backed work stays opt-in and local-cache-only.
- The current code path can consume the diffusers-style `black-forest-labs/FLUX.2-klein-4B`
  layout directly. The FP8 repo is still important, but it is a single-file checkpoint and needs a
  separate loader path before it becomes the main real target.
- The reduced synthetic prompt path now also uses the same single-text-encoder topology as the
  real FLUX.2 target.
- To unblock the real checkpoint-backed path today, provide a full local snapshot with weights for
  `black-forest-labs/FLUX.2-klein-4B`, then point `TNNX_FLUX_SNAPSHOT` at that snapshot.
- See `examples/flux/snapshot_setup.md` for the full operator checklist.
