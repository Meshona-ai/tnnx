# JAX Backend Notes For Future MLX Port

This file is the handoff log for every JAX-side backend change made during the first FLUX
iteration. The goal is to make the later MLX port mechanical instead of requiring a repo search.

## Core backend changes already made

### 1. ONNX float8 dtype ingest support

File:

- `src/tnnx/ingest/dtypes.py`

What changed:

- Added support for:
  - `TensorProto.FLOAT8E4M3FN -> "float8_e4m3fn"`
  - `TensorProto.FLOAT8E4M3FNUZ -> "float8_e4m3fnuz"`
  - `TensorProto.FLOAT8E5M2 -> "float8_e5m2"`
  - `TensorProto.FLOAT8E5M2FNUZ -> "float8_e5m2fnuz"`
  - `TensorProto.FLOAT8E8M0 -> "float8_e8m0fnu"`

MLX follow-up:

- Add the same ONNX enum coverage in the MLX path. Even if MLX cannot represent these natively,
  ingest must still recognize them so the graph can be lowered deterministically.

### 2. JAX cast map support for float8 ONNX enums

File:

- `src/tnnx/codegen/jax_codegen.py`

What changed:

- Extended the generated `_ONNX_DTYPE_MAP` in the JAX backend:
  - `17 -> jnp.float8_e4m3fn`
  - `18 -> jnp.float8_e4m3fnuz`
  - `19 -> jnp.float8_e5m2`
  - `20 -> jnp.float8_e5m2fnuz`
  - `24 -> jnp.float8_e8m0fnu`

MLX follow-up:

- Mirror the ONNX enum coverage in `src/tnnx/codegen/mlx_codegen.py`.
- Do not assume MLX has native float8 dtypes.
- The expected MLX strategy is an explicit fallback cast policy, likely to `float16` or `float32`,
  while preserving the ONNX cast decision in a documented and testable way.

### 3. Real FLUX transformer blocker ops: `Reciprocal` and `Squeeze`

Files:

- `src/tnnx/ingest/op_map.py`
- `src/tnnx/ir/schema.py`
- `src/tnnx/passes/shape_prop.py`
- `src/tnnx/codegen/jax_codegen.py`

What changed:

- Added ONNX ingest mappings:
  - `Reciprocal -> RECIPROCAL`
  - `Squeeze -> SQUEEZE`
- Added semantic schemas:
  - `RECIPROCAL` as a unary same-shape op
  - `SQUEEZE` with both attr-based and input-based axes support
- Extended shape propagation:
  - `RECIPROCAL` now flows through the same-shape set
  - `SQUEEZE` now infers output shape from either axes attrs, runtime axes inputs, or the
    no-axes “remove all singleton dims” behavior
- Extended JAX codegen:
  - `RECIPROCAL` lowers to `jnp.reciprocal(...)`
  - `SQUEEZE` lowers through a new `_onnx_squeeze(...)` helper
  - the helper supports:
    - no axes
    - axes attrs
    - axes as a tensor input
    - negative axes normalization

Why it mattered:

- The reduced graph-only real FLUX.2 transformer checkpoint smoke was blocked on exactly these two
  unsupported ONNX ops.
- After adding them, the same reduced checkpoint transformer lane now reaches `ready` in the JAX
  checkpoint-prep report.

MLX follow-up:

- Mirror the same two ONNX mappings and semantic schemas.
- Add an MLX `_onnx_squeeze(...)` helper with the same axes behavior.
- Lower `RECIPROCAL` to the MLX equivalent unary reciprocal op.
- Reuse the same reduced parity graph so the backend delta stays isolated to emission, not test
  setup.

## Directed tests already added

### 4. ONNX dtype contract tests

File:

- `tests/unit/test_onnx_dtype_mapping.py`

What changed:

- Added assertions that all currently supported float8 ONNX enums resolve to the new internal dtype
  strings.

MLX follow-up:

- No MLX-specific change required here beyond keeping the ingest contract aligned.

### 5. JAX float8 codegen contract

File:

- `tests/unit/test_flux_fp8_jax_codegen_contract.py`

What changed:

- Added a tiny IR graph with `CAST` to/from float8.
- Asserts the generated JAX module contains the expected float8 dtype map entries and cast calls.

MLX follow-up:

- Add a matching MLX contract test that asserts:
  - the same ONNX float8 enum ids are recognized
  - the emitted MLX code uses the chosen fallback dtype path
  - no nonexistent MLX float8 dtype names appear in the generated source

### 6. JAX FP8 runtime parity

File:

- `tests/integration/test_flux_fp8_jax_runtime_parity.py`

What changed:

- Added two reduced runtime parity cases:
  - a linear path
  - an attention-like path
- Both cases exercise:
  - `CAST` to float8
  - `CAST` back to float16
  - `QUANTIZE`
  - `DEQUANTIZE`
- The reference implementation intentionally uses JAX float8 casts so the backend behavior is
  tested against the same dtype semantics the generated module uses.

MLX follow-up:

- Reuse the exact same reduced graphs and weights.
- Only change the reference expectation to match the MLX fallback cast policy.
- Keep the test names parallel so the backend delta is easy to compare.

### 7. Reciprocal + Squeeze JAX tests

Files:

- `tests/unit/test_op_mapping.py`
- `tests/unit/test_shape_prop.py`
- `tests/unit/test_jax_codegen_reciprocal_squeeze.py`
- `tests/integration/test_jax_reciprocal_squeeze_runtime_parity.py`

What changed:

- Added direct mapping assertions for `Reciprocal` and `Squeeze`.
- Added a shape-prop contract covering `SQUEEZE` followed by `RECIPROCAL`.
- Added a focused JAX codegen unit test that asserts:
  - `RECIPROCAL` renders as `jnp.reciprocal(...)`
  - `SQUEEZE` renders the new `_onnx_squeeze(...)` helper
- Added a focused JAX runtime parity test with a tiny ONNX graph using:
  - `Squeeze`
  - `Reciprocal`

MLX follow-up:

- Keep the ONNX mapping and shape-prop tests shared.
- Add a matching MLX codegen unit test for `SQUEEZE` and `RECIPROCAL`.
- Reuse the same tiny parity graph for the MLX runtime test.

### 8. Real FLUX VAE blocker op: `InstanceNormalization`

Files:

- `src/tnnx/ingest/op_map.py`
- `src/tnnx/ir/schema.py`
- `src/tnnx/passes/shape_prop.py`
- `src/tnnx/codegen/jax_codegen.py`

What changed:

- Added ONNX ingest mapping:
  - `InstanceNormalization -> INSTANCENORM`
- Added semantic schema:
  - `INSTANCENORM` with `data`, `scale`, `bias`, plus optional `epsilon`
- Extended shape propagation:
  - `INSTANCENORM` now flows through the same-shape set
- Extended JAX codegen:
  - `INSTANCENORM` lowers through a new `_onnx_instancenorm(...)` helper
  - the helper normalizes per sample, per channel across spatial axes (`axis >= 2`)

Why it mattered:

- After the reduced real FLUX.2 transformer lane reached `ready`, the next full graph-only sweep
  showed the real `vae_decoder` blocked on exactly `InstanceNormalization`.
- After adding it, the same full graph-only FLUX.2 sweep now reaches `ready` for both
  `vae_decoder` and `transformer`.

MLX follow-up:

- Mirror the same ONNX mapping and schema.
- Add an MLX `_onnx_instancenorm(...)` helper with the same per-instance, per-channel behavior.
- Reuse the same small runtime parity graph for the MLX backend.

## FLUX JAX example surfaces already added

Files:

- `examples/flux/source.py`
- `examples/flux/transpile_and_generate_jax.py`
- `tests/integration/test_flux_vae_decoder_jax_parity.py`
- `tests/integration/test_flux_transformer_jax_parity.py`
- `tests/integration/test_flux_jax_image_smoke.py`
- `tests/integration/test_flux_text_encoder_source.py`
- `tests/integration/test_flux_text_encoder_jax_parity.py`
- `tests/integration/test_flux_transformer_input_builder.py`
- `tests/integration/test_flux_transformer_checkpoint_source.py`

What changed:

- Added a reduced synthetic FLUX-like transformer and VAE decoder.
- Added a JAX smoke path that exports both to ONNX, transpiles both, runs a reduced denoising loop,
  and writes a PNG.
- The reduced JAX smoke path now accepts injected `prompt_embeddings` + `pooled_prompt`, so the
  eventual real prompt-encoder lane can plug into the existing image-generation harness.
- The JAX demo CLI can now load that prompt input pair from a `.npz` fixture file, which makes the
  handoff path usable outside direct Python calls.
- The reduced full prompt-to-image path is now active: the synthetic single text encoder is
  transpiled to JAX, its hidden state is bridged into reduced prompt inputs, and that feeds the
  existing transformer + VAE smoke harness.
- The reduced prompt-to-image JAX path now explicitly runs a standalone PyTorch reference first,
  then executes the generated JAX path.
- Added a second synthetic proof lane that matches the FLUX.2 single-text-encoder topology:
  `run_flux_pytorch_dummy_flux2_demo(...)` and `run_flux_jax_dummy_flux2_demo(...)`.
- That lane uses tiny random weights and tiny random inputs, explicitly bridges one text-encoder
  hidden-state tensor into the transformer prompt shape, and keeps the proof path cheap while
  staying aligned with the real FLUX.2 component graph.
- The prompt-to-image CLI path now also accepts token fixture `.npz` files, so the reduced text
  encoder lane can be driven by explicit token ids instead of only the baked-in demo ids.
- The checkpoint-backed prompt-to-image phase now has an explicit gated contract test that walks
  the detected real FLUX checkpoint topology through graph-only export and unsupported-op checks
  before surfacing the remaining real assembly blocker.
- The example layer now exposes `prepare_flux_jax_checkpoint_artifacts(...)`, which runs the same
  real submodule export + JAX transpile sweep and writes a JSON readiness report for local
  checkpoint debugging outside pytest.
- That checkpoint-prep path can now be narrowed to a single real lane with
  `--checkpoint-submodule`, so blocker burn-down can stay focused on the active submodule instead
  of sweeping the whole detected topology every time.
- The checkpoint-prep CLI now also supports a cheap proof mode with
  `--checkpoint-graph-only --checkpoint-reduced-shapes`.
- In that mode, the heavy real lanes now use reduced configs derived from the checkpoint metadata
  with random weights and export ONNX without serializing model weights. The transformer lane also
  uses smaller sample inputs. That keeps the proof path focused on export shape and ONNX op
  surface instead of full checkpoint load cost.
- The checkpoint-prep JSON report now records `used_reduced_config` per submodule, so later MLX
  work can tell whether a lane was exercised with true checkpoint weights or a reduced proof
  config.
- The real weighted `vae_decoder` lane is now also proven: the snapshot-gated JAX transpile test
  passes against the local `black-forest-labs/FLUX.2-klein-4B` snapshot with full weights.
- That checkpoint-prep path now degrades to a structured report when the target snapshot is
  missing, and the CLI can target an alternate local cache key via `--model-id`.
- Current observed local asset state:
  `black-forest-labs/FLUX.2-klein-4B` is now cached locally for real checkpoint iteration, and
  the new loader-free asset inspection confirms that `vae_decoder`, `transformer`, and
  `text_encoder` all have config + weights present in that snapshot. The remaining real blocker is
  now the transformer component load itself still being killed in this environment. Meanwhile
  `black-forest-labs/FLUX.2-klein-4b-fp8` still remains a separate single-file repo.
- The current repo can now read `HUGGING_FACE_TOKEN` from the shell environment or `.env` for
  checkpoint downloads.
- Real asset format split:
  `black-forest-labs/FLUX.2-klein-4B` exposes the diffusers-style folder layout the current loader
  can consume, while `black-forest-labs/FLUX.2-klein-4b-fp8` is currently a single-file checkpoint
  repo and will need separate loader support.
- The real checkpoint CLI defaults now target `black-forest-labs/FLUX.2-klein-4B`, while the FP8
  repo remains tracked separately as the long-term single-file checkpoint target.
- The `examples` dependency group now includes `accelerate` so real FLUX.2 component loading can
  avoid the slower fallback path.
- Added directed source export and JAX parity tests for the active submodule lanes.
- Added a real `text_encoder` source export lane, gated by `RUN_FLUX_E2E=1` and local snapshot
  availability.
- The snapshot-gated real `text_encoder` and real `transformer` source/export tests now also use
  reduced configs with random weights rather than full checkpoint weights, so the test path stays
  focused on graph/export behavior instead of checkpoint load cost.
- Added a snapshot-gated real `text_encoder` JAX transpile/runtime smoke test on that same
  reduced random-weight config, so the text-encoder lane now has a cheap real runtime proof as
  well.
- Added a snapshot-gated real `vae_decoder` JAX runtime smoke test on tiny real latents, so the
  weighted VAE lane now has both transpile and runtime coverage against the local FLUX.2 snapshot.
- Added snapshot-gated real `transformer` JAX transpile/runtime smoke tests on the reduced
  random-weight config with reduced sequence lengths, so the real transformer lane now has a
  bounded runtime proof as well.
- Raised the reduced transformer channel cap to `32` so the reduced transformer output aligns with
  the real FLUX.2 VAE latent-channel count instead of stopping at `16`.
- Added `tests/integration/test_flux_reduced_checkpoint_bridge_jax.py`, which runs the reduced
  real `text_encoder` -> `transformer` -> real `vae_decoder` bridge end to end, validates the
  PyTorch reference first, then compares the generated JAX path on the same reduced real topology.
- Promoted that bridge into the example layer as
  `run_flux_jax_reduced_checkpoint_bridge_demo()`, which writes both the PyTorch reference image
  and the generated JAX image for the reduced real topology.
- Fixed ONNX `Reshape` emission in `src/tnnx/codegen/jax_codegen.py` so the generated JAX path now
  honors the default zero-copy semantics (`0` in the requested shape copies the corresponding input
  dimension unless `allowzero=1`).
- Added topological scheduling in `src/tnnx/codegen/jax_codegen.py` before JAX emission. This
  fixes reduced real transformer graphs where a `Split` producer can arrive after its consumers in
  the raw IR order.
- Added directed JAX coverage for that topology-order fix:
  `tests/unit/test_jax_codegen_topology_contract.py` and
  `tests/integration/test_jax_topology_runtime_parity.py`.
- Added directed JAX coverage for that reshape fix:
  `tests/unit/test_jax_codegen_reshape_contract.py` and
  `tests/integration/test_jax_reshape_runtime_parity.py`.
- Kept JAX parity for text-encoder behavior on the tiny synthetic FLUX.2 lane so the test stays on
  random tiny weights and inputs instead of a heavyweight real checkpoint.
- The checkpoint-prep report now records the fixed real submodule order for the FLUX.2 path:
  `vae_decoder`, `transformer`, `text_encoder`.
- The checkpoint-prep report now executes the existing PyTorch forward on each real submodule
  before starting ONNX export or JAX transpile, and it records the reference output shapes in the
  JSON report.
- Added `build_flux_transformer_sample_inputs(...)` and `FluxTransformerForwardWrapper(...)`.
- Added a bounded tiny-transformer contract test so the real transformer input builder and wrapper
  are exercised without a full checkpoint.
- Added an opt-in checkpoint-backed transformer source export test that can `xfail` with the
  unsupported-op list until backend coverage catches up.
- Added shared ONNX op-inventory helpers so every FLUX source test uses the same unsupported-op
  calculation path.
- Added an opt-in checkpoint-backed transformer JAX transpile smoke test that runs after export and
  can `xfail` on the remaining real backend blockers.
- `prepare_flux_jax_checkpoint_artifacts(..., checkpoint_reduced_shapes=True)` now uses reduced
  random-weight configs for the heavy submodules (`transformer`, `text_encoder`) during JAX
  transpile, not just in graph-only mode, so the aggregate reduced checkpoint artifact gate can
  pass without loading the full 4B-class weights for those lanes.
- Centralized the snapshot-gated FLUX test setup in `tests/_support/flux.py` so new real submodule
  lanes reuse one skip/setup path instead of copying it.
- The current Torch ONNX exporter may also fail before a graph is emitted, so the checkpoint-backed
  source test treats exporter instability as an expected blocker and reports it via `xfail`.
- The FLUX transformer export path now uses `torch.onnx.export(..., dynamo=True)` and depends on
  `onnxscript`, which is carried in the `examples` dependency group.
- Added a deterministic tiny-config FLUX.2 proof lane that shrinks the real FLUX.2 config shapes
  aggressively while keeping the same three-submodule topology (`text_encoder`, `transformer`,
  `vae_decoder`).
- That lane caps the expensive dimensions (`num_attention_heads <= 2`, `num_hidden_layers <= 2`,
  `joint_attention_dim <= 32`, `latent_channels <= 8`, `sample_size <= 16`) and fills every
  floating-point parameter with constants instead of random weights.
- The tiny text-encoder reducer now explicitly aligns its `hidden_size` to the tiny transformer's
  reduced `joint_attention_dim`, which avoids a shape adapter in the bounded end-to-end bridge.
- Added `run_flux_pytorch_tiny_config_e2e_demo()` as the first validation step for that lane.
- Added `run_flux_jax_tiny_config_e2e_demo()`, which calls the PyTorch reference first, then runs
  the generated JAX path on the same constant-weight topology and compares final-image tensors.
- The CLI now exposes the same PyTorch preflight directly via
  `uv run python -m examples.run_flux_jax --use-tiny-config-torch`.
- The pooled prompt helper now forces `float32` reduction so the tiny deterministic bridge stays
  in the same dtype lane on both PyTorch and JAX.
- The CLI exposes the same proof path via
  `uv run python -m examples.run_flux_jax --use-tiny-config`.

MLX follow-up:

- Reuse the exact same synthetic modules and fixtures.
- Mirror the JAX `Reshape` zero-copy behavior before attempting the real VAE runtime lane.
- Mirror the reduced-checkpoint prep semantics too: reduced MLX artifact prep should use reduced
  random-weight configs for `transformer` and `text_encoder`, not full checkpoint weights.
- Mirror the reduced transformer channel cap too; the bounded MLX bridge should preserve the
  `32`-channel handoff into the real VAE latent shape.
- Mirror the topology scheduling fix too; the MLX emitter should not assume incoming IR node order
  is already dependency-safe for emission.
- Port the same source export tests first.
- Then port the submodule parity tests.
- Only after those pass, add an MLX image smoke test using the same reduced denoising loop and
  image assertions.

## Important current blockers

- Real FLUX `transformer` export now has sample-input wiring, but the checkpoint-backed ONNX graph
  is still expected to surface unsupported ops.
- Real FLUX `transformer` now has an opt-in checkpoint-backed JAX transpile smoke lane, but it is
  still expected to surface unsupported ops or other backend blockers before it becomes a stable
  parity lane.
- Real `text_encoder` is now snapshot-gated but covered only by an opt-in source export test.
  Text-encoder parity stays synthetic and tiny on purpose.
- No MLX backend changes have been made yet in this FLUX iteration.

## Implementation order for the MLX port later

1. Mirror the float8 ingest coverage if anything diverges.
2. Add MLX cast fallback policy for the float8 ONNX enums.
3. Port the FP8 codegen contract test.
4. Port the FP8 runtime parity test using the same reduced graphs.
5. Port the reduced FLUX transformer + VAE synthetic parity tests.
6. Port the image smoke path last.
