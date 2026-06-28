---
name: tnnx-examples-model-zoo
description: Own examples, smoke jobs, model-zoo loaders, and compatibility reporting under examples/. Use for example ergonomics, model-zoo loader issues, readiness-reporting fixes, or keeping smoke workflows runnable in the default dev environment.
---

## Purpose

Own examples, smoke jobs, model-zoo loaders, compatibility reporting, and example ergonomics. Keep smoke examples clean and centralized, avoid duplicated model-family sources, and make sure readiness reporting stays honest while examples remain runnable under the default dev environment.

## Owned Paths

- `examples/`

## May Touch

- `tests/unit/test_examples_model_zoo_catalog.py`
- `tests/unit/test_examples_model_zoo_smoke.py`
- `tests/integration/test_model_zoo_generated_runtime_parity.py`
- `tests/integration/test_nanogpt_tiny_example_jax.py`
- `tests/integration/test_whisper_tiny_example_mlx.py`
- `tests/integration/test_whisper_audio_transpile_source.py`
- `tests/integration/test_whisper_audio_real_mlx.py`

## Must Not Own

- Low-level ONNX ingest and IR semantics belong to `tnnx-ingest-ir`
- Pass behavior and backend helper implementations belong to `tnnx-graph-passes`, `tnnx-jax-codegen`, and `tnnx-mlx-codegen`
- Test harness policy, snapshot discipline, and skip defaults belong to `tnnx-test-quality`

## Primary Responsibilities

- Keep examples clean, runnable, and centralized under one maintained source of truth per model family
- Maintain model-zoo loaders, smoke jobs, and compatibility reporting
- Keep readiness reporting honest and never over-claim support
- Own loader fallbacks versus real instantiation behavior
- Preserve default-dev-environment usability for example paths

## Non-Goals

- Owning low-level semantic mapping or backend code generation decisions
- Duplicating example logic across multiple model-family entry points
- Changing global test harness policy for convenience
- Hiding broken runtime paths behind inflated compatibility reporting

## Required Checks

- `tests/unit/test_examples_model_zoo_catalog.py`
- `tests/unit/test_examples_model_zoo_smoke.py`
- `tests/integration/test_model_zoo_generated_runtime_parity.py`
- `tests/integration/test_nanogpt_tiny_example_jax.py` when NanoGPT paths changed
- `tests/integration/test_whisper_tiny_example_mlx.py` and `tests/integration/test_whisper_audio_transpile_source.py` when Whisper paths changed
- `tests/integration/test_whisper_audio_real_mlx.py` when default MLX runtime flow changed

## Typical Commands

- `uv run pytest -q tests/unit/test_examples_model_zoo_catalog.py tests/unit/test_examples_model_zoo_smoke.py`
- `uv run pytest -q tests/integration/test_model_zoo_generated_runtime_parity.py`
- `uv run pytest -q tests/integration/test_nanogpt_tiny_example_jax.py tests/integration/test_whisper_tiny_example_mlx.py tests/integration/test_whisper_audio_transpile_source.py`
- `RUN_MLX_E2E=1 uv run pytest -q tests/integration/test_whisper_audio_real_mlx.py`

## Escalate / Hand Off When

- The requested fix is really about ingest semantics, shape propagation, or backend emission rather than example-layer behavior; hand off to the relevant compiler owner
- The work becomes primarily about test harness policy, flaky external env assumptions, or snapshot process; hand off to `tnnx-test-quality`
- A model-zoo issue depends on new backend support rather than loader or reporting logic; hand off to the backend owner and stay as reviewer only if needed
- When handing off, use `.agents/templates/handoff.md` exactly: `Goal`, `Files changed or intended`, `Risks / edge cases`, `Checks run`, `What another subagent should do next`

## Definition Of Done

- The example or model-zoo change is complete and limited to the owned behavior
- The relevant smoke, example, and parity checks were run or explicitly called out if skipped
- Docs are updated only when example workflow or ownership guidance changed
- No unrelated files were edited
- `.agents/templates/checklist.md` is satisfied before closing the task

## Common Failure Modes

- Copying the same model-family logic into multiple example files
- Letting smoke examples drift away from the default dev environment
- Reporting support that only works in a hand-tuned local setup
- Falling back to fake loaders that hide real execution failures
- Fixing example symptoms when the real bug is in the compiler core

## Diffusion-Family Onboarding

When adding a large diffusion-family model (for example FLUX-like pipelines), prefer a staged
standalone example under `examples/<model>/` before any `examples/model_zoo` promotion.

- Start with one backend only and keep the first milestone small.
- Keep scheduler and tokenizer orchestration in Python until the core model path is proven.
- Stage submodules linearly: decoder first, then core denoiser, then reduced image path, then
  prompt encoders.
- Add fixture-driven seams before the real heavy path is ready:
  first prompt-embedding fixtures, then token-id fixtures, then real prompt encoders.
- For the fastest workflow proof, add a synthetic tiny-topology lane with random weights and very
  small tensors that preserves the target model family's component layout; run the PyTorch
  reference through that lane before any generated-backend image E2E work.
- After the synthetic proof, add a deterministic tiny-config lane derived from the real model
  config: shrink the expensive dimensions aggressively, fill weights with constants, use
  zero/one-valued inputs where possible, and validate the PyTorch reference first before running
  the generated backend on the same bounded topology.
- Keep one blocker ledger in the example folder instead of scattering unsupported-op notes across
  multiple tests.
- Add one source export test and one runtime parity test per active submodule lane.
- Gate expensive image-generation checks behind an env var and local-cache requirements.
- Add a reusable checkpoint-prep command or helper that writes a readiness report for real local
  assets; this keeps checkpoint debugging out of the smoke path.
- Before any generated-backend image E2E run, execute and validate the existing PyTorch reference
  path first. Do not start by validating generated JAX/MLX image output in isolation.
- Promote the model into `examples/model_zoo` only after the standalone path is stable and the
  readiness report can stay honest.

## LLM-Family Onboarding

When adding a decoder-only LLM family (for example GPT-, Llama-, or Qwen-like
text generation), keep the first milestone as a standalone example under
`examples/<model>/` and prove one backend end to end before any model-zoo
promotion.

- Capture runtime versions (`torch`, `transformers`, `onnx`, backend runtime)
  before ONNX export so export regressions can be tied back to the local toolchain.
- Start with a deterministic tiny synthetic lane that preserves the target
  model family's export structure and keep it always-on in tests.
- Force the tiny lane to the smallest topology that still exercises the new
  backend/operator surface; use it to land missing ops before touching the real
  checkpoint path.
- Require prompt string -> decoded string acceptance, not just prompt-window
  logits parity. A blank or special-token-only decode is not a completed lane.
- Gate real-checkpoint E2E tests behind an env var plus explicit cache/network
  expectations so the default suite stays deterministic.
- Promote the model into `examples/model_zoo` only after the standalone
  generated-backend path is stable, reproducible, and honest about checkpoint
  requirements.

## Example Prompts

- `Add a clean model-zoo example for a new model family without duplicating loaders.`
- `The compatibility matrix is overstating support for a model-zoo entry.`
- `A Whisper example should run again in the default dev environment.`
- `Fix a broken model-zoo loader in examples/model_zoo/loaders.py.`
