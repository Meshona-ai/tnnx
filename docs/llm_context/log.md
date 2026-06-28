# Context Log

## 2026-06-28 - Second-pass audit and context pack

- Base ref: `local-main-initial-review-base`
- Base SHA: `91bf6abf1f392863d0c3cdca692d3272efdceb4b`
- Branch: `codex/tnnx-second-pass-plan`
- Files/pages changed: created `llms.txt`, `docs/llm_context/**`, and `.codex/plans/tnnx_second_pass/**`.
- Reason: second-pass audit, executable improvement plan, and persistent LLM-facing repo wiki.
- Validation performed: ruff check PASS; ruff format PASS; ty PASS; default pytest FAIL with 15 pre-existing failures before examples deps; examples-enabled pytest FAIL with 2 host `ffmpeg` failures; build PASS; clean wheel import PASS on Python 3.14; CLI help PASS; CLI JAX smoke PASS; generated JAX py_compile PASS; ResNet-18 model-zoo smoke PASS; low-level residue grep PASS/no hits.
- Unresolved questions: whether default dev should include examples deps; whether `RUN_MLX_E2E=1` should remain default; whether Python 3.14-only is intentional; which research scripts are retained; whether metadata-only config fields should stay public.

## 2026-06-28 - T01/T02 validation default cleanup

- Files/pages changed: `pyproject.toml`, `uv.lock`, `examples/whisper_audio/whisper_hf_tiny_source.py`, Whisper real-audio integration tests, `tests/unit/test_whisper_audio_ffmpeg.py`, `docs/llm_context/validation.md`, `known_limits.md`, `model_zoo.md`, and `context_pack_lint.md`.
- Reason: make default pytest match its dependency contract and run Whisper real-audio MLX tests by fixing the observed `ffmpeg`/x265 dylib mismatch instead of skipping.
- Validation performed: `uv sync --dev` PASS; focused FLUX tests PASS (`23 passed`); Whisper real-audio MLX plus fallback unit test PASS (`4 passed`); default pytest PASS (`225 passed, 12 skipped`); ruff/type checks PASS.
- Environmental note: `brew upgrade ffmpeg` was attempted but `/opt/homebrew` is not writable by the current user. The code fallback uses the installed x265 4.1 keg when `ffmpeg` reports missing `libx265.215.dylib`.

## 2026-06-28 - Execution implementation refresh

- Files/pages changed: CI workflow, validation scripts, codegen common/JAX/MLX, ingest, IR schema, prune pass, package config, docs, examples docs, tests, and context pack.
- Reason: execute T03-T23: CI/local gates, MLX topology parity, metadata-only config docs, ONNX shape diagnostics, prune behavior, IR invariants, docs links, generated operator docs, model tiers, sdist policy, research-script deletion, shared codegen helpers, Python 3.14 decision, trust-boundary tests, unused-helper deletion, residue guard, context-pack gate, generated-code gate, skip marker policy, and FLUX note consolidation.
- Validation performed during implementation: backend snapshots/parity PASS (`33 passed`); ingest/IR/pass/security focused PASS (`53 passed`); named model focused PASS (`15 passed, 2 skipped`); FLUX focused PASS (`8 passed, 1 skipped`); generated-code script PASS; docs link/operator/residue scripts PASS; build and package-content check PASS. The completed full-suite and clean-room package checks are recorded in the final validation entry below.

## 2026-06-28 - Final execution validation

- Files/pages changed: final context-pack truthfulness cleanup and removal of `.codex/plans/tnnx_second_pass/**` from the final branch head.
- Reason: finish the second-pass execution branch with no stale plan references, no retained temporary plan files, and validation facts matching the final tree.
- Validation performed: final-tree `bash scripts/check.sh` PASS, including ruff, format, ty, default pytest (`238 passed, 12 skipped`), build, package contents, generated-code runtime, residue, docs links, operator docs, and context-pack check. Clean-room wheel import and installed `tnnx --help` PASS from `/private/tmp` with Python 3.14. CLI JAX/MLX tiny ONNX smoke PASS with expected runtime outputs. ResNet-18 model-zoo JAX/MLX smoke PASS. `git diff --check` PASS.
