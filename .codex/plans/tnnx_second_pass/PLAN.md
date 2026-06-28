# tnnx Second-Pass Plan

## Purpose

Produce a verified audit, executable improvement plan, and LLM-facing repo context pack without product changes.

## Intended Final State

The execution branch should have reliable default validation, honest support tiers, no retired low-level residue, clearer public API/config semantics, backend parity gates, smaller verified surface area, and a maintained context pack.

- REVIEW_BASE_REF: `local-main-initial-review-base`
- REVIEW_BASE_SHA: `91bf6abf1f392863d0c3cdca692d3272efdceb4b`
- planning branch name: `codex/tnnx-second-pass-plan`
- PLAN_COMMIT_SHA: reported in the final completion response after the local plan commit exists; a commit cannot contain its own hash.

## Setup Notes

The incoming repo had no `HEAD` and all files were untracked. I recorded that state, verified `origin/main` had no advertised `main` head, created a baseline commit from the existing file contents, and created this isolated planning branch from that fixed base. This did not change product file contents.

## Retained Behavior Contract

Retain ONNX ingest, GraphIR construction/serialization/validation, conservative graph passes, JAX codegen, MLX codegen, `tnnx transpile`, examples/model-zoo paths that have evidence, and named model lanes at their validated tiers.

## LLM Context-Pack Contract

Keep `llms.txt`, `docs/llm_context/index.md`, topic pages, `log.md`, and machine indexes synchronized with code/tests/docs. Treat indexes as derived.

## Non-Goals

No product fixes, refactors, dependency changes, test rewrites, feature expansion, new backend/model support, or README/CI changes outside the allowed context pack during this audit commit.

## Current Architecture Summary

ONNX -> ingest -> GraphIR -> passes -> JAX/MLX codegen -> generated module + weights + metadata. The core is direct Python with explicit dictionaries and dispatch functions.

## Target Simplified Architecture

Keep the direct design. Add invariants and validation where they remove ambiguity. Delete or shrink only verified unused/stale surfaces. Share backend helpers only where existing tests prove behavior.

## Invariants That Must Remain True

- Unknown ONNX ops fail clearly.
- Generated JAX and MLX modules expose `load_weights` and `forward`.
- `weights.npz`, `compile_metadata.json`, and optional `graph_ir.json` artifact contract remains stable.
- JAX/MLX semantic dispatch stays aligned with `SEMANTIC_SCHEMAS`.
- Snapshot updates happen only after intentional backend output changes.
- Retired low-level/web surfaces stay absent.

## Severity-Ranked Deficiencies

| Severity | Count |
| --- | --- |
| HIGH | 5 |
| LOW | 3 |
| MEDIUM | 12 |

See `FINDINGS.md` for full evidence.

## Milestone Ordering

1. Fix validation environment and CI gates.
2. Fix confirmed backend/IR correctness gaps.
3. Clarify public API/config and docs support boundaries.
4. Verify/de-delete/shrink research, docs, helpers, and package contents.
5. Refresh context pack and release/demo docs after product behavior is stable.

## Complete Topologically Ordered Task List

### T00: Complete audit/context pack only

- ID: T00
- status: DONE
- phase: Context artifacts
- title: Complete audit/context pack only
- intended outcome: This commit only adds plan and context artifacts; no product files.
- finding IDs addressed: F005,F020
- rationale and evidence: see `FINDINGS.md` entries F005,F020; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality
- dependencies: none
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T00.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: This commit only adds plan and context artifacts; no product files.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: complete audit/context pack only`
### T01: Fix default dev/example dependency contract

- ID: T01
- status: TODO
- phase: Validation
- title: Fix default dev/example dependency contract
- intended outcome: Either include examples deps in the default test environment or gate FLUX tests behind an examples marker/group.
- finding IDs addressed: F001
- rationale and evidence: see `FINDINGS.md` entries F001; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality + tnnx-examples-model-zoo
- dependencies: none
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T01.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Either include examples deps in the default test environment or gate FLUX tests behind an examples marker/group.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: fix default dev/example dependency contract`
### T02: Make Whisper real-audio tests opt-in and environment-probed

- ID: T02
- status: TODO
- phase: Validation
- title: Make Whisper real-audio tests opt-in and environment-probed
- intended outcome: Unset default host-dependent MLX E2E or skip with a direct ffmpeg probe before running decode.
- finding IDs addressed: F002
- rationale and evidence: see `FINDINGS.md` entries F002; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality + tnnx-examples-model-zoo
- dependencies: none
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T02.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Unset default host-dependent MLX E2E or skip with a direct ffmpeg probe before running decode.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: make whisper real-audio tests opt-in and environment-probed`
### T03: Add minimal CI for existing gates

- ID: T03
- status: TODO
- phase: Validation
- title: Add minimal CI for existing gates
- intended outcome: Add one workflow for ruff, format, ty, pytest default, build, wheel smoke, and residue grep.
- finding IDs addressed: F003,F019
- rationale and evidence: see `FINDINGS.md` entries F003,F019; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality
- dependencies: none
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T03.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Add one workflow for ruff, format, ty, pytest default, build, wheel smoke, and residue grep.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: add minimal ci for existing gates`
### T04: Add MLX topology scheduler parity

- ID: T04
- status: TODO
- phase: Backend parity
- title: Add MLX topology scheduler parity
- intended outcome: Add failing MLX out-of-order SPLIT characterization, then share or port scheduler.
- finding IDs addressed: F004
- rationale and evidence: see `FINDINGS.md` entries F004; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-mlx-codegen
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T04.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Add failing MLX out-of-order SPLIT characterization, then share or port scheduler.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: add mlx topology scheduler parity`
### T05: Clarify or remove metadata-only config fields

- ID: T05
- status: TODO
- phase: API
- title: Clarify or remove metadata-only config fields
- intended outcome: Decide behavior for deterministic, emit_shape_asserts, and opset; keep as metadata only only if docs/tests say so.
- finding IDs addressed: F006
- rationale and evidence: see `FINDINGS.md` entries F006; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-ingest-ir + tnnx-jax-codegen
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T05.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Decide behavior for deterministic, emit_shape_asserts, and opset; keep as metadata only only if docs/tests say so.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: clarify or remove metadata-only config fields`
### T06: Surface ONNX shape-inference diagnostics

- ID: T06
- status: TODO
- phase: Ingest
- title: Surface ONNX shape-inference diagnostics
- intended outcome: Replace swallowed infer-shapes errors with a recorded diagnostic or explicit strict-mode failure.
- finding IDs addressed: F007,F017
- rationale and evidence: see `FINDINGS.md` entries F007,F017; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-ingest-ir
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T06.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Replace swallowed infer-shapes errors with a recorded diagnostic or explicit strict-mode failure.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: surface onnx shape-inference diagnostics`
### T07: Implement or de-advertise prune pass

- ID: T07
- status: TODO
- phase: Graph passes
- title: Implement or de-advertise prune pass
- intended outcome: Characterize dead live/unreachable nodes, then implement small pruning or remove it from default/docs.
- finding IDs addressed: F008
- rationale and evidence: see `FINDINGS.md` entries F008; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-graph-passes
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T07.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Characterize dead live/unreachable nodes, then implement small pruning or remove it from default/docs.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: implement or de-advertise prune pass`
### T08: Strengthen IR invariants

- ID: T08
- status: TODO
- phase: IR
- title: Strengthen IR invariants
- intended outcome: Add duplicate-output, producer, cycle/topology, and output provenance checks.
- finding IDs addressed: F009
- rationale and evidence: see `FINDINGS.md` entries F009; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-ingest-ir
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T08.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Add duplicate-output, producer, cycle/topology, and output provenance checks.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: strengthen ir invariants`
### T09: Fix stale docs links

- ID: T09
- status: TODO
- phase: Docs
- title: Fix stale docs links
- intended outcome: Remove or create `docs/demo-runbook.md`; add a simple repo-relative docs link check.
- finding IDs addressed: F010
- rationale and evidence: see `FINDINGS.md` entries F010; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T09.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Remove or create `docs/demo-runbook.md`; add a simple repo-relative docs link check.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: fix stale docs links`
### T10: Regenerate supported operator docs from code

- ID: T10
- status: TODO
- phase: Docs/operators
- title: Regenerate supported operator docs from code
- intended outcome: Replace stale README table with generated source-of-truth operator matrix.
- finding IDs addressed: F011
- rationale and evidence: see `FINDINGS.md` entries F011; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-ingest-ir + tnnx-test-quality
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T10.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Replace stale README table with generated source-of-truth operator matrix.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: regenerate supported operator docs from code`
### T11: Normalize named-model support tiers

- ID: T11
- status: TODO
- phase: Examples/model zoo
- title: Normalize named-model support tiers
- intended outcome: Mark each model Tier 0-5, ready/experimental/planned, assets, exact gate, and blockers.
- finding IDs addressed: F012
- rationale and evidence: see `FINDINGS.md` entries F012; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-examples-model-zoo
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T11.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Mark each model Tier 0-5, ready/experimental/planned, assets, exact gate, and blockers.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: normalize named-model support tiers`
### T12: Define package contents

- ID: T12
- status: TODO
- phase: Packaging
- title: Define package contents
- intended outcome: Decide whether tests/examples/research/.agents belong in sdist; update build config and smoke contents.
- finding IDs addressed: F013
- rationale and evidence: see `FINDINGS.md` entries F013; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T12.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Decide whether tests/examples/research/.agents belong in sdist; update build config and smoke contents.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: define package contents`
### T13: Audit research scripts for keep/delete

- ID: T13
- status: TODO
- phase: Cleanup
- title: Audit research scripts for keep/delete
- intended outcome: Run reachability/docs checks, then keep with owner or delete stale one-off scripts.
- finding IDs addressed: F014
- rationale and evidence: see `FINDINGS.md` entries F014; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-examples-model-zoo + tnnx-test-quality
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T13.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Run reachability/docs checks, then keep with owner or delete stale one-off scripts.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: audit research scripts for keep/delete`
### T14: Extract the smallest shared codegen helpers

- ID: T14
- status: TODO
- phase: LOC reduction
- title: Extract the smallest shared codegen helpers
- intended outcome: After MLX scheduler parity, centralize attr/slot/helpers only where snapshots prove no drift.
- finding IDs addressed: F015
- rationale and evidence: see `FINDINGS.md` entries F015; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-jax-codegen + tnnx-mlx-codegen
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T14.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: After MLX scheduler parity, centralize attr/slot/helpers only where snapshots prove no drift.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: extract the smallest shared codegen helpers`
### T15: Decide Python version support

- ID: T15
- status: TODO
- phase: Packaging
- title: Decide Python version support
- intended outcome: Probe 3.12/3.13 or document 3.14-only with CI and install messaging.
- finding IDs addressed: F016
- rationale and evidence: see `FINDINGS.md` entries F016; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T15.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Probe 3.12/3.13 or document 3.14-only with CI and install messaging.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: decide python version support`
### T16: Add trust-boundary/security characterization

- ID: T16
- status: TODO
- phase: Security
- title: Add trust-boundary/security characterization
- intended outcome: Test external ONNX data path handling, `.env` token behavior, download opt-in, and ffmpeg failure diagnostics.
- finding IDs addressed: F017
- rationale and evidence: see `FINDINGS.md` entries F017; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-ingest-ir + tnnx-examples-model-zoo
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T16.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Test external ONNX data path handling, `.env` token behavior, download opt-in, and ffmpeg failure diagnostics.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: add trust-boundary/security characterization`
### T17: Verify and remove unused public helpers

- ID: T17
- status: TODO
- phase: Shrink
- title: Verify and remove unused public helpers
- intended outcome: Check imports/docs/tests, then delete or keep `utils`, `coerce_attr`, `NameGenerator`, and common ordering helper.
- finding IDs addressed: F018
- rationale and evidence: see `FINDINGS.md` entries F018; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-jax-codegen + tnnx-test-quality
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T17.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Check imports/docs/tests, then delete or keep `utils`, `coerce_attr`, `NameGenerator`, and common ordering helper.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: verify and remove unused public helpers`
### T18: Add low-level/web residue guard

- ID: T18
- status: TODO
- phase: Residue
- title: Add low-level/web residue guard
- intended outcome: Codify the clean grep commands so retired FPGA/HLS/C/native/web surfaces stay out.
- finding IDs addressed: F019
- rationale and evidence: see `FINDINGS.md` entries F019; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T18.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Codify the clean grep commands so retired FPGA/HLS/C/native/web surfaces stay out.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: add low-level/web residue guard`
### T19: Add context-pack lint/update gate

- ID: T19
- status: TODO
- phase: LLM context
- title: Add context-pack lint/update gate
- intended outcome: Make a lightweight parser/check for index/log/machine indexes and require updates with material changes.
- finding IDs addressed: F020
- rationale and evidence: see `FINDINGS.md` entries F020; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T19.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Make a lightweight parser/check for index/log/machine indexes and require updates with material changes.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: add context-pack lint/update gate`
### T20: Add generated-code import/compile gates for both backends

- ID: T20
- status: TODO
- phase: Validation
- title: Add generated-code import/compile gates for both backends
- intended outcome: For representative models, require generated source parse/compile/import and no placeholder unresolved ops.
- finding IDs addressed: F004,F011
- rationale and evidence: see `FINDINGS.md` entries F004,F011; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-jax-codegen + tnnx-mlx-codegen
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T20.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: For representative models, require generated source parse/compile/import and no placeholder unresolved ops.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: add generated-code import/compile gates for both backends`
### T21: Clean skip/xfail policy

- ID: T21
- status: TODO
- phase: Validation
- title: Clean skip/xfail policy
- intended outcome: Name markers/env gates for expensive, network, snapshot, ffmpeg, MLX, FLUX, and Qwen lanes.
- finding IDs addressed: F001,F002,F012
- rationale and evidence: see `FINDINGS.md` entries F001,F002,F012; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T21.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Name markers/env gates for expensive, network, snapshot, ffmpeg, MLX, FLUX, and Qwen lanes.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: clean skip/xfail policy`
### T22: Consolidate historical FLUX notes

- ID: T22
- status: TODO
- phase: Docs/shrink
- title: Consolidate historical FLUX notes
- intended outcome: Compress `jax_backend_notes.md` and `submodule_plan.md` into current state plus archived context.
- finding IDs addressed: F012,F014,F015
- rationale and evidence: see `FINDINGS.md` entries F012,F014,F015; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-examples-model-zoo
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T22.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Compress `jax_backend_notes.md` and `submodule_plan.md` into current state plus archived context.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: consolidate historical flux notes`
### T23: Refresh context pack after each product task

- ID: T23
- status: TODO
- phase: LLM context
- title: Refresh context pack after each product task
- intended outcome: Update docs/llm_context pages, indexes, and log in the same PR as behavior changes.
- finding IDs addressed: F005,F020
- rationale and evidence: see `FINDINGS.md` entries F005,F020; baseline commands in `BASELINE.md` reproduce the current state.
- owner skill or primary reviewer: tnnx-test-quality
- dependencies: complete earlier validation/dependency tasks that affect the same files
- exact files/globs/symbols/config surfaces: see affected files in linked findings plus `FILE_CLASSIFICATION.tsv` task links for T23.
- context-pack pages/indexes that must be updated: `docs/llm_context/index.md`, `validation.md`, `known_limits.md`, `code_index.json`, `symbol_index.jsonl`, `graph_edges.tsv`; update backend/model-specific pages when touched.
- preconditions: checkout `codex/tnnx-second-pass-plan` at or after `REVIEW_BASE_SHA=91bf6abf1f392863d0c3cdca692d3272efdceb4b`; no unrelated worktree changes.
- pre-change command and expected result: run the linked validation gate from `VALIDATION_MATRIX.md`; it should reproduce the baseline failure or pass state named by the finding.
- exact edit/delete/refactor/test/doc/context-pack action: Update docs/llm_context pages, indexes, and log in the same PR as behavior changes.
- explicit non-goals: no new model/backend support, no speculative plugin framework, no weakening retained-behavior tests to make failures disappear.
- retained behavior at risk: ONNX ingest, IR serialization, JAX/MLX generated-code contracts, CLI artifact contract, and named model gates listed in `validation.md`.
- characterization test required before change: required for any BUG, BACKEND-PARITY, SECURITY, DELETE, or SHRINK action; skip only for pure docs/index updates.
- acceptance commands: run the task gate plus `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check src`, relevant pytest subset, and context-pack lint.
- expected results: all required gates pass, or environment-dependent gates skip with explicit reason.
- backend/model coverage: see linked finding; never broaden support claims beyond passing gates.
- context-pack acceptance checks: all touched source/test/doc paths remain represented in `code_index.json` and `index.md`; append `log.md`.
- rollback or task-revert procedure: revert the task commit only; no reset/clean; rerun the task gate and context-pack lint.
- hard-block condition: failing retained-behavior gate without a narrower characterization test or owner handoff.
- expected local commit message: `fix: refresh context pack after each product task`

## Validation Strategy

Use `VALIDATION_MATRIX.md`. Every task runs its focused gate plus lint/type/context checks. Do not weaken retained-behavior tests to hide failures.

## Context-Pack Update Strategy

Update the relevant Markdown page, `index.md`, `log.md`, and machine indexes in the same commit as any product behavior or public docs change. For pure plan work, update only plan/context artifacts.

## Decision Log

See `docs/llm_context/decisions.md`.

## Final Definition Of Done

- Baseline evidence is recorded.
- Findings map to tasks or blockers.
- File classification covers every tracked file plus this plan/context pack.
- Residue hits are absent or planned.
- Named models have tiered gates or blockers.
- Machine indexes parse.
- Only allowed plan/context files changed after the review base.
- A local plan/context commit exists and nothing is pushed.
