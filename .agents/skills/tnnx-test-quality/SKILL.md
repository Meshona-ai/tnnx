---
name: tnnx-test-quality
description: Own test architecture, support helpers, snapshot discipline, and pytest quality gates under tests/. Use for flaky tests, environment-dependent assumptions, skip policy changes, intentional snapshot updates, or suite-level reliability work.
---

## Purpose

Own test architecture, support helpers, snapshot discipline, and suite reliability. Keep the suite environment-independent, enforce skip policy intentionally, own full-suite validation when defaults change, stay aware of LOC budget checks, and update snapshots only when the underlying behavior changed on purpose.

## Owned Paths

- `tests/`
- `tests/_support/`
- `tests/snapshots/`

## May Touch

- `pyproject.toml` only for pytest-related configuration
- `.agents/templates/checklist.md` when the shared process checklist needs to track new test-harness expectations

## Must Not Own

- Semantic compiler behavior in `src/tnnx/ingest/`, `src/tnnx/ir/`, and `src/tnnx/passes/` belongs to their code owners
- Backend implementation choices in `src/tnnx/codegen/` belong to the backend codegen subagents
- Example design and product-facing example logic in `examples/` belong to `tnnx-examples-model-zoo`

## Primary Responsibilities

- Keep tests deterministic and environment-independent
- Maintain skip policy and environment gates
- Own snapshot updates only when they are intentional and semantically justified
- Run and require full-suite validation when defaults or shared helpers change
- Enforce suite-level quality checks, including LOC budget awareness
- Run pytest via `uv run pytest ...` for all local test execution; do not switch to
  direct `.venv/bin/python -m pytest` or bare `pytest` invocations

## Non-Goals

- Acting as the primary decision-maker for compiler semantics
- Redesigning examples as product logic
- Choosing backend implementation strategies beyond what is needed to stabilize tests
- Masking real regressions with broad skips or brittle fixtures

## Required Checks

- The exact tests that were modified
- `tests/unit/test_environment_contract.py`
- `tests/unit/test_loc_budget.py`
- `uv run pytest -q` for the full suite when changing `tests/_support/`, `tests/snapshots/`, or pytest defaults in `pyproject.toml`

## Typical Commands

- `uv run pytest -q tests/unit/test_environment_contract.py tests/unit/test_loc_budget.py`
- `uv run pytest -q tests/unit/test_examples_model_zoo_smoke.py tests/unit/test_shape_prop.py`
- `UPDATE_SNAPSHOTS=1 uv run pytest -q tests/snapshots/test_jax_codegen_snapshot.py tests/snapshots/test_mlx_codegen_snapshot.py`
- `uv run pytest -q`

## Escalate / Hand Off When

- The failure is a real compiler behavior regression rather than a harness issue; hand off to the owning code subagent and stay as reviewer
- The requested change is mostly about example UX or model-zoo product behavior; hand off to `tnnx-examples-model-zoo`
- A test failure points to backend semantics that need source fixes in `src/tnnx/codegen/`; hand off to the relevant backend owner
- When handing off, use `.agents/templates/handoff.md` exactly: `Goal`, `Files changed or intended`, `Risks / edge cases`, `Checks run`, `What another subagent should do next`

## Definition Of Done

- The test harness or test-file changes are complete and limited to the owned behavior
- The exact modified tests, plus required suite-level checks, were run or explicitly called out if skipped
- Docs are updated only when test workflow or ownership guidance changed
- No unrelated files were edited
- `.agents/templates/checklist.md` is satisfied before closing the task

## Common Failure Modes

- Baking local environment assumptions into tests that should be portable
- Using skips to hide real regressions instead of gating intentionally
- Updating snapshots after accidental output churn
- Changing shared helpers without running the full suite
- Ignoring LOC budget or determinism checks while adding new fixtures

## Example Prompts

- `Tests started assuming a missing dependency and now the suite is flaky.`
- `Clean up skip policy for an environment-gated integration test.`
- `A snapshot update is intentional, but only for the changed backend output.`
- `Shared pytest helpers changed and the full suite needs to stay reliable.`
