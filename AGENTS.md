# `tnnx` Project Subagents

This repository uses repo-local project subagents committed under `.agents/skills/`. Use the minimal set of specialists needed for the task. Default to one primary subagent, and let that owner edit its owned area directly.

## Available Project Subagents

- `tnnx-ingest-ir`
  Skill: `.agents/skills/tnnx-ingest-ir/SKILL.md`
  Primary ownership: `src/tnnx/ingest/`, `src/tnnx/ir/`
- `tnnx-graph-passes`
  Skill: `.agents/skills/tnnx-graph-passes/SKILL.md`
  Primary ownership: `src/tnnx/passes/`
- `tnnx-jax-codegen`
  Skill: `.agents/skills/tnnx-jax-codegen/SKILL.md`
  Primary ownership: `src/tnnx/codegen/jax_codegen.py`, `src/tnnx/codegen/common.py`
- `tnnx-mlx-codegen`
  Skill: `.agents/skills/tnnx-mlx-codegen/SKILL.md`
  Primary ownership: `src/tnnx/codegen/mlx_codegen.py`
- `tnnx-examples-model-zoo`
  Skill: `.agents/skills/tnnx-examples-model-zoo/SKILL.md`
  Primary ownership: `examples/`
- `tnnx-test-quality`
  Skill: `.agents/skills/tnnx-test-quality/SKILL.md`
  Primary ownership: `tests/`, `tests/_support/`, `tests/snapshots/`

Only these 6 project subagents are active in phase 1. Do not add optional subagents until repeated demand justifies them.

## When To Use Them

- Use `tnnx-ingest-ir` for ONNX parsing, initializer extraction, optional input slot handling, GraphIR schema compatibility, and deterministic IR naming issues.
- Use `tnnx-graph-passes` for normalization, pruning, runtime-driven shape propagation, broadcast semantics, or `Resize` / `Expand` / `Reshape` inference issues.
- Use `tnnx-jax-codegen` for JAX code emission, shared backend helper changes in `src/tnnx/codegen/common.py`, JAX parity regressions, or JAX snapshot drift.
- Use `tnnx-mlx-codegen` for MLX code emission, MLX layout issues, MLX-specific tolerance questions, or MLX parity regressions.
- Use `tnnx-examples-model-zoo` for examples, smoke jobs, model-zoo loaders, compatibility reporting, and example ergonomics.
- Use `tnnx-test-quality` for test determinism, snapshot discipline, skip policy, environment gates, and full-suite quality guardrails.

## Routing Rules

- Default to one primary subagent.
- Add one secondary subagent only when it is necessary for a narrow review or a minimal cross-boundary change.
- If a task touches more than two subagents, use one primary owner and one reviewer specialist. Do not fan out to many specialists.
- If a change touches public runtime behavior and tests, the code owner stays primary and `tnnx-test-quality` is the reviewer.
- If a change touches examples plus core compiler semantics, the core compiler owner stays primary and `tnnx-examples-model-zoo` is the reviewer.
- If a task touches `src/tnnx/codegen/common.py`, the active backend owner remains primary; default to `tnnx-jax-codegen` unless the change is MLX-only.
- Prefer the owner whose `Owned Paths` contain the main edited files. Treat every other specialist as advisory unless a handoff is required.

## Ownership And Escalation

- Each subagent is allowed to edit its owned paths directly.
- A subagent may edit outside its owned paths only when the external change is directly required to wire its owned behavior, the external change is minimal, and the subagent also updates or adds the required tests for that cross-boundary change.
- If any one of those conditions is not true, the current owner must hand off instead of expanding scope.
- `src/tnnx/ingest/`, `src/tnnx/ir/`, `src/tnnx/passes/`, `src/tnnx/codegen/`, `examples/`, and `tests/` are covered by the primary owners above. Do not leave those areas without an explicit owner during a task.
- `src/tnnx/api.py` is wiring-only territory: whichever primary owner needs the smallest direct orchestration change may touch it, then hand back.
- Keep cross-boundary edits small, local, and tied to the owned behavior that justified them.

## Handoff Contract

Use `.agents/templates/handoff.md` whenever work moves from one subagent to another. Every handoff must include:

1. `Goal`
2. `Files changed or intended`
3. `Risks / edge cases`
4. `Checks run`
5. `What another subagent should do next`

Use `.agents/templates/checklist.md` before finishing a task to confirm ownership, checks, docs impact, and unrelated-edit discipline.
