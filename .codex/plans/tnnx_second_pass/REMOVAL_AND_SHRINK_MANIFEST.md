# Removal And Shrink Manifest

These are hypotheses with gates, not approved deletions.

| Category | Candidate | Evidence | Estimated reduction | Risk | Task | Context pages | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VERIFY delete | scripts/research/** | 3699 LOC, not in default checks/docs as product surface | 0-11 files / 0-3699 LOC | may contain useful benchmark evidence | T13 | validation.md; decisions.md | reachability/docs check plus owner decision |
| VERIFY delete | src/tnnx/utils/** | only exports stable_dict and TranspilerError; no product callers | 2-3 files / small LOC | external imports may rely on submodule | T17 | public_api_cli.md | public-use grep and deprecation decision |
| VERIFY shrink | src/tnnx/codegen/common.py | NameGenerator/order_nodes helper not used by backends | 1 file or repurpose | tests import helper; possible external use | T17 | backends_jax_mlx.md | public-use grep; replace with real scheduler only if needed |
| SHRINK | JAX/MLX attr and slot helpers | duplicated helper patterns in both backend files | 100-500 LOC estimate | snapshot/runtime drift | T14 | backends_jax_mlx.md; operators.md | snapshots and parity pass |
| SHRINK | examples/flux/jax_backend_notes.md and submodule_plan.md | historical handoff notes duplicate current docs/context | 100-500 LOC estimate | loss of hard-won model history | T22 | model_zoo.md; decisions.md | current-state summary preserves decisions and gates |
| FIX/SHRINK | sdist contents | sdist has 215 entries including examples/tests/scripts/.agents; wheel has 30 package entries | package bytes/files | sdist may intentionally include tests/examples | T12 | validation.md | build contents checked against policy |
| FIX or remove | metadata-only config knobs | deterministic/emit_shape_asserts/opset serialized but not behavioral | API surface simplification | breaking public API | T05 | public_api_cli.md | API tests and docs updated |

No candidate is safe to remove from Ponytail output alone.
