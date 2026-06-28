# Context Index

Review base: `local-main-initial-review-base` / `91bf6abf1f392863d0c3cdca692d3272efdceb4b`. Generated: `2026-06-28T13:27:25+00:00`.

## Read First

- `README.md`: root LLM entrypoint and project support boundary.
- `architecture.md`: current package layout and ownership boundaries.
- `dataflow.md`: ONNX -> IR -> passes -> JAX/MLX artifact flow.
- `validation.md`: commands, baseline results, and tiered gates.
- `context_pack_lint.md`: current context-pack drift checks and update rules.

## Pages

- `README.md`: how to use and maintain this context pack.
- `SCHEMA.md`: page/index conventions and forbidden wiki behavior.
- `index.md`: this content-oriented index.
- `log.md`: append-only audit/context update log.
- `architecture.md`: package layout, API/CLI/example entrypoints, retained and removed surfaces.
- `dataflow.md`: graph construction, constants, shape/dtype flow, codegen contract.
- `public_api_cli.md`: Python exports, CLI flags, config keys, stable and unstable behavior.
- `operators.md`: semantic operator map and add-operator workflow.
- `backends_jax_mlx.md`: JAX/MLX emitters, shared contracts, parity risks.
- `model_zoo.md`: named model lanes, assets, tiers, commands, blockers.
- `validation.md`: local checks, CI target, clean install, baseline status.
- `extension_guide.md`: practical steps for operators, models, backends, tests, and context updates.
- `known_limits.md`: unsupported operators/paths, backend gaps, environment gates, baseline failures.
- `decisions.md`: current architecture decisions and non-goals.
- `context_pack_lint.md`: orphan/stale/contradiction checks for this pack.
- `code_index.json`: machine-readable path index.
- `symbol_index.jsonl`: machine-readable symbol index.
- `graph_edges.tsv`: lightweight relationship graph.

## Key Source Directories

- `src/tnnx/ingest/`: ONNX load, initializer extraction, dtype/op mapping.
- `src/tnnx/ir/`: GraphIR types, validation, JSON serde.
- `src/tnnx/passes/`: normalize, prune, shape propagation.
- `src/tnnx/codegen/`: JAX and MLX generated Python emitters.
- `src/tnnx/runtime/`: weight `.npz` helpers.
- `examples/`: small models, Qwen, Whisper, FLUX, model-zoo smoke paths.
- `tests/`: unit, integration, snapshots, environment contracts.
- `scripts/`: reusable validation gates used by local checks and CI.

## Key Commands

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run ty check src`
- `uv run pytest -q`
- `uv sync --dev --group examples && uv run pytest -q`
- `uv build`
- `uv run python scripts/check_generated_code.py`
- `uv run python scripts/check_package_contents.py`
- `uv run python scripts/check_residue.py`
- `uv run python scripts/check_context_pack.py`
- `uv run tnnx transpile --onnx <model.onnx> --target jax --out <out>`
- `uv run python -m examples.model_zoo.transpile_smoke --target jax --model ResNet-18`

## Extension Points

- Operators: `src/tnnx/ingest/op_map.py`, `src/tnnx/ir/schema.py`, `src/tnnx/passes/shape_prop.py`, `src/tnnx/codegen/jax_codegen.py`, `src/tnnx/codegen/mlx_codegen.py`, tests and snapshots.
- Backends: `src/tnnx/codegen/*_codegen.py`, `src/tnnx/codegen/common.py` only after evidence.
- Models: `examples/model_zoo/*`, `examples/qwen/*`, `examples/flux/*`, `examples/whisper_audio/*`.
- Public API/CLI: `src/tnnx/api.py`, `src/tnnx/cli.py`, `src/tnnx/config.py`, `src/tnnx/__init__.py`.

## Support Boundaries

Current code supports 69 semantic IR ops in both JAX and MLX dispatch. ONNX ingest maps 70 ONNX op spellings into 67 semantic ops. `RELU6` and `SILU` are schema/codegen-supported but not currently produced by `ONNX_TO_SEMANTIC` directly; treat them as generated-IR/internal support until a mapping or lowering is verified.

Retired low-level and web/server scope stays out. The reusable residue guard found no true hits.
