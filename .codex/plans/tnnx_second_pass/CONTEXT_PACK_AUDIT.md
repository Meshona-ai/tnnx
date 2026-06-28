# Context Pack Audit

## Created Files

- `docs/llm_context/README.md`
- `docs/llm_context/SCHEMA.md`
- `docs/llm_context/index.md`
- `docs/llm_context/log.md`
- `docs/llm_context/architecture.md`
- `docs/llm_context/dataflow.md`
- `docs/llm_context/public_api_cli.md`
- `docs/llm_context/operators.md`
- `docs/llm_context/backends_jax_mlx.md`
- `docs/llm_context/model_zoo.md`
- `docs/llm_context/validation.md`
- `docs/llm_context/extension_guide.md`
- `docs/llm_context/known_limits.md`
- `docs/llm_context/decisions.md`
- `docs/llm_context/context_pack_lint.md`
- `docs/llm_context/code_index.json`
- `docs/llm_context/symbol_index.jsonl`
- `docs/llm_context/graph_edges.tsv`
- `llms.txt`

## Source Areas Covered

- Architecture/dataflow/public API: `src/tnnx/**`, `pyproject.toml`, `README.md`.
- Operators/backends: `src/tnnx/ingest/op_map.py`, `src/tnnx/ir/schema.py`, `src/tnnx/passes/shape_prop.py`, `src/tnnx/codegen/**`, snapshots and parity tests.
- Models/examples: `examples/**`, model-zoo commands, Qwen/Whisper/FLUX/NanoGPT/GPT-2 docs and tests.
- Validation: `scripts/check.sh`, pytest output, build/wheel/CLI smoke, residue grep.
- Cleanup: research scripts, package contents, helpers, historical docs.

## Known Omissions

- No Git history analysis beyond current base because the repo started with no commits in this worktree.
- No full real Qwen/FLUX checkpoint run because env gates/assets were absent.
- No Whisper real-audio success because host `ffmpeg` is broken.

## Machine-Index Generation Method

Python stdlib AST parsed `.py` files for imports, classes, functions, constants. Path classifications use repo-relative heuristic rules tied to findings/tasks. Graph edges include imports, op mappings, schema/operator handling, tests by filename, and document edges.

## Future Maintenance Rules

Follow `docs/llm_context/SCHEMA.md`; append `log.md`; refresh indexes after material changes.
