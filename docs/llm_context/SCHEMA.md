# Context Pack Schema

## Page Conventions

- Start each page with a single H1 matching the file purpose.
- Prefer repo-relative source references such as `src/tnnx/api.py` or `tests/integration/test_cli_transpile_jax.py`.
- Keep claims short and traceable to code, tests, docs, config, or observed baseline output.
- Use `VERIFY` for uncertain support, deletion, or reachability claims.
- Use stable headings: Scope, Sources, Current Behavior, Known Limits, Validation, Update Triggers.

## Frontmatter

Frontmatter is optional. If added later, keep it small: `owner`, `status`, `last_verified_sha`, `source_paths`.

## Cross-Linking

- Link human pages through `index.md`.
- Link plan work through `.codex/plans/tnnx_second_pass/PLAN.md`.
- Do not deep-link to generated indexes as the only source for a claim.

## Machine Indexes

- `code_index.json` is a compact path index with disposition, subsystem, imports, exports, context links, line counts, byte sizes, and SHA-256 fingerprints.
- The `code_index.json` entry for `docs/llm_context/code_index.json` itself uses null fingerprint fields because the index is self-referential.
- `symbol_index.jsonl` is one JSON object per function/class/constant/test-like symbol.
- `graph_edges.tsv` captures lightweight relationships: imports, handles_operator, maps_onnx_op, tests_backend, tests_model_path, documents.

## Required Updates

- Code change: update `code_index.json`, `symbol_index.jsonl`, relevant architecture/operator/backend/model pages, and `log.md`.
- Test change: update `validation.md`, `known_limits.md` if gates change, indexes, and `log.md`.
- Docs change: update `index.md`, `context_pack_lint.md`, and `log.md`.
- Public API or CLI change: update `public_api_cli.md`, `validation.md`, `llms.txt` if entrypoints change.
- Model/backend/operator change: update the specific model/backend/operator pages and validation gates.

## Forbidden Wiki Behavior

- Do not invent support claims from names or aspirational docs.
- Do not treat stale summaries as truth when code/tests disagree.
- Do not copy large source blocks into Markdown.
- Do not include secrets, token values, or private cache paths.
- Do not mark code dead from grep alone; require reachability review.
