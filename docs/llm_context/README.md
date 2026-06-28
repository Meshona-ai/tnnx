# LLM Context Pack

This directory is the durable operating map for future Codex runs, LLM agents, ML engineers, and reviewers working on `tnnx`.

Use it in this order:

1. Read `index.md` for the page map and support boundaries.
2. Read `architecture.md`, `dataflow.md`, and `public_api_cli.md` before editing source.
3. Read `operators.md` and `backends_jax_mlx.md` before touching codegen or semantic ops.
4. Read `model_zoo.md` and `validation.md` before changing examples or tests.
5. Read `known_limits.md` before interpreting support claims.

Canonical sources are the code, tests, `pyproject.toml`, README/docs, and observed command output from validation runs. Markdown pages here are curated summaries. JSON/JSONL/TSV files are derived aids and must be refreshed after material changes. Temporary `.codex/plans/**` execution artifacts are excluded from the final branch head and from generated context indexes.

Update rules:

- Append `log.md` for every audit or context-pack update.
- Update `index.md` when pages are added, removed, renamed, or materially changed.
- Update machine indexes after code, tests, docs, public API, model-path, or backend changes.
- Mark uncertain support claims as `VERIFY`; do not turn docs claims into facts without code/tests or command evidence.
