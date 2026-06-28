# Context Pack Lint

## Orphan Pages

None. `index.md` lists every context-pack page generated in this pass.

## Stale Claims

- README supported-op table is stale versus schema/codegen and is tracked by F011/T10.
- `docs/README.md` references missing `demo-runbook.md` and is tracked by F010/T09.
- Model docs mix ready, experimental, planned, and environment-gated paths; tracked by F012/T11.

## Unresolved References

- `docs/demo-runbook.md` missing.
- Full Qwen, FLUX, and Whisper real lanes need asset/env status to be verified per environment.

## Pages Missing Index Entries

None for generated pages.

## Code Paths Missing Context References

`code_index.json` includes every base tracked file plus generated plan/context artifacts. Very small internal files are indexed but not each described in prose.

## Contradictions Found

- `examples/README.md` says `uv sync --dev` for examples, but FLUX default tests require the separate examples dependency group for `onnxscript`/`diffusers`.
- `RUN_MLX_E2E=1` in pytest config makes an environment-heavy audio lane default despite docs describing it as a real demo.

## Must Update During Execution

Any task touching source/tests/docs/public API/models/backends must update `log.md`, `index.md` if page coverage changes, and machine indexes.
