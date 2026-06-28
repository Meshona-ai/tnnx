# Context Pack Lint

## Orphan Pages

None. `index.md` lists every context-pack page generated in this pass.

## Stale Claims

None known after final refresh. `scripts/check_context_pack.py`, `scripts/check_operator_docs.py`, and `scripts/check_docs_links.py` are the reusable drift gates.

## Unresolved References

- Full Qwen and FLUX real lanes still require explicit env gates and assets. Whisper real audio is validated in the default suite on this host.

## Pages Missing Index Entries

None for generated pages.

## Code Paths Missing Context References

`code_index.json` includes tracked and unignored workspace files except temporary `.codex/plans/**` execution artifacts. Very small internal files are indexed but not each described in prose.

## Contradictions Found

- None known.

## Must Update During Execution

Any task touching source/tests/docs/public API/models/backends must update `log.md`, `index.md` if page coverage changes, and machine indexes. Use `uv run python scripts/check_context_pack.py --write`.
