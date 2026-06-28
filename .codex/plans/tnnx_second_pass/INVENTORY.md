# Inventory

Every tracked review-base file plus generated plan/context artifact is classified in `FILE_CLASSIFICATION.tsv`.

Disposition meanings: KEEP, FIX, SHRINK, DELETE-DEAD, DELETE-STALE, VERIFY, GENERATED, THIRD-PARTY/NOTICE, LLM-CONTEXT.

## Summary

| Disposition | Count |
| --- | --- |
| FIX | 36 |
| GENERATED | 10 |
| KEEP | 130 |
| LLM-CONTEXT | 29 |
| SHRINK | 3 |
| THIRD-PARTY/NOTICE | 2 |
| VERIFY | 33 |

## Notes

- `VERIFY` means do not delete without direct and transitive reachability checks.
- `GENERATED` currently applies to snapshot expected generated code.
- `THIRD-PARTY/NOTICE` applies to checked-in data/audio assets needing license/provenance verification.
- `LLM-CONTEXT` applies to this plan and context pack.

See `FILE_CLASSIFICATION.tsv` for per-path owner, task links, confidence, and context page links.
