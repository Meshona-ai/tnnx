# Metrics

## Method

Counts are from `git ls-files`, `wc -l`, a Python line counter for text files, `uv build`, and derived AST/index generation at review base `91bf6abf1f392863d0c3cdca692d3272efdceb4b`.

## LOC By Area

| Area | Files | LOC |
| --- | --- | --- |
| src/tnnx | 26 | 4421 |
| tests | 111 | 9079 |
| examples | 44 | 8702 |
| docs before context pack | 3 | 215 |
| scripts | 11 | 3699 |
| .agents | 8 | 634 |
| all tracked | 214 | 29447 |

## Dependencies

- Runtime deps: 2 (`numpy`, `onnx`).
- Dev group: 13.
- Examples group: 3 direct entries (`accelerate`, `diffusers`, `onnxscript`).

## Generated/Large Artifact Inventory

- Snapshot expected generated code under `tests/snapshots/expected`: retained as `GENERATED` test assets.
- `examples/terminator.mp3`: tracked example audio asset, `THIRD-PARTY/NOTICE` verification needed.
- `examples/whisper_audio/assets/mel_filters.npz`: tracked data asset, `THIRD-PARTY/NOTICE` verification needed.
- Ignored local generated/build/cache outputs: `.venv`, `dist`, `.pytest_cache`, `.ruff_cache`, `__pycache__`.

## Estimated Reduction Opportunities

- Research scripts: 0 to 11 files, 0 to 3699 LOC after reachability review.
- Historical FLUX docs: 1 to 2 files, about 100 to 500 LOC after consolidation.
- Unused helpers: 2 to 5 files/symbol groups, about 20 to 80 LOC after public-use review.
- Codegen duplication: about 100 to 500 LOC if shared helpers are extracted without behavior drift.
- Packaging/sdist: up to examples/tests/scripts/agents from sdist if policy excludes them; wheel already package-only.

Estimates are not deletion proof.

## Context-Pack Size

This audit creates 15 Markdown context pages, 3 machine indexes, root `llms.txt`, and 10 plan files. Machine-index entries: 243 file entries, 1118 symbol entries, 1455 graph edges.
