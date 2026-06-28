# Execution Log

## Setup

- execution branch: `codex/tnnx-second-pass-exec`
- execution worktree: `/private/tmp/tnnx-second-pass-exec`
- EXECUTION_BASE_REF: `local main`
- EXECUTION_BASE_SHA: `91bf6abf1f392863d0c3cdca692d3272efdceb4b`
- REVIEW_BASE_SHA: `91bf6abf1f392863d0c3cdca692d3272efdceb4b`
- PLAN_COMMIT_SHA: `b1eafa9e631ed94cbb510bda553616c48cdd5f59`
- imported plan/context commit on execution branch: `ef9bfb2`
- drift: none; fetched `origin` successfully, but no `origin/main` ref exists.
- planning worktree status: clean on `codex/tnnx-second-pass-plan`
- execution worktree status before log creation: clean on `codex/tnnx-second-pass-exec`

## Task Status

| Task | Status | Notes |
| --- | --- | --- |
| T00 | PASS | Plan/context commit verified: one commit over review base; only `.codex/plans/tnnx_second_pass/`, `docs/llm_context/`, and `llms.txt`; mandatory context files present; machine indexes parse. |
| T01 | TODO | Not started. |
| T02 | TODO | Not started. |
| T03 | TODO | Not started. |
| T04 | TODO | Not started. |
| T05 | TODO | Not started. |
| T06 | TODO | Not started. |
| T07 | TODO | Not started. |
| T08 | TODO | Not started. |
| T09 | TODO | Not started. |
| T10 | TODO | Not started. |
| T11 | TODO | Not started. |
| T12 | TODO | Not started. |
| T13 | TODO | Not started. |
| T14 | TODO | Not started. |
| T15 | TODO | Not started. |
| T16 | TODO | Not started. |
| T17 | TODO | Not started. |
| T18 | TODO | Not started. |
| T19 | TODO | Not started. |
| T20 | TODO | Not started. |
| T21 | TODO | Not started. |
| T22 | TODO | Not started. |
| T23 | IN-PROGRESS | Context pack imported; future product tasks must update it in the same commit. |

## Commands Run

| Command | Exit | Result |
| --- | ---: | --- |
| `git worktree list` | 0 | Planning worktree found at `/private/tmp/tnnx-second-pass-plan`; no execution worktree existed. |
| `git fetch --prune origin` | 0 | Remote refs fetched; `origin/main` remains absent. |
| `git diff-tree --no-commit-id --name-only -r codex/tnnx-second-pass-plan` | 0 | Plan commit touches only allowed plan/context paths. |
| `git diff --quiet main..codex/tnnx-second-pass-plan -- . ':!/.codex/plans/tnnx_second_pass/**' ':!/docs/llm_context/**' ':!/llms.txt'` | 0 | No product-file drift in plan commit. |
| `uv run python - <<'PY' ... docs/llm_context parse ... PY` | 0 | `code_index.json`: 243 entries; `symbol_index.jsonl`: 1118 entries; `graph_edges.tsv`: 1455 edges. |
| `git worktree add -b codex/tnnx-second-pass-exec /private/tmp/tnnx-second-pass-exec 91bf6abf1f392863d0c3cdca692d3272efdceb4b` | 0 | Created isolated execution worktree and branch. |
| `git cherry-pick b1eafa9e631ed94cbb510bda553616c48cdd5f59` | 0 | Imported plan/context as `ef9bfb2`. |

## Baseline Reconfirmation

- `uv run ruff check .`: PASS, `All checks passed!`.
- `uv run ruff format --check .`: PASS, `171 files already formatted`.
- `uv run ty check src`: PASS, `All checks passed!`.
- `uv run tnnx --help`: PASS, help renders.
- `uv run tnnx transpile --help`: PASS, help renders.
- context-pack parse/link check: PASS, `code_index.json` 243 entries, `symbol_index.jsonl` 1118 entries, `graph_edges.tsv` 1455 edges.
- `uv run pytest -q` before examples group: FAIL as expected, `15 failed, 206 passed, 15 skipped, 86 warnings`; failures are 13 FLUX tests missing `onnxscript` and 2 Whisper ffmpeg failures.
- `uv sync --dev --group examples`: PASS, installed `accelerate`, `diffusers`, `onnx-ir`, `onnxscript`, and metadata dependencies from lock/cache.
- `uv run pytest -q` after examples group: FAIL as expected, `2 failed, 222 passed, 12 skipped, 131 warnings`; both failures are Whisper real-audio MLX tests blocked by host `ffmpeg`.
- `ffmpeg -version`: FAIL/blocked, exit 134, missing `/opt/homebrew/opt/x265/lib/libx265.215.dylib`.
- `uv build`: PASS, built `dist/tnnx-0.1.0.tar.gz` and `dist/tnnx-0.1.0-py3-none-any.whl`.
- clean wheel import from `/private/tmp`: PASS, `tnnx.__version__ == 0.1.0`.
- tiny ONNX CLI JAX transpile: PASS, wrote `weights.npz`, `compile_metadata.json`, `model_jax.py`, and `graph_ir.json`.
- generated JAX `py_compile`: PASS.
- `uv run python -m examples.model_zoo.transpile_smoke --target jax --model ResNet-18`: PASS, ONNX export and transpile succeeded.
- low-level path/extension grep: PASS/no hits.
- low-level terminology/backend-claim greps: only plan/context/`llms.txt` support-boundary wording; no product/config residue.

Baseline conclusion: plan evidence is current on the execution branch. T01 and T02 remain required to make default validation pass without installing examples manually or requiring a healthy host `ffmpeg`.
