# Context Log

## 2026-06-28 - Second-pass audit and context pack

- Base ref: `local-main-initial-review-base`
- Base SHA: `91bf6abf1f392863d0c3cdca692d3272efdceb4b`
- Branch: `codex/tnnx-second-pass-plan`
- Files/pages changed: created `llms.txt`, `docs/llm_context/**`, and `.codex/plans/tnnx_second_pass/**`.
- Reason: second-pass audit, executable improvement plan, and persistent LLM-facing repo wiki.
- Validation performed: ruff check PASS; ruff format PASS; ty PASS; default pytest FAIL with 15 pre-existing failures before examples deps; examples-enabled pytest FAIL with 2 host `ffmpeg` failures; build PASS; clean wheel import PASS on Python 3.14; CLI help PASS; CLI JAX smoke PASS; generated JAX py_compile PASS; ResNet-18 model-zoo smoke PASS; low-level residue grep PASS/no hits.
- Unresolved questions: whether default dev should include examples deps; whether `RUN_MLX_E2E=1` should remain default; whether Python 3.14-only is intentional; which research scripts are retained; whether metadata-only config fields should stay public.
