# Testing And Gates

Default validation:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check src
uv run pytest -q
uv build
uv run python scripts/check_package_contents.py
uv run python scripts/check_generated_code.py
uv run python scripts/check_residue.py
uv run python scripts/check_docs_links.py
uv run python scripts/check_operator_docs.py
uv run python scripts/check_context_pack.py
```

`scripts/check.sh` runs the same local gate.

## Pytest Markers

- `expensive`: checkpoint, model-zoo, or long runtime path.
- `ffmpeg`: real audio boundary.
- `flux`: FLUX model family.
- `mlx`: MLX backend/runtime.
- `network`: may require cached or downloadable upstream assets when explicitly enabled.
- `qwen`: Qwen model family.
- `snapshot`: requires a local model snapshot or cached checkpoint assets.

Markers describe tests; they do not skip tests by themselves.

## Environment Gates

- Whisper real audio runs by default when the local `openai/whisper-tiny` snapshot exists; otherwise
  the snapshot-backed tests skip. The loader retries the observed Homebrew x265 dylib mismatch before
  failing with ffmpeg stderr.
- `RUN_FLUX_E2E=1` plus `TNNX_FLUX_SNAPSHOT=/abs/path` enables checkpoint-backed FLUX lanes.
- `RUN_QWEN_JAX_E2E=1` and `RUN_QWEN_MLX_E2E=1` enable full Qwen lanes.
- `--local-files-only` keeps Qwen real paths offline and fails if required assets are not cached.
