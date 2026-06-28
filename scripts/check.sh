#!/usr/bin/env bash
set -euo pipefail

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
