from __future__ import annotations

# ruff: noqa: E402,I001

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tnnx_evidence import main


if __name__ == "__main__":
    raise SystemExit(main())
