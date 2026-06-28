from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PREFIXES = (".codex/plans/", "docs/llm_context/")
EXCLUDED_FILES = {"llms.txt", "scripts/check_residue.py"}
PATH_RE = re.compile(
    r"(^|/)(api-server|cmake|fpga|frontend|hls|rtl|web)(/|$)"
    r"|(\.(c|cc|cpp|cu|go|h|hpp|mm|rs)$)"
    r"|(^|/)CMakeLists\.txt$",
    re.IGNORECASE,
)
TEXT_RE = re.compile(
    r"\b("
    r"FPGA|HLS|RTL|Verilog|VHDL|SystemVerilog|CMake|"
    r"native backend|native runtime|web server|API server"
    r")\b",
    re.IGNORECASE,
)


def _git_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def _is_excluded(path: str) -> bool:
    return path in EXCLUDED_FILES or path.startswith(EXCLUDED_PREFIXES)


def main() -> int:
    hits: list[str] = []
    for rel in _git_files():
        if _is_excluded(rel):
            continue
        if PATH_RE.search(rel):
            hits.append(f"path:{rel}")
            continue
        path = ROOT / rel
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in TEXT_RE.finditer(text):
            hits.append(f"text:{rel}:{match.start()}:{match.group(0)}")
    if hits:
        print("Retired low-level/web residue found:", file=sys.stderr)
        for hit in hits:
            print(f"  {hit}", file=sys.stderr)
        return 1
    print("No retired low-level/web residue found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
