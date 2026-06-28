from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
CODE_MD_RE = re.compile(r"`([^`]+\.md)`")
EXCLUDED_PREFIXES = (".agents/", ".codex/plans/")


def _git_markdown_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        ROOT / line
        for line in result.stdout.splitlines()
        if line and not line.startswith(EXCLUDED_PREFIXES)
    ]


def _target_exists(source: Path, target: str) -> bool:
    target = target.strip()
    if not target or target.startswith("#"):
        return True
    parsed = urlparse(target)
    if parsed.scheme or target.startswith("mailto:"):
        return True
    path_part = unquote(target.split("#", 1)[0])
    if not path_part:
        return True
    candidate = (source.parent / path_part).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return False
    return candidate.exists()


def main() -> int:
    missing: list[str] = []
    for path in _git_markdown_files():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        targets = list(LINK_RE.findall(text))
        if rel in {"README.md", "docs/README.md"}:
            targets.extend(CODE_MD_RE.findall(text))
        for target in targets:
            if not _target_exists(path, target):
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
    if missing:
        print("Missing Markdown targets:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 1
    print("Markdown links resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
