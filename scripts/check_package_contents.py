from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
SDIST_DENY = (
    ".agents/",
    ".codex/",
    "examples/",
    "scripts/",
    "tests/",
)
WHEEL_ALLOW_PREFIXES = ("tnnx/", "tnnx-")


def _latest(pattern: str) -> Path:
    matches = sorted(DIST.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"Missing dist artifact matching {pattern}; run uv build first.")
    return matches[-1]


def _strip_root(name: str) -> str:
    parts = name.split("/", 1)
    return parts[1] if len(parts) == 2 else name


def main() -> int:
    try:
        sdist = _latest("tnnx-*.tar.gz")
        wheel = _latest("tnnx-*.whl")
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = [_strip_root(member.name) for member in archive.getmembers()]
    bad_sdist = [
        name for name in sdist_names if any(name.startswith(prefix) for prefix in SDIST_DENY)
    ]

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
    bad_wheel = [name for name in wheel_names if not name.startswith(WHEEL_ALLOW_PREFIXES)]

    if bad_sdist or bad_wheel:
        if bad_sdist:
            print("Unexpected sdist entries:", file=sys.stderr)
            for name in bad_sdist[:40]:
                print(f"  {name}", file=sys.stderr)
        if bad_wheel:
            print("Unexpected wheel entries:", file=sys.stderr)
            for name in bad_wheel[:40]:
                print(f"  {name}", file=sys.stderr)
        return 1
    print("Package contents match policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
