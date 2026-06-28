from __future__ import annotations

import os
from pathlib import Path


def assert_snapshot(snapshot_name: str, actual: str, *, test_file: str) -> None:
    snapshot_path = Path(test_file).resolve().parent / "expected" / snapshot_name
    update = os.getenv("UPDATE_SNAPSHOTS", "").lower() in {"1", "true", "yes"}
    if update:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(actual, encoding="utf-8")

    assert snapshot_path.exists(), (
        f"Snapshot missing: {snapshot_path}. Run with UPDATE_SNAPSHOTS=1 to generate snapshots."
    )
    expected = snapshot_path.read_text(encoding="utf-8")
    assert actual.rstrip("\n") == expected.rstrip("\n")
