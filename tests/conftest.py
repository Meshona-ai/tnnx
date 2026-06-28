from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

WHISPER_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--whisper-tiny"


def _missing_whisper_snapshot_reason() -> str | None:
    ref_path = WHISPER_CACHE_ROOT / "refs" / "main"
    if not ref_path.exists():
        return f"Missing Whisper snapshot ref: {ref_path}"

    snapshot = WHISPER_CACHE_ROOT / "snapshots" / ref_path.read_text(encoding="utf-8").strip()
    if not snapshot.exists():
        return f"Missing Whisper snapshot directory: {snapshot}"
    return None


def _add_markers(item: pytest.Item, names: Iterable[str]) -> None:
    for name in names:
        item.add_marker(getattr(pytest.mark, name))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = item.path.as_posix()
        markers: set[str] = set()
        if "mlx" in path:
            markers.add("mlx")
        if "whisper_audio" in path:
            markers.update({"ffmpeg", "mlx"})
        if "tests/integration/test_whisper_audio" in path:
            markers.add("snapshot")
            if reason := _missing_whisper_snapshot_reason():
                item.add_marker(pytest.mark.skip(reason=reason))
        if "flux" in path:
            markers.add("flux")
        if "qwen" in path:
            markers.add("qwen")
        if "checkpoint" in path or "snapshot" in path:
            markers.update({"expensive", "snapshot"})
        if "RUN_FLUX_E2E" in path or "RUN_QWEN" in path:
            markers.update({"expensive", "network"})
        _add_markers(item, markers)
