from __future__ import annotations

from collections.abc import Iterable

import pytest


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
        if "flux" in path:
            markers.add("flux")
        if "qwen" in path:
            markers.add("qwen")
        if "checkpoint" in path or "snapshot" in path:
            markers.update({"expensive", "snapshot"})
        if "RUN_FLUX_E2E" in path or "RUN_QWEN" in path:
            markers.update({"expensive", "network"})
        _add_markers(item, markers)
