from __future__ import annotations

import importlib.metadata as metadata
import sys

from tests import conftest


def test_python_version_is_314() -> None:
    assert sys.version_info[:2] == (3, 14)


def test_package_is_installable() -> None:
    assert metadata.version("tnnx") == "0.1.0"


def test_package_imports() -> None:
    import tnnx

    assert hasattr(tnnx, "__version__")


def test_whisper_audio_snapshot_gate_reports_missing_cache(monkeypatch, tmp_path) -> None:
    cache_root = tmp_path / "whisper-tiny"
    monkeypatch.setattr(conftest, "WHISPER_CACHE_ROOT", cache_root)

    assert conftest._missing_whisper_snapshot_reason() == (
        f"Missing Whisper snapshot ref: {cache_root / 'refs' / 'main'}"
    )
