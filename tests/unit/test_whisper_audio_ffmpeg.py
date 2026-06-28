from __future__ import annotations

import sys
from pathlib import Path
from subprocess import CalledProcessError

import pytest


def _source_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.whisper_audio import whisper_hf_tiny_source as source

    return source


def test_ffmpeg_retry_env_uses_installed_legacy_x265(tmp_path: Path, monkeypatch) -> None:
    source = _source_module()
    lib_dir = tmp_path / "x265" / "4.1" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libx265.215.dylib").touch()
    monkeypatch.setattr(source, "_X265_CELLARS", (tmp_path / "x265",))
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/existing")

    env = source._ffmpeg_retry_env(
        "Library not loaded: /opt/homebrew/opt/x265/lib/libx265.215.dylib"
    )

    assert env is not None
    assert env["DYLD_LIBRARY_PATH"].split(":")[:2] == [str(lib_dir), "/existing"]


def test_run_ffmpeg_error_includes_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source_module()

    def fail_run(*args, **kwargs):
        raise CalledProcessError(1, args[0], stderr=b"ffmpeg exploded")

    monkeypatch.setattr(source, "run", fail_run)

    with pytest.raises(RuntimeError, match="ffmpeg exploded"):
        source._run_ffmpeg(["ffmpeg", "-version"])
