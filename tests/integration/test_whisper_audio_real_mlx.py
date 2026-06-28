from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_real_whisper_audio_transcription(tmp_path: Path) -> None:
    if os.getenv("RUN_MLX_E2E", "0") != "1":
        pytest.skip("Set RUN_MLX_E2E=1 to run the full MLX Whisper transpile+decode path.")

    from examples.whisper_audio.transpile_and_transcribe import run_demo

    result = run_demo("examples/terminator.mp3", out_dir=tmp_path)
    assert Path(str(result["text_file"])).exists()
    assert Path(str(result["json_file"])).exists()
    assert str(result["text"]).strip() != ""
