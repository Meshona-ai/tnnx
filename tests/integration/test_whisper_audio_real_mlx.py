from __future__ import annotations

from pathlib import Path


def test_real_whisper_audio_transcription(tmp_path: Path) -> None:
    from examples.whisper_audio.transpile_and_transcribe import run_demo

    result = run_demo("examples/terminator.mp3", out_dir=tmp_path)
    assert Path(str(result["text_file"])).exists()
    assert Path(str(result["json_file"])).exists()
    assert str(result["text"]).strip() != ""
