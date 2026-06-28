from __future__ import annotations

import json
import sys
from pathlib import Path

import onnx

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig
from tnnx.ingest.op_map import ONNX_TO_SEMANTIC


def test_whisper_audio_source_exports_supported_graph(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.whisper_audio.whisper_hf_tiny_source import (
        DEFAULT_DECODE_TOKENS,
        WhisperTinyForTranspile,
        build_initial_tokens,
        build_padded_tokens,
        export_onnx,
    )

    model, cfg = WhisperTinyForTranspile.from_hf_snapshot()
    mel = model.model.encoder.conv1.weight.new_zeros((1, cfg.num_mel_bins, 3000))
    tokens = build_padded_tokens(
        cfg,
        build_initial_tokens(cfg),
        decode_tokens=DEFAULT_DECODE_TOKENS,
    )

    onnx_path = export_onnx(
        tmp_path / "whisper_tiny.onnx",
        model=model,
        mel=mel,
        tokens=tokens,
    )
    exported = onnx.load(onnx_path)
    supported = set(ONNX_TO_SEMANTIC.keys()) | {"Constant"}
    unsupported = sorted({node.op_type for node in exported.graph.node} - supported)
    assert not unsupported, f"Unsupported ONNX ops: {unsupported}"

    out_dir = tmp_path / "generated_mlx"
    manifest = transpile_onnx(
        str(onnx_path),
        "mlx",
        str(out_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )
    assert (out_dir / "model_mlx.py").exists()
    assert manifest.weights_file.exists()


def test_whisper_audio_transpiled_mlx_e2e_when_enabled(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.whisper_audio.transpile_and_transcribe import run_demo

    result = run_demo("examples/terminator.mp3", out_dir=tmp_path)
    assert Path(str(result["text_file"])).exists()
    assert Path(str(result["json_file"])).exists()
    payload = json.loads(Path(str(result["json_file"])).read_text(encoding="utf-8"))
    assert payload["text"] == result["text"]
