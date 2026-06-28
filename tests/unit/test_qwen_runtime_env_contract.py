from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _qwen_jax_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.qwen import infer_qwen3_5_from_transformers_jax as qwen_jax

    return qwen_jax


def test_qwen_prebuilt_snapshot_honors_local_files_only(monkeypatch: pytest.MonkeyPatch) -> None:
    qwen_jax = _qwen_jax_module()
    captured: dict[str, object] = {}

    def _fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/tmp/qwen-snapshot"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)

    snapshot = qwen_jax._resolve_prebuilt_snapshot(local_files_only=True)

    assert snapshot == Path("/tmp/qwen-snapshot")
    assert captured["repo_id"] == "onnx-community/Qwen3.5-0.8B-ONNX"
    assert captured["local_files_only"] is True
    assert "onnx/decoder_model_merged_fp16.onnx" in captured["allow_patterns"]
