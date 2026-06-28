from __future__ import annotations

from pathlib import Path

import pytest


def test_whisper_tiny_example_mlx(tmp_path: Path) -> None:
    _ = pytest.importorskip("torch")
    _ = pytest.importorskip("mlx.core")
    from examples.run_whisper_tiny_mlx import run_demo

    result = run_demo(tmp_path)
    assert Path(str(result["onnx_path"])).exists()
    assert Path(str(result["generated_module"])).exists()
    assert Path(str(result["weights_path"])).exists()
    assert float(result["max_abs"]) < 5e-4
