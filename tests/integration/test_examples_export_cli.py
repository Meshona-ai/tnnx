from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "args", "onnx_name", "generated_dir", "entrypoint_name"),
    [
        (
            "examples.model_mlp",
            [],
            "tiny_mlp.onnx",
            "generated_tiny_mlp_jax",
            "model_jax.py",
        ),
        (
            "examples.model_whisper_tiny",
            ["--target", "mlx"],
            "whisper_tiny.onnx",
            "generated_whisper_tiny_mlx",
            "model_mlx.py",
        ),
    ],
)
def test_example_model_export_clis_write_to_output_dir(
    module_name: str,
    args: list[str],
    onnx_name: str,
    generated_dir: str,
    entrypoint_name: str,
    tmp_path: Path,
) -> None:
    _ = pytest.importorskip("torch")
    module = __import__(module_name, fromlist=["main"])

    exit_code = module.main([*args, "--output-dir", str(tmp_path)])
    assert exit_code == 0

    assert (tmp_path / onnx_name).exists()
    assert (tmp_path / generated_dir / "graph_ir.json").exists()
    assert (tmp_path / generated_dir / entrypoint_name).exists()
    assert (tmp_path / generated_dir / "weights.npz").exists()
