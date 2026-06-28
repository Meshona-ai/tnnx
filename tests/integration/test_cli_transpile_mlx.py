from __future__ import annotations

from pathlib import Path
from subprocess import run

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _write_simple_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.ones((4, 3), dtype=np.float32), name="w")
    node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"])
    model = helper.make_model(helper.make_graph([node], "cli_graph", [x], [y], [w]))
    onnx.save(model, path)


def test_cli_transpile_mlx(tmp_path: str) -> None:
    onnx_path = Path(tmp_path) / "model.onnx"
    out_dir = Path(tmp_path) / "generated_mlx"
    _write_simple_onnx(onnx_path)
    result = run(
        [
            "uv",
            "run",
            "tnnx",
            "transpile",
            "--onnx",
            str(onnx_path),
            "--target",
            "mlx",
            "--out",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "model_mlx.py").exists()
    assert (out_dir / "weights.npz").exists()


def test_cli_transpile_mlx_no_graph_ir(tmp_path: str) -> None:
    onnx_path = Path(tmp_path) / "model.onnx"
    out_dir = Path(tmp_path) / "generated_mlx_no_graph_ir"
    _write_simple_onnx(onnx_path)
    result = run(
        [
            "uv",
            "run",
            "tnnx",
            "transpile",
            "--onnx",
            str(onnx_path),
            "--target",
            "mlx",
            "--out",
            str(out_dir),
            "--no-graph-ir",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (out_dir / "graph_ir.json").exists()
    assert (out_dir / "model_mlx.py").exists()
    assert (out_dir / "weights.npz").exists()
