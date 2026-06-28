from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig


def _write_simple_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.ones((4, 3), dtype=np.float32), name="w")
    node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"])
    model = helper.make_model(helper.make_graph([node], "determinism_graph", [x], [y], [w]))
    onnx.save(model, path)


@pytest.mark.parametrize(
    ("target", "cfg", "artifacts"),
    [
        ("jax", CompileConfig(), ["graph_ir.json", "compile_metadata.json", "model_jax.py"]),
        ("jax", CompileConfig(emit_graph_ir=False), ["compile_metadata.json", "model_jax.py"]),
        ("mlx", CompileConfig(), ["graph_ir.json", "compile_metadata.json", "model_mlx.py"]),
        ("mlx", CompileConfig(emit_graph_ir=False), ["compile_metadata.json", "model_mlx.py"]),
    ],
)
def test_transpile_artifacts_are_deterministic(
    target: str,
    cfg: CompileConfig,
    artifacts: list[str],
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    _write_simple_onnx(onnx_path)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    manifest_a = transpile_onnx(str(onnx_path), target, str(out_a), config=cfg)
    manifest_b = transpile_onnx(str(onnx_path), target, str(out_b), config=cfg)

    assert [path.name for path in manifest_a.files] == [path.name for path in manifest_b.files]
    assert manifest_a.entrypoint == manifest_b.entrypoint
    assert manifest_a.weights_file is not None
    assert manifest_b.weights_file is not None
    assert manifest_a.weights_file.name == manifest_b.weights_file.name

    for artifact in artifacts:
        left = (out_a / artifact).read_text(encoding="utf-8")
        right = (out_b / artifact).read_text(encoding="utf-8")
        assert left == right

    if not cfg.emit_graph_ir:
        assert not (out_a / "graph_ir.json").exists()
        assert not (out_b / "graph_ir.json").exists()
