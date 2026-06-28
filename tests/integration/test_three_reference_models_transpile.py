from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig


def _write_reference_onnx(kind: str, path: Path) -> None:
    if kind == "mlp":
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        w1 = numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), name="w1")
        b1 = numpy_helper.from_array(np.zeros((4,), dtype=np.float32), name="b1")
        w2 = numpy_helper.from_array(np.ones((4, 2), dtype=np.float32), name="w2")
        b2 = numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="b2")
        nodes = [
            helper.make_node("MatMul", inputs=["x", "w1"], outputs=["h0"]),
            helper.make_node("Add", inputs=["h0", "b1"], outputs=["h1"]),
            helper.make_node("Relu", inputs=["h1"], outputs=["h2"]),
            helper.make_node("MatMul", inputs=["h2", "w2"], outputs=["h3"]),
            helper.make_node("Add", inputs=["h3", "b2"], outputs=["y"]),
        ]
        graph = helper.make_graph(nodes, "mlp_graph", [x], [y], [w1, b1, w2, b2])
        onnx.save(helper.make_model(graph), path)
        return

    if kind == "residual":
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        w = numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), name="w")
        b = numpy_helper.from_array(np.zeros((4,), dtype=np.float32), name="b")
        nodes = [
            helper.make_node("MatMul", inputs=["x", "w"], outputs=["h0"]),
            helper.make_node("Add", inputs=["h0", "b"], outputs=["h1"]),
            helper.make_node("Relu", inputs=["h1"], outputs=["h2"]),
            helper.make_node("Add", inputs=["h2", "x"], outputs=["y"]),
        ]
        graph = helper.make_graph(nodes, "residual_graph", [x], [y], [w, b])
        onnx.save(helper.make_model(graph), path)
        return

    if kind == "indexing":
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 6])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
        idx = numpy_helper.from_array(np.array([0, 2, 4], dtype=np.int64), name="idx")
        starts = numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts")
        ends = numpy_helper.from_array(np.array([3], dtype=np.int64), name="ends")
        axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
        steps = numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps")
        nodes = [
            helper.make_node("Gather", inputs=["x", "idx"], outputs=["g"], axis=1),
            helper.make_node(
                "Slice",
                inputs=["g", "starts", "ends", "axes", "steps"],
                outputs=["y"],
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "indexing_graph",
            [x],
            [y],
            [idx, starts, ends, axes, steps],
        )
        onnx.save(helper.make_model(graph), path)
        return

    raise ValueError(f"Unknown reference model: {kind}")


@pytest.mark.parametrize("kind", ["mlp", "residual", "indexing"])
@pytest.mark.parametrize("target", ["jax", "mlx"])
def test_three_reference_models_transpile_end_to_end(
    kind: str,
    target: str,
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / f"{kind}.onnx"
    _write_reference_onnx(kind, onnx_path)

    out_dir = tmp_path / f"generated_{target}_{kind}"
    transpile_onnx(str(onnx_path), target, str(out_dir), config=CompileConfig())

    assert (out_dir / "graph_ir.json").exists()
    assert (out_dir / "weights.npz").exists()

    if target == "jax":
        assert (out_dir / "model_jax.py").exists()
    elif target == "mlx":
        assert (out_dir / "model_mlx.py").exists()
    else:
        raise ValueError(f"Unknown target: {target}")
