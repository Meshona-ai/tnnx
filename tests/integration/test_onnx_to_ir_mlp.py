from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tnnx.ingest.onnx_reader import load_onnx_to_ir


def test_onnx_to_ir_mlp(tmp_path: str) -> None:
    w = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3), name="w")
    b = numpy_helper.from_array(np.ones(3, dtype=np.float32), name="b")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    gemm = helper.make_node("Gemm", inputs=["x", "w", "b"], outputs=["h"], name="gemm0")
    relu = helper.make_node("Relu", inputs=["h"], outputs=["y"], name="relu0")
    graph = helper.make_graph([gemm, relu], "mlp", [x], [y], initializer=[w, b])
    model = helper.make_model(graph, producer_name="test")
    path = str(tmp_path / "mlp.onnx")
    onnx.save(model, path)

    ir, weights = load_onnx_to_ir(path)
    assert ir.name == "mlp"
    assert ir.inputs == ["x"]
    assert ir.outputs == ["y"]
    assert [node.op for node in ir.nodes] == ["GEMM", "RELU"]
    assert sorted(weights.keys()) == ["b", "w"]
