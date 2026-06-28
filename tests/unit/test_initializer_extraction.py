from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from tnnx.ingest.onnx_reader import extract_initializers


def test_extract_initializers() -> None:
    w = numpy_helper.from_array(np.array([[1.0, 2.0]], dtype=np.float32), name="w")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"])
    graph = helper.make_graph([node], "g", [x], [y], initializer=[w])
    model = helper.make_model(graph, producer_name="test")

    extracted = extract_initializers(model)
    assert list(extracted.keys()) == ["w"]
    assert extracted["w"].dtype == np.float32
