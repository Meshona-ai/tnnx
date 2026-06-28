from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper

from tnnx.ingest.onnx_reader import UnsupportedOpError, load_onnx_to_ir


def test_unsupported_onnx_op_raises_error(tmp_path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Abs", inputs=["x"], outputs=["y"], name="abs0")
    graph = helper.make_graph([node], "unsupported", [x], [y])
    model = helper.make_model(graph, producer_name="test")
    path = str(tmp_path / "bad.onnx")
    onnx.save(model, path)

    with pytest.raises(UnsupportedOpError, match="Abs"):
        load_onnx_to_ir(path)
