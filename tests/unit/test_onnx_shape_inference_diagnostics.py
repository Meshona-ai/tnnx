from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from tnnx.ingest.onnx_reader import load_onnx_to_ir


def _identity_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "shape_diag", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, path)


def _model_with_external_initializer(path: Path, location: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    weight = TensorProto()
    weight.name = "w"
    weight.data_type = TensorProto.FLOAT
    weight.dims.extend([1])
    weight.data_location = TensorProto.EXTERNAL
    location_entry = weight.external_data.add()
    location_entry.key = "location"
    location_entry.value = location
    node = helper.make_node("Add", ["x", "w"], ["y"])
    graph = helper.make_graph([node], "external_data", [x], [y], initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, path)


def test_shape_inference_failure_is_recorded(monkeypatch, tmp_path: Path) -> None:
    onnx_path = tmp_path / "identity.onnx"
    _identity_model(onnx_path)

    def fail_infer_shapes(model):
        raise RuntimeError("boom")

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", fail_infer_shapes)

    ir, _ = load_onnx_to_ir(onnx_path)

    assert ir.metadata["shape_inference"] == "failed"
    assert "RuntimeError: boom" in ir.metadata["shape_inference_error"]


def test_shape_inference_disabled_is_recorded(tmp_path: Path) -> None:
    onnx_path = tmp_path / "identity.onnx"
    _identity_model(onnx_path)

    ir, _ = load_onnx_to_ir(onnx_path, infer_shapes=False)

    assert ir.metadata["shape_inference"] == "disabled"


def test_external_data_path_escape_is_rejected(tmp_path: Path) -> None:
    onnx_path = tmp_path / "external.onnx"
    _model_with_external_initializer(onnx_path, "../secret.bin")

    with pytest.raises(ValueError, match="outside the model directory"):
        load_onnx_to_ir(onnx_path)
