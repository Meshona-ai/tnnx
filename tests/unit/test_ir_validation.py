from __future__ import annotations

import pytest

from tnnx.ir.schema import validate_graph
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def _base_ir() -> GraphIR:
    return GraphIR(
        name="base",
        opset=18,
        tensors={
            "x": TensorRef(name="x", dtype="float32", shape=[2, 3], kind="input"),
            "y": TensorRef(name="y", dtype="float32", shape=[2, 3], kind="output"),
        },
        nodes=[OpNode(id="n0", op="RELU", inputs=["x"], outputs=["y"], attrs={})],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )


def test_validate_graph_happy_path() -> None:
    validate_graph(_base_ir())


def test_validate_graph_rejects_unknown_attr_type() -> None:
    ir = _base_ir()
    ir.nodes[0].attrs["bad"] = {"nested": "not-supported"}

    with pytest.raises(TypeError, match="unsupported attribute type"):
        validate_graph(ir)


def test_validate_graph_rejects_wrong_input_arity() -> None:
    ir = _base_ir()
    ir.nodes[0].inputs = []

    with pytest.raises(ValueError, match="expects at least"):
        validate_graph(ir)
