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


def test_validate_graph_rejects_duplicate_producers() -> None:
    ir = _base_ir()
    ir.tensors["z"] = TensorRef(name="z", dtype="float32", shape=[2, 3], kind="intermediate")
    ir.nodes = [
        OpNode(id="n0", op="RELU", inputs=["x"], outputs=["z"], attrs={}),
        OpNode(id="n1", op="RELU", inputs=["x"], outputs=["z"], attrs={}),
        OpNode(id="n2", op="RELU", inputs=["z"], outputs=["y"], attrs={}),
    ]

    with pytest.raises(ValueError, match="produced by both"):
        validate_graph(ir)


def test_validate_graph_accepts_out_of_order_nodes_with_provenance() -> None:
    ir = _base_ir()
    ir.tensors["z"] = TensorRef(name="z", dtype="float32", shape=[2, 3], kind="intermediate")
    ir.nodes = [
        OpNode(id="n1", op="RELU", inputs=["z"], outputs=["y"], attrs={}),
        OpNode(id="n0", op="RELU", inputs=["x"], outputs=["z"], attrs={}),
    ]

    validate_graph(ir)


def test_validate_graph_rejects_input_without_provenance() -> None:
    ir = _base_ir()
    ir.tensors["z"] = TensorRef(name="z", dtype="float32", shape=[2, 3], kind="intermediate")
    ir.nodes = [OpNode(id="n0", op="RELU", inputs=["z"], outputs=["y"], attrs={})]

    with pytest.raises(ValueError, match="has no producer"):
        validate_graph(ir)


def test_validate_graph_rejects_output_without_provenance() -> None:
    ir = _base_ir()
    ir.nodes = []

    with pytest.raises(ValueError, match="has no producer"):
        validate_graph(ir)
