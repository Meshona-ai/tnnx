from __future__ import annotations

from tnnx.codegen.common import order_nodes_for_emission
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def test_topological_emit_order_is_stable() -> None:
    ir = GraphIR(
        name="g",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1], "input"),
            "a": TensorRef("a", "float32", [1], "intermediate"),
            "y": TensorRef("y", "float32", [1], "output"),
        },
        nodes=[
            OpNode(id="n0", op="RELU", inputs=["x"], outputs=["a"], attrs={}),
            OpNode(id="n1", op="RELU", inputs=["a"], outputs=["y"], attrs={}),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    ordered = order_nodes_for_emission(ir)
    assert [n.id for n in ordered] == ["n0", "n1"]
