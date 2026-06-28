from __future__ import annotations

from tnnx.ir.types import GraphIR, OpNode, TensorRef
from tnnx.passes.prune import prune_dead_nodes


def test_prune_dead_nodes_removes_unreachable_nodes() -> None:
    ir = GraphIR(
        name="prune",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1], "input"),
            "live": TensorRef("live", "float32", [1], "intermediate"),
            "dead": TensorRef("dead", "float32", [1], "intermediate"),
            "y": TensorRef("y", "float32", [1], "output"),
        },
        nodes=[
            OpNode(id="live0", op="RELU", inputs=["x"], outputs=["live"], attrs={}),
            OpNode(id="dead0", op="RELU", inputs=["x"], outputs=["dead"], attrs={}),
            OpNode(id="live1", op="RELU", inputs=["live"], outputs=["y"], attrs={}),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )

    out = prune_dead_nodes(ir)

    assert [node.id for node in out.nodes] == ["live0", "live1"]
    assert "dead" not in out.tensors


def test_prune_dead_nodes_keeps_graph_input_output_passthrough() -> None:
    ir = GraphIR(
        name="passthrough",
        opset=18,
        tensors={"x": TensorRef("x", "float32", [1], "input")},
        nodes=[],
        inputs=["x"],
        outputs=["x"],
        metadata={},
    )

    out = prune_dead_nodes(ir)

    assert out.nodes == []
    assert list(out.tensors) == ["x"]
