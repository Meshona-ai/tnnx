from __future__ import annotations

from tnnx.ir.serde import dumps_graph_json, loads_graph_json
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def test_graph_ir_roundtrip_is_stable() -> None:
    ir = GraphIR(
        name="mlp_v0",
        opset=18,
        tensors={
            "x": TensorRef(name="x", dtype="float32", shape=["N", 8], kind="input"),
            "w1": TensorRef(name="w1", dtype="float32", shape=[8, 16], kind="initializer"),
            "y": TensorRef(name="y", dtype="float32", shape=["N", 16], kind="output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="GEMM",
                inputs=["x", "w1"],
                outputs=["y"],
                attrs={"transB": 0},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={"deterministic": True},
    )
    payload_1 = dumps_graph_json(ir)
    payload_2 = dumps_graph_json(ir)

    assert payload_1 == payload_2
    parsed = loads_graph_json(payload_1)
    assert parsed.to_dict() == ir.to_dict()
