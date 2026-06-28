from __future__ import annotations

from tnnx.codegen.jax_codegen import render_jax_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def _out_of_order_split_ir() -> GraphIR:
    return GraphIR(
        name="out_of_order_split",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 2, 2], "input"),
            "split_a": TensorRef("split_a", "float32", [1, 1, 2], "intermediate"),
            "split_b": TensorRef("split_b", "float32", [1, 1, 2], "intermediate"),
            "y": TensorRef("y", "float32", [1, 1, 2], "output"),
        },
        nodes=[
            OpNode(
                id="consume",
                op="IDENTITY",
                inputs=["split_a"],
                outputs=["y"],
                attrs={},
            ),
            OpNode(
                id="produce",
                op="SPLIT",
                inputs=["x"],
                outputs=["split_a", "split_b"],
                attrs={"axis": 1, "num_outputs": 2},
            ),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )


def test_jax_codegen_schedules_nodes_before_emitting_code() -> None:
    source = render_jax_module(_out_of_order_split_ir())

    split_index = source.index(
        '_node_produce = _onnx_split(tensors["x"], None, axis=1, num_outputs=2)'
    )
    consume_index = source.index('tensors["y"] = tensors["split_a"]')

    assert split_index < consume_index
