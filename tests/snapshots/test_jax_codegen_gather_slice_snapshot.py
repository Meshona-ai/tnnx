from __future__ import annotations

from tnnx.codegen.jax_codegen import render_jax_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef

from .snapshot_utils import assert_snapshot


def test_jax_codegen_gather_slice_snapshot() -> None:
    ir = GraphIR(
        name="gather_slice",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [4, 6], "input"),
            "indices": TensorRef("indices", "int64", [3], "initializer"),
            "starts": TensorRef("starts", "int64", [1], "initializer"),
            "ends": TensorRef("ends", "int64", [1], "initializer"),
            "axes": TensorRef("axes", "int64", [1], "initializer"),
            "steps": TensorRef("steps", "int64", [1], "initializer"),
            "g": TensorRef("g", "float32", [4, 3], "intermediate"),
            "y": TensorRef("y", "float32", [4, 2], "output"),
        },
        nodes=[
            OpNode(id="n0", op="GATHER", inputs=["x", "indices"], outputs=["g"], attrs={"axis": 1}),
            OpNode(
                id="n1",
                op="SLICE",
                inputs=["g", "starts", "ends", "axes", "steps"],
                outputs=["y"],
                attrs={},
            ),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    code = render_jax_module(ir)
    assert_snapshot("jax_gather_slice.py", code, test_file=__file__)
