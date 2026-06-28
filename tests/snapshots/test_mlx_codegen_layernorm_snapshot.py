from __future__ import annotations

from tnnx.codegen.mlx_codegen import render_mlx_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef

from .snapshot_utils import assert_snapshot


def test_mlx_codegen_layernorm_snapshot() -> None:
    ir = GraphIR(
        name="ln",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [2, 4], "input"),
            "scale": TensorRef("scale", "float32", [4], "initializer"),
            "bias": TensorRef("bias", "float32", [4], "initializer"),
            "y": TensorRef("y", "float32", [2, 4], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="LAYERNORM",
                inputs=["x", "scale", "bias"],
                outputs=["y"],
                attrs={"axis": -1, "epsilon": 1e-5},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    code = render_mlx_module(ir)
    assert_snapshot("mlx_layernorm.py", code, test_file=__file__)
