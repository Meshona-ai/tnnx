from __future__ import annotations

import pytest

from tnnx.codegen.mlx_codegen import render_mlx_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def _conv_ir(*, pads: list[int]) -> GraphIR:
    return GraphIR(
        name="conv",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 1, 4, 4], "input"),
            "w": TensorRef("w", "float32", [1, 1, 2, 2], "initializer"),
            "b": TensorRef("b", "float32", [1], "initializer"),
            "y": TensorRef("y", "float32", [1, 1, 5, 5], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="CONV2D",
                inputs=["x", "w", "b"],
                outputs=["y"],
                attrs={"strides": [1, 1], "pads": pads},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )


def test_mlx_conv_codegen_accepts_symmetric_pads() -> None:
    code = render_mlx_module(_conv_ir(pads=[1, 2, 1, 2]))
    assert "padding=(1, 2)" in code


def test_mlx_conv_codegen_rejects_asymmetric_pads() -> None:
    with pytest.raises(ValueError, match="symmetric ONNX pads"):
        _ = render_mlx_module(_conv_ir(pads=[1, 2, 0, 2]))
