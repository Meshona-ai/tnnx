from __future__ import annotations

from tnnx.codegen.jax_codegen import render_jax_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def test_jax_codegen_renders_instance_norm_helper() -> None:
    ir = GraphIR(
        name="instance_norm",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 2, 4, 4], "input"),
            "scale": TensorRef("scale", "float32", [2], "initializer"),
            "bias": TensorRef("bias", "float32", [2], "initializer"),
            "y": TensorRef("y", "float32", [1, 2, 4, 4], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="INSTANCENORM",
                inputs=["x", "scale", "bias"],
                outputs=["y"],
                attrs={"epsilon": 1e-5},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )

    rendered = render_jax_module(ir)

    assert "def _onnx_instancenorm" in rendered
    assert "_onnx_instancenorm(" in rendered
