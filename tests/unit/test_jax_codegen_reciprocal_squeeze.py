from __future__ import annotations

from tnnx.codegen.jax_codegen import render_jax_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def test_jax_codegen_renders_reciprocal_and_squeeze() -> None:
    ir = GraphIR(
        name="reciprocal_squeeze",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 1, 4], "input"),
            "axes": TensorRef("axes", "int64", [1], "initializer"),
            "sq": TensorRef("sq", "float32", [1, 4], "intermediate"),
            "y": TensorRef("y", "float32", [1, 4], "output"),
        },
        nodes=[
            OpNode(id="n0", op="SQUEEZE", inputs=["x", "axes"], outputs=["sq"], attrs={}),
            OpNode(id="n1", op="RECIPROCAL", inputs=["sq"], outputs=["y"], attrs={}),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )

    rendered = render_jax_module(ir)

    assert "def _onnx_squeeze" in rendered
    assert "_onnx_squeeze(" in rendered
    assert "jnp.reciprocal(" in rendered
