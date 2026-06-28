from __future__ import annotations

from tnnx.codegen.jax_codegen import render_jax_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def test_jax_codegen_renders_bfloat16_cast_map_entry() -> None:
    ir = GraphIR(
        name="bfloat16_cast",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 4], "input"),
            "x_bf16": TensorRef("x_bf16", "bfloat16", [1, 4], "intermediate"),
            "y": TensorRef("y", "float32", [1, 4], "output"),
        },
        nodes=[
            OpNode(id="n0", op="CAST", inputs=["x"], outputs=["x_bf16"], attrs={"to": 16}),
            OpNode(id="n1", op="CAST", inputs=["x_bf16"], outputs=["y"], attrs={"to": 1}),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )

    rendered = render_jax_module(ir)

    assert "16: jnp.bfloat16" in rendered
    assert "_onnx_cast(" in rendered
