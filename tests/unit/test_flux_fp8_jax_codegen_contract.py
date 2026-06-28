from __future__ import annotations

from onnx import TensorProto

from tnnx.codegen.jax_codegen import render_jax_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def _fp8_cast_ir() -> GraphIR:
    return GraphIR(
        name="fp8_cast",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 4], "input"),
            "x_fp8": TensorRef("x_fp8", "float8_e4m3fn", [1, 4], "intermediate"),
            "x_back": TensorRef("x_back", "float32", [1, 4], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="CAST",
                inputs=["x"],
                outputs=["x_fp8"],
                attrs={"to": int(TensorProto.FLOAT8E4M3FN)},
            ),
            OpNode(
                id="n1",
                op="CAST",
                inputs=["x_fp8"],
                outputs=["x_back"],
                attrs={"to": int(TensorProto.FLOAT)},
            ),
        ],
        inputs=["x"],
        outputs=["x_back"],
        metadata={},
    )


def test_jax_codegen_renders_float8_cast_map_entries() -> None:
    source = render_jax_module(_fp8_cast_ir())

    assert "(17, 'float8_e4m3fn')" in source
    assert "(18, 'float8_e4m3fnuz')" in source
    assert "(19, 'float8_e5m2')" in source
    assert "(20, 'float8_e5m2fnuz')" in source
    assert "(24, 'float8_e8m0fnu')" in source
    assert "getattr(jnp, _jnp_dtype_name, None)" in source
    assert "_ONNX_DTYPE_MAP[_onnx_dtype] = _jnp_dtype" in source
    assert '_onnx_cast(tensors["x"], 17)' in source
    assert '_onnx_cast(tensors["x_fp8"], 1)' in source
