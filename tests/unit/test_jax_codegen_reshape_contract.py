from __future__ import annotations

import numpy as np
import pytest

from tnnx.codegen.jax_codegen import render_jax_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def _reshape_ir() -> GraphIR:
    return GraphIR(
        name="reshape_zero_copy",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [2, 3, 4], "input"),
            "shape": TensorRef("shape", "int64", [2], "initializer"),
            "y": TensorRef("y", "float32", [2, 12], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="RESHAPE",
                inputs=["x", "shape"],
                outputs=["y"],
                attrs={},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )


def test_jax_codegen_renders_onnx_reshape_helper_for_zero_copy_shapes() -> None:
    source = render_jax_module(_reshape_ir())

    assert "def _onnx_reshape(data, shape, *, allowzero=False):" in source
    assert "if dim == 0:" in source
    assert '_onnx_reshape(tensors["x"], params["shape"], allowzero=False)' in source


def test_jax_codegen_keeps_shape_subgraphs_static_for_jit() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    ir = GraphIR(
        name="expand_with_shape_subgraph",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 4], "input"),
            "shape_len": TensorRef("shape_len", "int64", [1], "initializer"),
            "target": TensorRef("target", "int64", [2], "initializer"),
            "neg_one": TensorRef("neg_one", "int64", [], "initializer"),
            "ones": TensorRef("ones", "int64", [2], "intermediate"),
            "negative": TensorRef("negative", "int64", [2], "intermediate"),
            "mask": TensorRef("mask", "bool", [2], "intermediate"),
            "shape": TensorRef("shape", "int64", [2], "intermediate"),
            "y": TensorRef("y", "float32", [1, 4], "output"),
        },
        nodes=[
            OpNode("n0", "CONSTANT_OF_SHAPE", ["shape_len"], ["ones"], {"value": 1}),
            OpNode("n1", "MUL", ["ones", "neg_one"], ["negative"], {}),
            OpNode("n2", "EQUAL", ["target", "negative"], ["mask"], {}),
            OpNode("n3", "WHERE", ["mask", "ones", "target"], ["shape"], {}),
            OpNode("n4", "EXPAND", ["x", "shape"], ["y"], {}),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    source = render_jax_module(ir)
    namespace: dict[str, object] = {}
    exec(source, namespace)
    params = {
        "shape_len": np.asarray([2], dtype=np.int64),
        "target": np.asarray([1, -1], dtype=np.int64),
        "neg_one": np.asarray(-1, dtype=np.int64),
    }

    result = namespace["forward_jit"](params, {"x": jnp.ones((1, 4), dtype=jnp.float32)})
    jax.block_until_ready(result["y"])

    np.testing.assert_allclose(np.asarray(result["y"]), np.ones((1, 4), dtype=np.float32))
    assert 'tensors["shape"] = np.where' in source
