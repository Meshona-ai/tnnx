from __future__ import annotations

from tnnx.codegen.jax_codegen import render_jax_module
from tnnx.codegen.mlx_codegen import render_mlx_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def _spatial_ir() -> GraphIR:
    return GraphIR(
        name="spatial",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 3, 8, 8], "input"),
            "scale": TensorRef("scale", "float32", [3], "initializer"),
            "bias": TensorRef("bias", "float32", [3], "initializer"),
            "mean": TensorRef("mean", "float32", [3], "initializer"),
            "var": TensorRef("var", "float32", [3], "initializer"),
            "sizes": TensorRef("sizes", "int64", [4], "initializer"),
            "q_scale": TensorRef("q_scale", "float32", [1], "initializer"),
            "zero": TensorRef("zero", "uint8", [1], "initializer"),
            "bn": TensorRef("bn", "float32", [1, 3, 8, 8], "intermediate"),
            "pool": TensorRef("pool", "float32", [1, 3, 4, 4], "intermediate"),
            "up": TensorRef("up", "float32", [1, 3, 8, 8], "intermediate"),
            "flat": TensorRef("flat", "float32", [1, 192], "intermediate"),
            "cast": TensorRef("cast", "float32", [1, 192], "intermediate"),
            "clip": TensorRef("clip", "float32", [1, 192], "intermediate"),
            "q": TensorRef("q", "uint8", [1, 192], "intermediate"),
            "dq": TensorRef("dq", "float32", [1, 192], "intermediate"),
            "y": TensorRef("y", "float32", [1, 192], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="BATCHNORM",
                inputs=["x", "scale", "bias", "mean", "var"],
                outputs=["bn"],
                attrs={"epsilon": 1e-5},
            ),
            OpNode(
                id="n1",
                op="MAXPOOL",
                inputs=["bn"],
                outputs=["pool"],
                attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
            ),
            OpNode(
                id="n2",
                op="UPSAMPLE",
                inputs=["pool", "sizes"],
                outputs=["up"],
                attrs={"mode": "nearest", "input_slots": [0, 3]},
            ),
            OpNode(id="n3", op="FLATTEN", inputs=["up"], outputs=["flat"], attrs={"axis": 1}),
            OpNode(id="n4", op="CAST", inputs=["flat"], outputs=["cast"], attrs={"to": 1}),
            OpNode(
                id="n5",
                op="CLIP",
                inputs=["cast"],
                outputs=["clip"],
                attrs={"min": 0.0, "max": 6.0},
            ),
            OpNode(
                id="n6", op="QUANTIZE", inputs=["clip", "q_scale", "zero"], outputs=["q"], attrs={}
            ),
            OpNode(
                id="n7", op="DEQUANTIZE", inputs=["q", "q_scale", "zero"], outputs=["dq"], attrs={}
            ),
            OpNode(id="n8", op="SIGMOID", inputs=["dq"], outputs=["y"], attrs={}),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )


def _sequence_ir() -> GraphIR:
    return GraphIR(
        name="sequence",
        opset=18,
        tensors={
            "start": TensorRef("start", "int64", [], "initializer"),
            "limit": TensorRef("limit", "int64", [], "initializer"),
            "delta": TensorRef("delta", "int64", [], "initializer"),
            "cond": TensorRef("cond", "bool", [1, 5], "input"),
            "alt": TensorRef("alt", "float32", [1, 5], "input"),
            "scale": TensorRef("scale", "float32", [5], "initializer"),
            "bias": TensorRef("bias", "float32", [5], "initializer"),
            "rng": TensorRef("rng", "float32", [5], "intermediate"),
            "unsq": TensorRef("unsq", "float32", [1, 5], "intermediate"),
            "sel": TensorRef("sel", "float32", [1, 5], "intermediate"),
            "norm": TensorRef("norm", "float32", [1, 5], "intermediate"),
            "sin": TensorRef("sin", "float32", [1, 5], "intermediate"),
            "y": TensorRef("y", "float32", [1, 5], "output"),
        },
        nodes=[
            OpNode(
                id="n0", op="ARANGE", inputs=["start", "limit", "delta"], outputs=["rng"], attrs={}
            ),
            OpNode(id="n1", op="UNSQUEEZE", inputs=["rng"], outputs=["unsq"], attrs={"axes": [0]}),
            OpNode(id="n2", op="WHERE", inputs=["cond", "unsq", "alt"], outputs=["sel"], attrs={}),
            OpNode(
                id="n3",
                op="RMSNORM",
                inputs=["sel", "scale", "bias"],
                outputs=["norm"],
                attrs={"axis": -1, "epsilon": 1e-5},
            ),
            OpNode(id="n4", op="SIN", inputs=["norm"], outputs=["sin"], attrs={}),
            OpNode(id="n5", op="COS", inputs=["sin"], outputs=["y"], attrs={}),
        ],
        inputs=["cond", "alt"],
        outputs=["y"],
        metadata={},
    )


def _pad_ir() -> GraphIR:
    return GraphIR(
        name="pad_only",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 3, 8, 8], "input"),
            "pads": TensorRef("pads", "int64", [8], "initializer"),
            "y": TensorRef("y", "float32", [1, 3, 10, 10], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="PAD",
                inputs=["x", "pads"],
                outputs=["y"],
                attrs={"mode": "constant"},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )


def test_jax_codegen_renders_extended_helpers() -> None:
    spatial = render_jax_module(_spatial_ir())
    sequence = render_jax_module(_sequence_ir())
    padded = render_jax_module(_pad_ir())

    assert "def _onnx_batchnorm" in spatial
    assert "def _onnx_pool" in spatial
    assert "def _onnx_resize_nearest" in spatial
    assert "def _onnx_flatten" in spatial
    assert "def _onnx_cast" in spatial
    assert "def _onnx_clip" in spatial
    assert "def _onnx_quantize" in spatial
    assert "def _onnx_dequantize" in spatial
    assert "def _onnx_pad" in padded
    assert "_onnx_pad(" in padded
    assert "def _onnx_arange" in sequence
    assert "def _onnx_unsqueeze" in sequence
    assert "jnp.where(" in sequence
    assert "jnp.mean(jnp.square" in sequence


def test_mlx_codegen_renders_extended_helpers() -> None:
    spatial = render_mlx_module(_spatial_ir())
    sequence = render_mlx_module(_sequence_ir())
    padded = render_mlx_module(_pad_ir())

    assert "def _onnx_batchnorm" in spatial
    assert "def _onnx_pool" in spatial
    assert "def _onnx_resize_nearest" in spatial
    assert "def _onnx_flatten" in spatial
    assert "def _onnx_cast" in spatial
    assert "def _onnx_clip" in spatial
    assert "def _onnx_quantize" in spatial
    assert "def _onnx_dequantize" in spatial
    assert "def _onnx_pad" in padded
    assert "_onnx_pad(" in padded
    assert "def _onnx_arange" in sequence
    assert "def _onnx_unsqueeze" in sequence
    assert "mx.where(" in sequence
    assert "mx.mean(" in sequence
