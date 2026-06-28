from __future__ import annotations

from tnnx.codegen.mlx_codegen import render_mlx_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef

from .snapshot_utils import assert_snapshot


def test_mlx_codegen_snapshot() -> None:
    ir = GraphIR(
        name="mlp",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 4], "input"),
            "w": TensorRef("w", "float32", [4, 3], "initializer"),
            "b": TensorRef("b", "float32", [3], "initializer"),
            "y": TensorRef("y", "float32", [1, 3], "output"),
        },
        nodes=[OpNode(id="n0", op="GEMM", inputs=["x", "w", "b"], outputs=["y"], attrs={})],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )

    code = render_mlx_module(ir)
    assert_snapshot("mlx_mlp.py", code, test_file=__file__)


def test_mlx_codegen_qwen_ops_snapshot() -> None:
    ir = GraphIR(
        name="qwen_ops",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [3, 3], "input"),
            "axis": TensorRef("axis", "int64", [], "initializer"),
            "mod_rhs": TensorRef("mod_rhs", "float32", [1], "initializer"),
            "reduce_axes": TensorRef("reduce_axes", "int64", [1], "initializer"),
            "diag": TensorRef("diag", "int64", [], "initializer"),
            "indices": TensorRef("indices", "int64", [2, 1], "initializer"),
            "updates": TensorRef("updates", "float32", [2, 3], "initializer"),
            "soft": TensorRef("soft", "float32", [3, 3], "intermediate"),
            "cum": TensorRef("cum", "float32", [3, 3], "intermediate"),
            "mod": TensorRef("mod", "float32", [3, 3], "intermediate"),
            "sum_unused": TensorRef("sum_unused", "float32", [3, 1], "intermediate"),
            "tri": TensorRef("tri", "float32", [3, 3], "intermediate"),
            "y": TensorRef("y", "float32", [3, 3], "output"),
        },
        nodes=[
            OpNode(id="n0", op="SOFTPLUS", inputs=["x"], outputs=["soft"], attrs={}),
            OpNode(
                id="n1",
                op="CUMSUM",
                inputs=["soft", "axis"],
                outputs=["cum"],
                attrs={"exclusive": 1, "reverse": 1},
            ),
            OpNode(
                id="n2",
                op="MOD",
                inputs=["cum", "mod_rhs"],
                outputs=["mod"],
                attrs={"fmod": 1},
            ),
            OpNode(
                id="n3",
                op="REDUCESUM",
                inputs=["mod", "reduce_axes"],
                outputs=["sum_unused"],
                attrs={"keepdims": 1},
            ),
            OpNode(
                id="n4",
                op="TRILU",
                inputs=["mod", "diag"],
                outputs=["tri"],
                attrs={"upper": 0},
            ),
            OpNode(
                id="n5",
                op="SCATTERND",
                inputs=["tri", "indices", "updates"],
                outputs=["y"],
                attrs={},
            ),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )

    code = render_mlx_module(ir)
    assert_snapshot("mlx_qwen_ops.py", code, test_file=__file__)


def test_mlx_codegen_qwen_fused_decoder_ops_snapshot() -> None:
    ir = GraphIR(
        name="qwen_fused_decoder_ops",
        opset=18,
        tensors={
            "rot_in": TensorRef("rot_in", "float32", [1, 2, 1, 4], "input"),
            "pos": TensorRef("pos", "int64", [1, 1], "input"),
            "cos_cache": TensorRef("cos_cache", "float32", [8, 2], "initializer"),
            "sin_cache": TensorRef("sin_cache", "float32", [8, 2], "initializer"),
            "skip": TensorRef("skip", "float32", [1, 2, 1, 4], "input"),
            "gamma": TensorRef("gamma", "float32", [4], "initializer"),
            "q": TensorRef("q", "float32", [1, 1, 4], "input"),
            "k": TensorRef("k", "float32", [1, 1, 2], "input"),
            "v": TensorRef("v", "float32", [1, 1, 2], "input"),
            "past_k": TensorRef("past_k", "float32", [1, 1, 0, 2], "input"),
            "past_v": TensorRef("past_v", "float32", [1, 1, 0, 2], "input"),
            "bias": TensorRef("bias", "float32", [1, 1, 1, 1], "input"),
            "rot_out": TensorRef("rot_out", "float32", [1, 2, 1, 4], "intermediate"),
            "skip_norm": TensorRef("skip_norm", "float32", [1, 2, 1, 4], "output"),
            "skip_residual": TensorRef("skip_residual", "float32", [1, 2, 1, 4], "intermediate"),
            "attn_out": TensorRef("attn_out", "float32", [1, 1, 4], "output"),
            "present_k": TensorRef("present_k", "float32", [1, 1, 1, 2], "output"),
            "present_v": TensorRef("present_v", "float32", [1, 1, 1, 2], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="ROTARYEMBEDDING",
                inputs=["rot_in", "pos", "cos_cache", "sin_cache"],
                outputs=["rot_out"],
                attrs={"rotary_embedding_dim": 4, "num_heads": 2, "interleaved": 0},
            ),
            OpNode(
                id="n1",
                op="SKIPRMSNORM",
                inputs=["rot_out", "skip", "gamma"],
                outputs=["skip_norm", "skip_residual"],
                attrs={"epsilon": 1e-6},
            ),
            OpNode(
                id="n2",
                op="GROUPQUERYATTENTION",
                inputs=["q", "k", "v", "past_k", "past_v", "bias"],
                outputs=["attn_out", "present_k", "present_v"],
                attrs={
                    "num_heads": 2,
                    "kv_num_heads": 1,
                    "scale": 0.5,
                    "softcap": 0.0,
                    "input_slots": [0, 1, 2, 3, 4, 10],
                },
            ),
        ],
        inputs=["rot_in", "pos", "skip", "q", "k", "v", "past_k", "past_v", "bias"],
        outputs=["skip_norm", "attn_out", "present_k", "present_v"],
        metadata={},
    )

    code = render_mlx_module(ir)
    assert_snapshot("mlx_qwen_fused_decoder_ops.py", code, test_file=__file__)
