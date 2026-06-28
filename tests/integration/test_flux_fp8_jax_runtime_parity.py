# ruff: noqa: E501

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx import TensorProto

from tnnx.codegen.jax_codegen import emit_jax_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef
from tnnx.runtime.weights import save_weights_npz


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _linear_case() -> tuple[
    GraphIR, dict[str, np.ndarray[Any, Any]], dict[str, np.ndarray[Any, Any]]
]:
    weights = {
        "scale": np.array([0.25], dtype=np.float32),
        "zero": np.array([7], dtype=np.uint8),
        "w": np.array(
            [
                [0.5, -1.0, 0.25],
                [1.5, 0.0, -0.5],
                [-0.25, 0.75, 1.0],
                [0.125, -0.5, 0.875],
            ],
            dtype=np.float32,
        ),
    }
    ir = GraphIR(
        name="flux_fp8_linear",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 4], "input"),
            "scale": TensorRef("scale", "float32", [1], "initializer"),
            "zero": TensorRef("zero", "uint8", [1], "initializer"),
            "w": TensorRef("w", "float32", [4, 3], "initializer"),
            "x_fp8": TensorRef("x_fp8", "float8_e4m3fn", [1, 4], "intermediate"),
            "x_fp16": TensorRef("x_fp16", "float16", [1, 4], "intermediate"),
            "q": TensorRef("q", "uint8", [1, 4], "intermediate"),
            "dq": TensorRef("dq", "float32", [1, 4], "intermediate"),
            "y": TensorRef("y", "float32", [1, 3], "output"),
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
                outputs=["x_fp16"],
                attrs={"to": int(TensorProto.FLOAT16)},
            ),
            OpNode(
                id="n2", op="QUANTIZE", inputs=["x_fp16", "scale", "zero"], outputs=["q"], attrs={}
            ),
            OpNode(
                id="n3", op="DEQUANTIZE", inputs=["q", "scale", "zero"], outputs=["dq"], attrs={}
            ),
            OpNode(id="n4", op="MATMUL", inputs=["dq", "w"], outputs=["y"], attrs={}),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    inputs = {
        "x": np.array([[0.6, -1.2, 2.4, -0.3]], dtype=np.float32),
    }
    return ir, weights, inputs


def _attention_case() -> tuple[
    GraphIR, dict[str, np.ndarray[Any, Any]], dict[str, np.ndarray[Any, Any]]
]:
    weights = {
        "scale": np.array([0.125], dtype=np.float32),
        "zero": np.array([3], dtype=np.uint8),
        "v": np.array(
            [
                [[0.2, -0.5, 1.0], [1.5, 0.25, -0.75]],
            ],
            dtype=np.float32,
        ),
    }
    ir = GraphIR(
        name="flux_fp8_attention",
        opset=18,
        tensors={
            "q_in": TensorRef("q_in", "float32", [1, 2, 4], "input"),
            "k_in": TensorRef("k_in", "float32", [1, 2, 4], "input"),
            "scale": TensorRef("scale", "float32", [1], "initializer"),
            "zero": TensorRef("zero", "uint8", [1], "initializer"),
            "v": TensorRef("v", "float32", [1, 2, 3], "initializer"),
            "q_fp8": TensorRef("q_fp8", "float8_e5m2", [1, 2, 4], "intermediate"),
            "k_fp8": TensorRef("k_fp8", "float8_e5m2", [1, 2, 4], "intermediate"),
            "q_fp16": TensorRef("q_fp16", "float16", [1, 2, 4], "intermediate"),
            "k_fp16": TensorRef("k_fp16", "float16", [1, 2, 4], "intermediate"),
            "q_q": TensorRef("q_q", "uint8", [1, 2, 4], "intermediate"),
            "k_q": TensorRef("k_q", "uint8", [1, 2, 4], "intermediate"),
            "q_dq": TensorRef("q_dq", "float32", [1, 2, 4], "intermediate"),
            "k_dq": TensorRef("k_dq", "float32", [1, 2, 4], "intermediate"),
            "k_t": TensorRef("k_t", "float32", [1, 4, 2], "intermediate"),
            "scores": TensorRef("scores", "float32", [1, 2, 2], "intermediate"),
            "probs": TensorRef("probs", "float32", [1, 2, 2], "intermediate"),
            "y": TensorRef("y", "float32", [1, 2, 3], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="CAST",
                inputs=["q_in"],
                outputs=["q_fp8"],
                attrs={"to": int(TensorProto.FLOAT8E5M2)},
            ),
            OpNode(
                id="n1",
                op="CAST",
                inputs=["k_in"],
                outputs=["k_fp8"],
                attrs={"to": int(TensorProto.FLOAT8E5M2)},
            ),
            OpNode(
                id="n2",
                op="CAST",
                inputs=["q_fp8"],
                outputs=["q_fp16"],
                attrs={"to": int(TensorProto.FLOAT16)},
            ),
            OpNode(
                id="n3",
                op="CAST",
                inputs=["k_fp8"],
                outputs=["k_fp16"],
                attrs={"to": int(TensorProto.FLOAT16)},
            ),
            OpNode(
                id="n4",
                op="QUANTIZE",
                inputs=["q_fp16", "scale", "zero"],
                outputs=["q_q"],
                attrs={},
            ),
            OpNode(
                id="n5",
                op="QUANTIZE",
                inputs=["k_fp16", "scale", "zero"],
                outputs=["k_q"],
                attrs={},
            ),
            OpNode(
                id="n6",
                op="DEQUANTIZE",
                inputs=["q_q", "scale", "zero"],
                outputs=["q_dq"],
                attrs={},
            ),
            OpNode(
                id="n7",
                op="DEQUANTIZE",
                inputs=["k_q", "scale", "zero"],
                outputs=["k_dq"],
                attrs={},
            ),
            OpNode(
                id="n8",
                op="TRANSPOSE",
                inputs=["k_dq"],
                outputs=["k_t"],
                attrs={"perm": [0, 2, 1]},
            ),
            OpNode(id="n9", op="MATMUL", inputs=["q_dq", "k_t"], outputs=["scores"], attrs={}),
            OpNode(
                id="n10", op="SOFTMAX", inputs=["scores"], outputs=["probs"], attrs={"axis": -1}
            ),
            OpNode(id="n11", op="MATMUL", inputs=["probs", "v"], outputs=["y"], attrs={}),
        ],
        inputs=["q_in", "k_in"],
        outputs=["y"],
        metadata={},
    )
    inputs = {
        "q_in": np.array(
            [[[0.5, -1.0, 0.25, 0.75], [1.25, -0.5, 0.5, -1.5]]],
            dtype=np.float32,
        ),
        "k_in": np.array(
            [[[1.0, 0.25, -0.75, 0.5], [-0.5, 1.5, 0.5, -1.0]]],
            dtype=np.float32,
        ),
    }
    return ir, weights, inputs


def _reference(
    case_name: str,
    weights: dict[str, np.ndarray[Any, Any]],
    inputs: dict[str, np.ndarray[Any, Any]],
) -> np.ndarray[Any, Any]:
    jnp = pytest.importorskip("jax.numpy")

    if case_name == "linear":
        x = jnp.asarray(inputs["x"])
        x_fp8 = x.astype(jnp.float8_e4m3fn)
        x_fp16 = x_fp8.astype(jnp.float16)
        q = jnp.round(jnp.asarray(x_fp16) / jnp.asarray(weights["scale"])) + jnp.asarray(
            weights["zero"]
        )
        q = q.astype(jnp.asarray(weights["zero"]).dtype)
        dq = (
            jnp.asarray(q, dtype=jnp.float32) - jnp.asarray(weights["zero"], dtype=jnp.float32)
        ) * jnp.asarray(weights["scale"], dtype=jnp.float32)
        return np.asarray(jnp.matmul(dq, jnp.asarray(weights["w"])))

    q_in = jnp.asarray(inputs["q_in"])
    k_in = jnp.asarray(inputs["k_in"])
    q_fp16 = q_in.astype(jnp.float8_e5m2).astype(jnp.float16)
    k_fp16 = k_in.astype(jnp.float8_e5m2).astype(jnp.float16)
    scale = jnp.asarray(weights["scale"])
    zero = jnp.asarray(weights["zero"])
    q_q = (jnp.round(jnp.asarray(q_fp16) / scale) + zero).astype(zero.dtype)
    k_q = (jnp.round(jnp.asarray(k_fp16) / scale) + zero).astype(zero.dtype)
    q_dq = (
        jnp.asarray(q_q, dtype=jnp.float32) - jnp.asarray(zero, dtype=jnp.float32)
    ) * jnp.asarray(scale, dtype=jnp.float32)
    k_dq = (
        jnp.asarray(k_q, dtype=jnp.float32) - jnp.asarray(zero, dtype=jnp.float32)
    ) * jnp.asarray(scale, dtype=jnp.float32)
    scores = jnp.matmul(q_dq, jnp.transpose(k_dq, (0, 2, 1)))
    probs = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    return np.asarray(jnp.matmul(probs, jnp.asarray(weights["v"])))


@pytest.mark.parametrize(
    ("case_name", "builder"),
    [
        ("linear", _linear_case),
        ("attention", _attention_case),
    ],
)
def test_flux_fp8_jax_runtime_parity(case_name: str, builder: Any, tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")

    ir, weights, inputs = builder()
    module_path = emit_jax_module(ir, tmp_path / f"generated_{case_name}")
    weights_path = tmp_path / f"{case_name}_weights.npz"
    save_weights_npz(weights_path, weights)

    module = _load_module(module_path, f"generated_flux_fp8_{case_name}")
    params = module.load_weights(str(weights_path))
    actual = np.asarray(module.forward(params, inputs)["y"])
    expected = _reference(case_name, weights, inputs)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=2e-3, atol=2e-3), (
        f"JAX FP8 parity failed for {case_name}: max_abs={np.max(np.abs(actual - expected))}"
    )
