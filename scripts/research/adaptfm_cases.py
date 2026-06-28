from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


@dataclass(frozen=True, slots=True)
class CaseData:
    name: str
    graph_name: str
    seed: int
    inputs: dict[str, np.ndarray]
    expected: dict[str, np.ndarray]
    output_name: str
    atol: float
    rtol: float
    description: str


def available_cases() -> tuple[str, ...]:
    return tuple(_BUILDERS.keys())


def write_case_onnx(name: str, path: Path, *, seed: int = 0) -> CaseData:
    try:
        builder = _BUILDERS[name]
    except KeyError as exc:
        available = ", ".join(available_cases())
        raise ValueError(f"Unknown case {name!r}. Available: {available}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    model, case = builder(seed)
    onnx.save(model, path)
    return case


def _tensor_info(name: str, shape: tuple[int, ...]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _init(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(value.astype(np.float32)), name=name)


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(seed + 1009 * stream)


def _gelu(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[np.float32])
    return (0.5 * x * (1.0 + erf(x / np.float32(math.sqrt(2.0))))).astype(np.float32)


def _layer_norm(
    x: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (((x - mean) / np.sqrt(var + epsilon)) * scale + bias).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def _make_model(
    graph_name: str,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto],
) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializers)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])


def _linear(seed: int) -> tuple[onnx.ModelProto, CaseData]:
    graph_name = "adaptfm_linear"
    x = _rng(seed, 1).normal(size=(2, 4)).astype(np.float32)
    w = (_rng(seed, 2).normal(size=(4, 3)) * 0.25).astype(np.float32)
    b = (_rng(seed, 3).normal(size=(3,)) * 0.05).astype(np.float32)
    y = (x @ w + b).astype(np.float32)
    nodes = [
        helper.make_node("MatMul", inputs=["x", "w"], outputs=["mm"]),
        helper.make_node("Add", inputs=["mm", "b"], outputs=["y"]),
    ]
    model = _make_model(
        graph_name,
        nodes,
        [_tensor_info("x", x.shape)],
        [_tensor_info("y", y.shape)],
        [_init("w", w), _init("b", b)],
    )
    return model, CaseData(
        name="linear",
        graph_name=graph_name,
        seed=seed,
        inputs={"x": x},
        expected={"y": y},
        output_name="y",
        atol=1e-5,
        rtol=1e-5,
        description="Op-level MatMul+Add lowering and equivalence check.",
    )


def _mlp_norm(seed: int) -> tuple[onnx.ModelProto, CaseData]:
    graph_name = "adaptfm_mlp_norm_block"
    x = _rng(seed, 4).normal(size=(1, 3, 4)).astype(np.float32)
    w1 = (_rng(seed, 5).normal(size=(4, 8)) * 0.20).astype(np.float32)
    b1 = (_rng(seed, 6).normal(size=(8,)) * 0.05).astype(np.float32)
    w2 = (_rng(seed, 7).normal(size=(8, 4)) * 0.15).astype(np.float32)
    b2 = (_rng(seed, 8).normal(size=(4,)) * 0.05).astype(np.float32)
    scale = (1.0 + _rng(seed, 9).normal(size=(4,)) * 0.01).astype(np.float32)
    bias = (_rng(seed, 10).normal(size=(4,)) * 0.01).astype(np.float32)
    h1 = x @ w1 + b1
    h2 = _gelu(h1)
    h3 = h2 @ w2 + b2
    residual = h3 + x
    y = _layer_norm(residual, scale, bias)
    nodes = [
        helper.make_node("MatMul", inputs=["x", "w1"], outputs=["h0"]),
        helper.make_node("Add", inputs=["h0", "b1"], outputs=["h1"]),
        helper.make_node("Gelu", inputs=["h1"], outputs=["h2"]),
        helper.make_node("MatMul", inputs=["h2", "w2"], outputs=["h3"]),
        helper.make_node("Add", inputs=["h3", "b2"], outputs=["h4"]),
        helper.make_node("Add", inputs=["h4", "x"], outputs=["residual"]),
        helper.make_node(
            "LayerNormalization",
            inputs=["residual", "ln_scale", "ln_bias"],
            outputs=["y"],
            axis=-1,
            epsilon=1e-5,
        ),
    ]
    model = _make_model(
        graph_name,
        nodes,
        [_tensor_info("x", x.shape)],
        [_tensor_info("y", y.shape)],
        [
            _init("w1", w1),
            _init("b1", b1),
            _init("w2", w2),
            _init("b2", b2),
            _init("ln_scale", scale),
            _init("ln_bias", bias),
        ],
    )
    return model, CaseData(
        name="mlp_norm",
        graph_name=graph_name,
        seed=seed,
        inputs={"x": x},
        expected={"y": y},
        output_name="y",
        atol=5e-5,
        rtol=5e-5,
        description="Block-level MLP+GELU+residual+LayerNorm case.",
    )


def _tiny_transformer(seed: int) -> tuple[onnx.ModelProto, CaseData]:
    graph_name = "adaptfm_tiny_transformer_block"
    x = _rng(seed, 11).normal(size=(1, 3, 4)).astype(np.float32)
    wq = (_rng(seed, 12).normal(size=(4, 4)) * 0.18).astype(np.float32)
    wk = (_rng(seed, 13).normal(size=(4, 4)) * 0.18).astype(np.float32)
    wv = (_rng(seed, 14).normal(size=(4, 4)) * 0.18).astype(np.float32)
    wo = (_rng(seed, 15).normal(size=(4, 4)) * 0.16).astype(np.float32)
    scale_value = np.asarray([1.0 / math.sqrt(4.0)], dtype=np.float32)
    ln_scale = (1.0 + _rng(seed, 16).normal(size=(4,)) * 0.01).astype(np.float32)
    ln_bias = (_rng(seed, 17).normal(size=(4,)) * 0.01).astype(np.float32)

    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = (q @ np.swapaxes(k, -1, -2)) * scale_value
    probs = _softmax(scores)
    context = probs @ v
    projected = context @ wo
    residual = projected + x
    y = _layer_norm(residual, ln_scale, ln_bias)
    nodes = [
        helper.make_node("MatMul", inputs=["x", "wq"], outputs=["q"]),
        helper.make_node("MatMul", inputs=["x", "wk"], outputs=["k"]),
        helper.make_node("MatMul", inputs=["x", "wv"], outputs=["v"]),
        helper.make_node("Transpose", inputs=["k"], outputs=["kt"], perm=[0, 2, 1]),
        helper.make_node("MatMul", inputs=["q", "kt"], outputs=["scores_raw"]),
        helper.make_node("Mul", inputs=["scores_raw", "attn_scale"], outputs=["scores"]),
        helper.make_node("Softmax", inputs=["scores"], outputs=["probs"], axis=-1),
        helper.make_node("MatMul", inputs=["probs", "v"], outputs=["context"]),
        helper.make_node("MatMul", inputs=["context", "wo"], outputs=["projected"]),
        helper.make_node("Add", inputs=["projected", "x"], outputs=["residual"]),
        helper.make_node(
            "LayerNormalization",
            inputs=["residual", "ln_scale", "ln_bias"],
            outputs=["y"],
            axis=-1,
            epsilon=1e-5,
        ),
    ]
    model = _make_model(
        graph_name,
        nodes,
        [_tensor_info("x", x.shape)],
        [_tensor_info("y", y.shape)],
        [
            _init("wq", wq),
            _init("wk", wk),
            _init("wv", wv),
            _init("wo", wo),
            _init("attn_scale", scale_value),
            _init("ln_scale", ln_scale),
            _init("ln_bias", ln_bias),
        ],
    )
    return model, CaseData(
        name="tiny_transformer",
        graph_name=graph_name,
        seed=seed,
        inputs={"x": x},
        expected={"y": y},
        output_name="y",
        atol=1e-4,
        rtol=1e-4,
        description=(
            "FM-like block-level case: Q/K/V attention, Softmax, output projection, "
            "residual, and LayerNorm. It is not a full Qwen or Flux model."
        ),
    )


_BUILDERS: dict[str, Callable[[int], tuple[onnx.ModelProto, CaseData]]] = {
    "linear": _linear,
    "mlp_norm": _mlp_norm,
    "tiny_transformer": _tiny_transformer,
}
