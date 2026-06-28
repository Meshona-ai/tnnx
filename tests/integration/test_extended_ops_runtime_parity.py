from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tnnx.codegen.jax_codegen import emit_jax_module
from tnnx.codegen.mlx_codegen import emit_mlx_module
from tnnx.ingest.onnx_reader import load_onnx_to_ir
from tnnx.runtime.weights import save_weights_npz


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("generated_extended_ops", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _emit_module(target: str, ir: Any, out_dir: Path) -> Path:
    if target == "jax":
        return emit_jax_module(ir, out_dir)
    if target == "mlx":
        return emit_mlx_module(ir, out_dir)
    raise ValueError(f"Unsupported target: {target}")


def _require_runtime(target: str) -> None:
    if target == "jax":
        _ = pytest.importorskip("jax")
        return
    if target == "mlx":
        _ = pytest.importorskip("mlx.core")
        return
    raise ValueError(f"Unsupported target: {target}")


def _build_spatial_case(path: Path) -> tuple[dict[str, np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 192])

    scale = np.array([1.25, 0.75, 1.5], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    mean = np.array([0.5, -0.5, 0.25], dtype=np.float32)
    var = np.array([1.0, 1.5, 0.75], dtype=np.float32)
    sizes = np.array([1, 3, 8, 8], dtype=np.int64)
    clip_min = np.array([0.0], dtype=np.float32)
    clip_max = np.array([6.0], dtype=np.float32)

    nodes = [
        helper.make_node(
            "BatchNormalization",
            inputs=["x", "scale", "bias", "mean", "var"],
            outputs=["bn"],
            epsilon=1e-5,
        ),
        helper.make_node(
            "MaxPool",
            inputs=["bn"],
            outputs=["pool"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
        helper.make_node(
            "Resize",
            inputs=["pool", "", "", "sizes"],
            outputs=["up"],
            mode="nearest",
        ),
        helper.make_node("Flatten", inputs=["up"], outputs=["flat"], axis=1),
        helper.make_node("Cast", inputs=["flat"], outputs=["cast"], to=TensorProto.FLOAT),
        helper.make_node("Clip", inputs=["cast", "clip_min", "clip_max"], outputs=["clip"]),
        helper.make_node("Sigmoid", inputs=["clip"], outputs=["y"]),
    ]
    initializers = [
        numpy_helper.from_array(scale, name="scale"),
        numpy_helper.from_array(bias, name="bias"),
        numpy_helper.from_array(mean, name="mean"),
        numpy_helper.from_array(var, name="var"),
        numpy_helper.from_array(sizes, name="sizes"),
        numpy_helper.from_array(clip_min, name="clip_min"),
        numpy_helper.from_array(clip_max, name="clip_max"),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "extended_spatial", [x], [y], initializers),
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx.save(model, path)

    x_input = np.linspace(-2.0, 2.0, num=1 * 3 * 8 * 8, dtype=np.float32).reshape(1, 3, 8, 8)
    bn = (x_input - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + 1e-5)
    bn = bn * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)

    pool = np.empty((1, 3, 4, 4), dtype=np.float32)
    for oh in range(4):
        hs = oh * 2
        for ow in range(4):
            ws = ow * 2
            window = bn[:, :, hs : hs + 2, ws : ws + 2]
            pool[:, :, oh, ow] = np.max(window, axis=(2, 3))

    up = np.repeat(np.repeat(pool, 2, axis=2), 2, axis=3)
    flat = up.reshape(1, -1).astype(np.float32, copy=False)
    clipped = np.clip(flat, 0.0, 6.0)
    expected = 1.0 / (1.0 + np.exp(-clipped))
    return {"x": x_input}, expected.astype(np.float32, copy=False)


def _build_sequence_case(
    path: Path,
) -> tuple[dict[str, np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [1, 5])
    alt = helper.make_tensor_value_info("alt", TensorProto.FLOAT, [1, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 5])

    start = np.array(0, dtype=np.int64)
    limit = np.array(5, dtype=np.int64)
    delta = np.array(1, dtype=np.int64)
    scale = np.array([1.0, 0.5, 1.5, 0.75, 1.25], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.0, 0.25, -0.1], dtype=np.float32)

    nodes = [
        helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["rng"]),
        helper.make_node("Unsqueeze", inputs=["rng"], outputs=["unsq"], axes=[0]),
        helper.make_node("Where", inputs=["cond", "unsq", "alt"], outputs=["sel"]),
        helper.make_node(
            "RMSNormalization",
            inputs=["sel", "scale", "bias"],
            outputs=["norm"],
            axis=-1,
            epsilon=1e-5,
        ),
        helper.make_node("Sin", inputs=["norm"], outputs=["sin"]),
        helper.make_node("Cos", inputs=["sin"], outputs=["y"]),
    ]
    initializers = [
        numpy_helper.from_array(start, name="start"),
        numpy_helper.from_array(limit, name="limit"),
        numpy_helper.from_array(delta, name="delta"),
        numpy_helper.from_array(scale, name="scale"),
        numpy_helper.from_array(bias, name="bias"),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "extended_sequence", [cond, alt], [y], initializers),
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx.save(model, path)

    cond_input = np.array([[True, False, True, False, True]], dtype=bool)
    alt_input = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=np.float32)
    rng = np.arange(0.0, 5.0, 1.0, dtype=np.float32)
    unsq = rng.reshape(1, 5)
    sel = np.where(cond_input, unsq, alt_input)
    rms = np.mean(sel * sel, axis=-1, keepdims=True)
    norm = (sel / np.sqrt(rms + 1e-5)) * scale.reshape(1, -1) + bias.reshape(1, -1)
    expected = np.cos(np.sin(norm)).astype(np.float32, copy=False)
    return {"cond": cond_input, "alt": alt_input}, expected


def _build_split_case(
    path: Path,
) -> tuple[dict[str, np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6])

    split = np.array([2, 4], dtype=np.int64)

    nodes = [
        helper.make_node("Split", inputs=["x", "split"], outputs=["left", "right"], axis=1),
        helper.make_node("Tanh", inputs=["left"], outputs=["left_tanh"]),
        helper.make_node("Neg", inputs=["right"], outputs=["right_neg"]),
        helper.make_node("Concat", inputs=["left_tanh", "right_neg"], outputs=["y"], axis=1),
    ]
    initializers = [numpy_helper.from_array(split, name="split")]
    model = helper.make_model(
        helper.make_graph(nodes, "extended_split", [x], [y], initializers),
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx.save(model, path)

    x_input = np.array([[0.25, -0.75, 1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    left = np.tanh(x_input[:, :2]).astype(np.float32, copy=False)
    right = (-x_input[:, 2:]).astype(np.float32, copy=False)
    expected = np.concatenate([left, right], axis=1)
    return {"x": x_input}, expected


def _build_qwen_ops_case(
    path: Path,
) -> tuple[dict[str, np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 3])

    axis = np.array(1, dtype=np.int64)
    mod_rhs = np.array([2.5], dtype=np.float32)
    reduce_axes = np.array([1], dtype=np.int64)
    diag = np.array(0, dtype=np.int64)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array(
        [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
        dtype=np.float32,
    )

    nodes = [
        helper.make_node("Softplus", inputs=["x"], outputs=["soft"]),
        helper.make_node(
            "CumSum",
            inputs=["soft", "axis"],
            outputs=["cum"],
            exclusive=1,
            reverse=1,
        ),
        helper.make_node("Mod", inputs=["cum", "mod_rhs"], outputs=["mod"], fmod=1),
        helper.make_node(
            "ReduceSum",
            inputs=["mod", "reduce_axes"],
            outputs=["sum"],
            keepdims=1,
        ),
        helper.make_node("Trilu", inputs=["mod", "diag"], outputs=["tri"], upper=0),
        helper.make_node("Add", inputs=["tri", "sum"], outputs=["biased"]),
        helper.make_node("ScatterND", inputs=["biased", "indices", "updates"], outputs=["y"]),
    ]
    initializers = [
        numpy_helper.from_array(axis, name="axis"),
        numpy_helper.from_array(mod_rhs, name="mod_rhs"),
        numpy_helper.from_array(reduce_axes, name="reduce_axes"),
        numpy_helper.from_array(diag, name="diag"),
        numpy_helper.from_array(indices, name="indices"),
        numpy_helper.from_array(updates, name="updates"),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "extended_qwen_ops", [x], [y], initializers),
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx.save(model, path)

    x_input = np.array(
        [
            [0.25, -0.5, 1.5],
            [2.0, -1.0, 0.75],
            [1.25, 0.5, -2.0],
        ],
        dtype=np.float32,
    )
    soft = np.log1p(np.exp(x_input)).astype(np.float32, copy=False)
    flipped = np.flip(soft, axis=1)
    inclusive = np.cumsum(flipped, axis=1, dtype=np.float32)
    exclusive = np.concatenate(
        [np.zeros_like(inclusive[:, :1]), inclusive[:, :-1]],
        axis=1,
    )
    cum = np.flip(exclusive, axis=1)
    mod = np.fmod(cum, mod_rhs.astype(np.float32, copy=False))
    reduced = np.sum(mod, axis=1, keepdims=True, dtype=np.float32)
    tri = np.tril(mod, k=int(diag))
    biased = tri + reduced
    expected = biased.copy()
    expected[0, :] = updates[0]
    expected[2, :] = updates[1]
    return {"x": x_input}, expected.astype(np.float32, copy=False)


@pytest.mark.parametrize("target", ["jax", "mlx"])
@pytest.mark.parametrize("case_name", ["spatial", "sequence", "split"])
def test_extended_ops_runtime_parity(target: str, case_name: str, tmp_path: Path) -> None:
    _require_runtime(target)

    builders = {
        "spatial": _build_spatial_case,
        "sequence": _build_sequence_case,
        "split": _build_split_case,
    }
    onnx_path = tmp_path / f"{case_name}.onnx"
    inputs, expected = builders[case_name](onnx_path)

    ir, weights = load_onnx_to_ir(onnx_path)
    module_path = _emit_module(target, ir, tmp_path / f"generated_{target}_{case_name}")
    weights_path = tmp_path / f"{case_name}_weights.npz"
    save_weights_npz(weights_path, weights)

    module = _load_module(module_path)
    params = module.load_weights(str(weights_path))
    actual = np.asarray(module.forward(params, inputs)["y"])

    atol = 1e-5 if target == "jax" else 2e-5
    rtol = 1e-5 if target == "jax" else 2e-5
    assert np.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Extended-op parity failed for {target}/{case_name}: "
        f"max_abs={np.max(np.abs(actual - expected))}"
    )


def test_qwen_ops_runtime_parity_jax(tmp_path: Path) -> None:
    _require_runtime("jax")

    onnx_path = tmp_path / "qwen_ops.onnx"
    inputs, expected = _build_qwen_ops_case(onnx_path)

    ir, weights = load_onnx_to_ir(onnx_path)
    module_path = _emit_module("jax", ir, tmp_path / "generated_jax_qwen_ops")
    weights_path = tmp_path / "qwen_ops_weights.npz"
    save_weights_npz(weights_path, weights)

    module = _load_module(module_path)
    params = module.load_weights(str(weights_path))
    actual = np.asarray(module.forward(params, inputs)["y"])

    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-5), (
        f"Extended-op parity failed for jax/qwen_ops: max_abs={np.max(np.abs(actual - expected))}"
    )


def test_qwen_ops_runtime_parity_mlx(tmp_path: Path) -> None:
    _require_runtime("mlx")

    onnx_path = tmp_path / "qwen_ops.onnx"
    inputs, expected = _build_qwen_ops_case(onnx_path)

    ir, weights = load_onnx_to_ir(onnx_path)
    module_path = _emit_module("mlx", ir, tmp_path / "generated_mlx_qwen_ops")
    weights_path = tmp_path / "qwen_ops_weights.npz"
    save_weights_npz(weights_path, weights)

    module = _load_module(module_path)
    params = module.load_weights(str(weights_path))
    actual = np.asarray(module.forward(params, inputs)["y"])

    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-5), (
        f"Extended-op parity failed for mlx/qwen_ops: max_abs={np.max(np.abs(actual - expected))}"
    )
