from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tnnx.codegen.jax_codegen import emit_jax_module
from tnnx.ingest.onnx_reader import load_onnx_to_ir
from tnnx.runtime.weights import save_weights_npz


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generated_conv_model", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _conv2d_nchw(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, _, h, ww = x.shape
    oc, ic, kh, kw = w.shape
    out = np.zeros((n, oc, h - kh + 1, ww - kw + 1), dtype=np.float32)
    for n_i in range(n):
        for o in range(oc):
            for i in range(ic):
                for y in range(h - kh + 1):
                    for x_i in range(ww - kw + 1):
                        out[n_i, o, y, x_i] += np.sum(
                            x[n_i, i, y : y + kh, x_i : x_i + kw] * w[o, i]
                        )
            out[n_i, o] += b[o]
    return np.maximum(out, 0)


def _conv1d_ncw(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, _, length = x.shape
    oc, ic, kernel = w.shape
    out = np.zeros((n, oc, length), dtype=np.float32)
    x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1)))
    for n_i in range(n):
        for o in range(oc):
            for i in range(ic):
                for pos in range(length):
                    out[n_i, o, pos] += np.sum(x_pad[n_i, i, pos : pos + kernel] * w[o, i])
            out[n_i, o] += b[o]
    return np.maximum(out, 0)


def test_jax_parity_conv(tmp_path: str) -> None:
    jax = pytest.importorskip("jax")
    _ = jax
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3, 3])
    w_arr = np.ones((1, 1, 2, 2), dtype=np.float32)
    b_arr = np.array([0.5], dtype=np.float32)
    w = numpy_helper.from_array(w_arr, name="w")
    b = numpy_helper.from_array(b_arr, name="b")
    conv = helper.make_node("Conv", inputs=["x", "w", "b"], outputs=["h"])
    relu = helper.make_node("Relu", inputs=["h"], outputs=["y"])
    model = helper.make_model(helper.make_graph([conv, relu], "conv", [x_info], [y_info], [w, b]))
    onnx_path = Path(tmp_path) / "conv.onnx"
    onnx.save(model, onnx_path)

    ir, weights = load_onnx_to_ir(onnx_path)
    module_path = emit_jax_module(ir, Path(tmp_path))
    weights_path = Path(tmp_path) / "weights.npz"
    save_weights_npz(weights_path, weights)
    mod = _load_module(module_path)
    params = mod.load_weights(str(weights_path))

    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    out = np.asarray(mod.forward(params, {"x": x})["y"])
    ref = _conv2d_nchw(x, w_arr, b_arr)
    assert np.allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_jax_parity_conv1d(tmp_path: str) -> None:
    jax = pytest.importorskip("jax")
    _ = jax
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 8])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8])
    w_arr = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 9.0
    b_arr = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    w = numpy_helper.from_array(w_arr, name="w")
    b = numpy_helper.from_array(b_arr, name="b")
    conv = helper.make_node("Conv", inputs=["x", "w", "b"], outputs=["h"], pads=[1, 1])
    relu = helper.make_node("Relu", inputs=["h"], outputs=["y"])
    model = helper.make_model(helper.make_graph([conv, relu], "conv1d", [x_info], [y_info], [w, b]))
    onnx_path = Path(tmp_path) / "conv1d.onnx"
    onnx.save(model, onnx_path)

    ir, weights = load_onnx_to_ir(onnx_path)
    module_path = emit_jax_module(ir, Path(tmp_path))
    weights_path = Path(tmp_path) / "weights.npz"
    save_weights_npz(weights_path, weights)
    mod = _load_module(module_path)
    params = mod.load_weights(str(weights_path))

    x = np.arange(16, dtype=np.float32).reshape(1, 2, 8) / 8.0
    out = np.asarray(mod.forward(params, {"x": x})["y"])
    ref = _conv1d_ncw(x, w_arr, b_arr)
    assert np.allclose(out, ref, rtol=1e-4, atol=1e-5)
