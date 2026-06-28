from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tnnx.codegen.mlx_codegen import emit_mlx_module
from tnnx.ingest.onnx_reader import load_onnx_to_ir
from tnnx.runtime.weights import save_weights_npz


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generated_mlx_model", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mlx_parity_mlp(tmp_path: str) -> None:
    mlx = pytest.importorskip("mlx.core")
    _ = mlx
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w_arr = np.array([[1, 0, 2], [0, 1, 3], [1, 1, 1], [2, 0, 1]], dtype=np.float32)
    b_arr = np.array([1, 2, 3], dtype=np.float32)
    w = numpy_helper.from_array(w_arr, name="w")
    b = numpy_helper.from_array(b_arr, name="b")
    gemm = helper.make_node("Gemm", inputs=["x", "w", "b"], outputs=["y"])
    model = helper.make_model(helper.make_graph([gemm], "mlp", [x], [y], [w, b]))
    onnx_path = Path(tmp_path) / "mlp.onnx"
    onnx.save(model, onnx_path)

    ir, weights = load_onnx_to_ir(onnx_path)
    module_path = emit_mlx_module(ir, Path(tmp_path))
    weights_path = Path(tmp_path) / "weights.npz"
    save_weights_npz(weights_path, weights)
    mod = _load_module(module_path)

    params = mod.load_weights(str(weights_path))
    x_input = np.array([[1, 2, 3, 4]], dtype=np.float32)
    out = np.asarray(mod.forward(params, {"x": x_input})["y"])
    ref = x_input @ w_arr + b_arr
    assert np.allclose(out, ref, rtol=1e-4, atol=1e-5)
