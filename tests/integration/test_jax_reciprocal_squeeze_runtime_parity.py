from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tnnx.codegen.jax_codegen import emit_jax_module
from tnnx.ingest.onnx_reader import load_onnx_to_ir
from tnnx.runtime.weights import save_weights_npz


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("generated_jax_reciprocal_squeeze", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jax_reciprocal_squeeze_runtime_parity(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    axes = np.array([1], dtype=np.int64)

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Squeeze", inputs=["x", "axes"], outputs=["sq"]),
                helper.make_node("Reciprocal", inputs=["sq"], outputs=["y"]),
            ],
            "jax_reciprocal_squeeze",
            [x],
            [y],
            [numpy_helper.from_array(axes, name="axes")],
        ),
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx_path = tmp_path / "jax_reciprocal_squeeze.onnx"
    onnx.save(model, onnx_path)

    ir, weights = load_onnx_to_ir(onnx_path)
    module_path = emit_jax_module(ir, tmp_path / "generated_jax_reciprocal_squeeze")
    weights_path = tmp_path / "jax_reciprocal_squeeze_weights.npz"
    save_weights_npz(weights_path, weights)

    module = _load_module(module_path)
    params = module.load_weights(str(weights_path))
    x_input = np.array([[[2.0, 4.0, 8.0, 16.0]]], dtype=np.float32)
    actual = np.asarray(module.forward(params, {"x": x_input})["y"])
    expected = (1.0 / x_input.reshape(1, 4)).astype(np.float32, copy=False)

    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6), (
        f"JAX reciprocal+squeeze parity failed: max_abs={np.max(np.abs(actual - expected))}"
    )
