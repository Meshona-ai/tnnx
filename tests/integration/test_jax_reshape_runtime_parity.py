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
    spec = importlib.util.spec_from_file_location("generated_jax_reshape", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jax_reshape_zero_copy_runtime_parity(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 12])
    shape = numpy_helper.from_array(np.array([0, -1], dtype=np.int64), name="shape")
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Reshape", inputs=["x", "shape"], outputs=["y"]),
            ],
            "jax_reshape_zero_copy",
            [x],
            [y],
            [shape],
        ),
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx_path = tmp_path / "jax_reshape_zero_copy.onnx"
    onnx.save(model, onnx_path)

    ir, weights = load_onnx_to_ir(onnx_path)
    module_path = emit_jax_module(ir, tmp_path / "generated_jax_reshape")
    weights_path = tmp_path / "jax_reshape_zero_copy_weights.npz"
    save_weights_npz(weights_path, weights)
    module = _load_module(module_path)

    params = module.load_weights(str(weights_path))
    x_input = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    actual = np.asarray(module.forward(params, {"x": x_input})["y"])
    expected = np.reshape(x_input, (2, 12))

    assert np.array_equal(actual, expected)
