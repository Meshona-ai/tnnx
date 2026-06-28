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
    spec = importlib.util.spec_from_file_location("generated_jax_instance_norm", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jax_instance_norm_runtime_parity(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])
    scale = np.array([1.5, 0.5], dtype=np.float32)
    bias = np.array([0.25, -0.75], dtype=np.float32)

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "InstanceNormalization",
                    inputs=["x", "scale", "bias"],
                    outputs=["y"],
                    epsilon=1e-5,
                )
            ],
            "jax_instance_norm",
            [x],
            [y],
            [
                numpy_helper.from_array(scale, name="scale"),
                numpy_helper.from_array(bias, name="bias"),
            ],
        ),
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx_path = tmp_path / "jax_instance_norm.onnx"
    onnx.save(model, onnx_path)

    ir, weights = load_onnx_to_ir(onnx_path)
    module_path = emit_jax_module(ir, tmp_path / "generated_jax_instance_norm")
    weights_path = tmp_path / "jax_instance_norm_weights.npz"
    save_weights_npz(weights_path, weights)

    module = _load_module(module_path)
    params = module.load_weights(str(weights_path))
    x_input = np.array(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        ],
        dtype=np.float32,
    )
    actual = np.asarray(module.forward(params, {"x": x_input})["y"])

    mean = np.mean(x_input, axis=(2, 3), keepdims=True)
    var = np.var(x_input, axis=(2, 3), keepdims=True)
    expected = (x_input - mean) / np.sqrt(var + 1e-5)
    expected = expected * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)
    expected = expected.astype(np.float32, copy=False)

    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-5), (
        f"JAX instance norm parity failed: max_abs={np.max(np.abs(actual - expected))}"
    )
