from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from tnnx.codegen.jax_codegen import emit_jax_module
from tnnx.ingest.onnx_reader import load_onnx_to_ir
from tnnx.runtime.weights import save_weights_npz


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("generated_jax_bfloat16_cast", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jax_bfloat16_cast_runtime_parity(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Cast", inputs=["x"], outputs=["x_bf16"], to=TensorProto.BFLOAT16),
                helper.make_node("Cast", inputs=["x_bf16"], outputs=["y"], to=TensorProto.FLOAT),
            ],
            "jax_bfloat16_cast",
            [x],
            [y],
        ),
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx_path = tmp_path / "jax_bfloat16_cast.onnx"
    onnx.save(model, onnx_path)

    ir, weights = load_onnx_to_ir(onnx_path)
    module_path = emit_jax_module(ir, tmp_path / "generated_jax_bfloat16_cast")
    weights_path = tmp_path / "jax_bfloat16_cast_weights.npz"
    save_weights_npz(weights_path, weights)

    module = _load_module(module_path)
    params = module.load_weights(str(weights_path))
    x_input = np.array([[1.1, -2.3, 3.7, -4.9]], dtype=np.float32)
    actual = np.asarray(module.forward(params, {"x": x_input})["y"])

    import jax.numpy as jnp

    expected = np.asarray(jnp.asarray(x_input).astype(jnp.bfloat16).astype(jnp.float32))
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6), (
        f"JAX bfloat16 cast parity failed: max_abs={np.max(np.abs(actual - expected))}"
    )


def test_jax_bfloat16_weight_loader_handles_serialized_bfloat16(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    import jax.numpy as jnp

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", inputs=["x"], outputs=["y"]),
            ],
            "jax_bfloat16_weight_loader",
            [x],
            [y],
        ),
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx_path = tmp_path / "jax_bfloat16_weight_loader.onnx"
    onnx.save(model, onnx_path)

    ir, _ = load_onnx_to_ir(onnx_path)
    module_path = emit_jax_module(ir, tmp_path / "generated_jax_bfloat16_weight_loader")
    weights_path = tmp_path / "jax_bfloat16_weight_loader_weights.npz"
    expected_bf16 = jnp.asarray([[1.5, -2.25]], dtype=jnp.bfloat16)
    save_weights_npz(weights_path, {"bf16_weight": np.asarray(expected_bf16)})

    module = _load_module(module_path)
    params = module.load_weights(str(weights_path))

    actual = np.asarray(params["bf16_weight"].astype(jnp.float32))
    expected = np.asarray(expected_bf16.astype(jnp.float32))

    assert str(params["bf16_weight"].dtype) == "bfloat16"
    assert np.array_equal(actual, expected)
