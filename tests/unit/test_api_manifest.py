from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tnnx.api import _runtime_values_from_weights, transpile_onnx
from tnnx.config import CompileConfig, ResourceBudget


def _write_simple_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.ones((4, 3), dtype=np.float32), name="w")
    node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"])
    model = helper.make_model(helper.make_graph([node], "api_graph", [x], [y], [w]))
    onnx.save(model, path)


def test_transpile_manifest_uses_configured_entry_and_weights(tmp_path: str) -> None:
    onnx_path = Path(tmp_path) / "model.onnx"
    out_dir = Path(tmp_path) / "out"
    _write_simple_onnx(onnx_path)

    manifest = transpile_onnx(
        str(onnx_path),
        "jax",
        str(out_dir),
        config=CompileConfig(entrypoint="predict", weights_filename="custom_weights.npz"),
    )

    assert manifest.entrypoint == "predict"
    assert manifest.weights_file is not None
    assert manifest.weights_file.name == "custom_weights.npz"
    assert "compile_metadata.json" in {path.name for path in manifest.files}


def test_runtime_values_skip_large_initializers() -> None:
    runtime_values = _runtime_values_from_weights(
        {
            "shape": np.asarray([2, 3], dtype=np.int64),
            "large": np.ones((128,), dtype=np.float32),
        }
    )

    assert runtime_values["shape"] == [2, 3]
    assert "large" not in runtime_values


def test_transpile_onnx_skips_graph_ir_when_disabled(tmp_path: str) -> None:
    onnx_path = Path(tmp_path) / "model.onnx"
    out_dir = Path(tmp_path) / "out_no_graph_ir"
    _write_simple_onnx(onnx_path)

    _ = transpile_onnx(
        str(onnx_path),
        "jax",
        str(out_dir),
        config=CompileConfig(emit_graph_ir=False),
    )

    assert not (out_dir / "graph_ir.json").exists()
    assert (out_dir / "compile_metadata.json").exists()
    assert (out_dir / "model_jax.py").exists()
    assert (out_dir / "weights.npz").exists()


def test_transpile_onnx_records_pass_and_resource_metadata(tmp_path: str) -> None:
    onnx_path = Path(tmp_path) / "model.onnx"
    out_dir = Path(tmp_path) / "out_metadata"
    _write_simple_onnx(onnx_path)

    _ = transpile_onnx(
        str(onnx_path),
        "jax",
        str(out_dir),
        config=CompileConfig(
            enabled_passes=("normalize",),
            resource_budget=ResourceBudget(
                target_hardware="cpu-ci",
                preferred_dtype="float32",
                memory_budget_mb=512,
                latency_priority="auditability",
                notes="unit-test budget",
            ),
        ),
    )

    metadata = json.loads((out_dir / "compile_metadata.json").read_text(encoding="utf-8"))

    assert metadata["target"] == "jax"
    assert metadata["graph_name"] == "api_graph"
    assert metadata["applied_passes"] == ["normalize"]
    assert metadata["compile_config"]["enabled_passes"] == ["normalize"]
    assert metadata["compile_config"]["resource_budget"] == {
        "target_hardware": "cpu-ci",
        "preferred_dtype": "float32",
        "memory_budget_mb": 512,
        "latency_priority": "auditability",
        "notes": "unit-test budget",
    }
