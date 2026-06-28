from __future__ import annotations

import json
from pathlib import Path
from subprocess import run

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _write_simple_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.ones((4, 3), dtype=np.float32), name="w")
    node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"])
    model = helper.make_model(helper.make_graph([node], "cli_graph", [x], [y], [w]))
    onnx.save(model, path)


def test_cli_transpile_custom_weights_and_entry(tmp_path: str) -> None:
    onnx_path = Path(tmp_path) / "model.onnx"
    out_dir = Path(tmp_path) / "generated_custom"
    _write_simple_onnx(onnx_path)
    result = run(
        [
            "uv",
            "run",
            "tnnx",
            "transpile",
            "--onnx",
            str(onnx_path),
            "--target",
            "jax",
            "--out",
            str(out_dir),
            "--weights",
            "custom_weights.npz",
            "--entry",
            "predict",
            "--passes",
            "normalize",
            "--target-hardware",
            "cpu-test",
            "--preferred-dtype",
            "float32",
            "--memory-budget-mb",
            "256",
            "--latency-priority",
            "auditability",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "custom_weights.npz").exists()
    generated = (out_dir / "model_jax.py").read_text(encoding="utf-8")
    assert "predict = forward" in generated
    metadata = json.loads((out_dir / "compile_metadata.json").read_text(encoding="utf-8"))
    assert metadata["compile_config"]["enabled_passes"] == ["normalize"]
    assert metadata["applied_passes"] == ["normalize"]
    assert metadata["compile_config"]["resource_budget"]["target_hardware"] == "cpu-test"
    assert metadata["compile_config"]["resource_budget"]["memory_budget_mb"] == 256
