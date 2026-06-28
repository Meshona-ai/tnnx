from __future__ import annotations

import importlib.util
import py_compile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tnnx.api import transpile_onnx


def _write_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    bias = numpy_helper.from_array(np.asarray([1.0, 2.0], dtype=np.float32), name="bias")
    node = helper.make_node("Add", ["x", "bias"], ["y"])
    graph = helper.make_graph([node], "generated_code_gate", [x], [y], initializer=[bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, path)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_target(onnx_path: Path, root: Path, target: str) -> None:
    out_dir = root / target
    manifest = transpile_onnx(str(onnx_path), target, str(out_dir))
    module_path = out_dir / f"model_{target}.py"
    source = module_path.read_text(encoding="utf-8")
    if "Unsupported op for" in source or "TODO" in source:
        raise AssertionError(f"Generated {target} source contains unresolved placeholder text.")
    py_compile.compile(str(module_path), doraise=True)
    module = _load(module_path, f"generated_{target}_gate")
    params = module.load_weights(str(manifest.weights_file))
    actual = np.asarray(
        module.forward(params, {"x": np.asarray([[3.0, 4.0]], dtype=np.float32)})["y"]
    )
    expected = np.asarray([[4.0, 6.0]], dtype=np.float32)
    if not np.allclose(actual, expected):
        raise AssertionError(f"Generated {target} code returned {actual}, expected {expected}.")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tnnx-generated-code-") as tmp:
        root = Path(tmp)
        onnx_path = root / "tiny.onnx"
        _write_model(onnx_path)
        _check_target(onnx_path, root, "jax")
        _check_target(onnx_path, root, "mlx")
    print("Generated JAX/MLX code compiles, imports, and runs a tiny parity check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
