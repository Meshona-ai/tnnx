from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tnnx.codegen.mlx_codegen import emit_mlx_module
from tnnx.ir.types import GraphIR, OpNode, TensorRef


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generated_mlx_topology", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _out_of_order_split_ir() -> GraphIR:
    return GraphIR(
        name="out_of_order_split",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 2, 2], "input"),
            "split_a": TensorRef("split_a", "float32", [1, 1, 2], "intermediate"),
            "split_b": TensorRef("split_b", "float32", [1, 1, 2], "intermediate"),
            "y": TensorRef("y", "float32", [1, 1, 2], "output"),
        },
        nodes=[
            OpNode(
                id="consume",
                op="IDENTITY",
                inputs=["split_a"],
                outputs=["y"],
                attrs={},
            ),
            OpNode(
                id="produce",
                op="SPLIT",
                inputs=["x"],
                outputs=["split_a", "split_b"],
                attrs={"axis": 1, "num_outputs": 2},
            ),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )


def test_mlx_topology_scheduler_runtime_parity(tmp_path: Path) -> None:
    _ = pytest.importorskip("mlx.core")

    module_path = emit_mlx_module(_out_of_order_split_ir(), tmp_path / "generated_mlx_topology")
    module = _load_module(module_path)

    x_input = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    actual = np.asarray(module.forward({}, {"x": x_input})["y"])
    expected = np.split(x_input, 2, axis=1)[0]

    assert np.array_equal(actual, expected)
