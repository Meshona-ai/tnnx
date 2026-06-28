from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tnnx.codegen.jax_codegen import emit_jax_module
from tnnx.codegen.mlx_codegen import emit_mlx_module
from tnnx.ingest.onnx_reader import load_onnx_to_ir


def test_transpile_extended_ops_contract(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 192])

    scale = numpy_helper.from_array(np.ones((3,), dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="bias")
    mean = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="mean")
    var = numpy_helper.from_array(np.ones((3,), dtype=np.float32), name="var")
    sizes = numpy_helper.from_array(np.array([1, 3, 8, 8], dtype=np.int64), name="sizes")
    clip_min = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="clip_min")
    clip_max = numpy_helper.from_array(np.array([6.0], dtype=np.float32), name="clip_max")

    nodes = [
        helper.make_node(
            "BatchNormalization",
            inputs=["x", "scale", "bias", "mean", "var"],
            outputs=["bn"],
            epsilon=1e-5,
        ),
        helper.make_node(
            "MaxPool",
            inputs=["bn"],
            outputs=["pool"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
        helper.make_node(
            "Resize",
            inputs=["pool", "", "", "sizes"],
            outputs=["up"],
            mode="nearest",
        ),
        helper.make_node("Flatten", inputs=["up"], outputs=["flat"], axis=1),
        helper.make_node("Cast", inputs=["flat"], outputs=["cast"], to=TensorProto.FLOAT),
        helper.make_node("Clip", inputs=["cast", "clip_min", "clip_max"], outputs=["clip"]),
        helper.make_node("Sigmoid", inputs=["clip"], outputs=["y"]),
    ]

    graph = helper.make_graph(
        nodes,
        "extended_contract",
        [x],
        [y],
        [scale, bias, mean, var, sizes, clip_min, clip_max],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    onnx_path = tmp_path / "extended.onnx"
    onnx.save(model, onnx_path)

    ir, _ = load_onnx_to_ir(onnx_path)
    assert [node.op for node in ir.nodes] == [
        "BATCHNORM",
        "MAXPOOL",
        "UPSAMPLE",
        "FLATTEN",
        "CAST",
        "CLIP",
        "SIGMOID",
    ]

    jax_path = emit_jax_module(ir, tmp_path / "jax")
    mlx_path = emit_mlx_module(ir, tmp_path / "mlx")

    assert jax_path.exists()
    assert mlx_path.exists()
    assert Path(jax_path).read_text(encoding="utf-8")
    assert Path(mlx_path).read_text(encoding="utf-8")
