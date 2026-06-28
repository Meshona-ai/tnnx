from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tnnx.ingest.onnx_reader import load_onnx_to_ir


def test_onnx_to_ir_layernorm_mapping(tmp_path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    scale = numpy_helper.from_array(np.ones((4,), dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.zeros((4,), dtype=np.float32), name="bias")
    ln = helper.make_node(
        "LayerNormalization",
        inputs=["x", "scale", "bias"],
        outputs=["y"],
        axis=-1,
        epsilon=1e-5,
    )
    model = helper.make_model(helper.make_graph([ln], "ln_graph", [x], [y], [scale, bias]))
    path = Path(tmp_path) / "ln.onnx"
    onnx.save(model, path)

    ir, _ = load_onnx_to_ir(path)
    assert [node.op for node in ir.nodes] == ["LAYERNORM"]
