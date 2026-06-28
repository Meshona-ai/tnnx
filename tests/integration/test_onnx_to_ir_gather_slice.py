from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tnnx.ingest.onnx_reader import load_onnx_to_ir


def test_onnx_to_ir_gather_slice_mapping(tmp_path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 2])

    indices = numpy_helper.from_array(np.array([0, 2, 4], dtype=np.int64), name="indices")
    starts = numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([3], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps")

    gather = helper.make_node("Gather", inputs=["x", "indices"], outputs=["g"], axis=1)
    slice_node = helper.make_node(
        "Slice",
        inputs=["g", "starts", "ends", "axes", "steps"],
        outputs=["y"],
    )

    model = helper.make_model(
        helper.make_graph(
            [gather, slice_node],
            "gather_slice_graph",
            [x],
            [y],
            [indices, starts, ends, axes, steps],
        )
    )
    path = Path(tmp_path) / "gather_slice.onnx"
    onnx.save(model, path)

    ir, _ = load_onnx_to_ir(path)
    assert [node.op for node in ir.nodes] == ["GATHER", "SLICE"]
