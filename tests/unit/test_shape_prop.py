from __future__ import annotations

from tnnx.ir.types import GraphIR, OpNode, TensorRef
from tnnx.passes.shape_prop import propagate_shapes


def test_shape_prop_for_elementwise_and_relu() -> None:
    ir = GraphIR(
        name="g",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [2, 3], "input"),
            "b": TensorRef("b", "float32", [3], "initializer"),
            "h": TensorRef("h", "float32", [], "intermediate"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(id="n0", op="ADD", inputs=["x", "b"], outputs=["h"], attrs={}),
            OpNode(id="n1", op="RELU", inputs=["h"], outputs=["y"], attrs={}),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir)
    assert out.tensors["h"].shape == [2, 3]
    assert out.tensors["y"].shape == [2, 3]


def test_shape_prop_for_and_broadcast() -> None:
    ir = GraphIR(
        name="g_and",
        opset=18,
        tensors={
            "x": TensorRef("x", "bool", [1, 1, 16, 16], "input"),
            "b": TensorRef("b", "bool", [1, 1, 16, 1], "input"),
            "y": TensorRef("y", "bool", [], "output"),
        },
        nodes=[OpNode(id="n0", op="AND", inputs=["x", "b"], outputs=["y"], attrs={})],
        inputs=["x", "b"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir)
    assert out.tensors["y"].shape == [1, 1, 16, 16]


def test_shape_prop_for_pow_broadcast() -> None:
    ir = GraphIR(
        name="g_pow",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [2, 4], "input"),
            "exp": TensorRef("exp", "float32", [4], "input"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[OpNode(id="n0", op="POW", inputs=["x", "exp"], outputs=["y"], attrs={})],
        inputs=["x", "exp"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir)
    assert out.tensors["y"].shape == [2, 4]


def test_shape_prop_for_reduce_mean() -> None:
    ir = GraphIR(
        name="g_reduce_mean",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 16, 128], "input"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="REDUCEMEAN",
                inputs=["x"],
                outputs=["y"],
                attrs={"axes": [-1], "keepdims": 1},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir)
    assert out.tensors["y"].shape == [1, 16, 1]


def test_shape_prop_for_matmul_and_gemm() -> None:
    ir = GraphIR(
        name="g2",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [4, 8], "input"),
            "w": TensorRef("w", "float32", [8, 5], "initializer"),
            "w2": TensorRef("w2", "float32", [5, 5], "initializer"),
            "b": TensorRef("b", "float32", [5], "initializer"),
            "m": TensorRef("m", "float32", [], "intermediate"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(id="n0", op="MATMUL", inputs=["x", "w"], outputs=["m"], attrs={}),
            OpNode(
                id="n1",
                op="GEMM",
                inputs=["m", "w2", "b"],
                outputs=["y"],
                attrs={"transB": 1},
            ),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir)
    assert out.tensors["m"].shape == [4, 5]
    assert out.tensors["y"].shape == [4, 5]


def test_shape_prop_for_transpose_reshape_softmax() -> None:
    ir = GraphIR(
        name="g3",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [2, 3, 4], "input"),
            "shape": TensorRef("shape", "int64", [2], "initializer"),
            "t": TensorRef("t", "float32", [], "intermediate"),
            "r": TensorRef("r", "float32", [], "intermediate"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(id="n0", op="TRANSPOSE", inputs=["x"], outputs=["t"], attrs={"perm": [0, 2, 1]}),
            OpNode(id="n1", op="RESHAPE", inputs=["t", "shape"], outputs=["r"], attrs={}),
            OpNode(id="n2", op="SOFTMAX", inputs=["r"], outputs=["y"], attrs={"axis": -1}),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir, runtime_values={"shape": [2, 12]})
    assert out.tensors["t"].shape == [2, 4, 3]
    assert out.tensors["r"].shape == [2, 12]
    assert out.tensors["y"].shape == [2, 12]


def test_shape_prop_for_layernorm_and_rmsnorm() -> None:
    ir = GraphIR(
        name="g4",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [2, 4], "input"),
            "scale": TensorRef("scale", "float32", [4], "initializer"),
            "bias": TensorRef("bias", "float32", [4], "initializer"),
            "h": TensorRef("h", "float32", [], "intermediate"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="LAYERNORM",
                inputs=["x", "scale", "bias"],
                outputs=["h"],
                attrs={"axis": -1, "epsilon": 1e-5},
            ),
            OpNode(
                id="n1",
                op="RMSNORM",
                inputs=["h", "scale", "bias"],
                outputs=["y"],
                attrs={"axis": -1, "epsilon": 1e-5},
            ),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir)
    assert out.tensors["h"].shape == [2, 4]
    assert out.tensors["y"].shape == [2, 4]


def test_shape_prop_for_gather_and_slice() -> None:
    ir = GraphIR(
        name="g5",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [10, 4], "input"),
            "idx": TensorRef("idx", "int64", [3], "initializer"),
            "starts": TensorRef("starts", "int64", [1], "initializer"),
            "ends": TensorRef("ends", "int64", [1], "initializer"),
            "axes": TensorRef("axes", "int64", [1], "initializer"),
            "steps": TensorRef("steps", "int64", [1], "initializer"),
            "g": TensorRef("g", "float32", [], "intermediate"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(id="n0", op="GATHER", inputs=["x", "idx"], outputs=["g"], attrs={"axis": 0}),
            OpNode(
                id="n1",
                op="SLICE",
                inputs=["g", "starts", "ends", "axes", "steps"],
                outputs=["y"],
                attrs={},
            ),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(
        ir,
        runtime_values={
            "starts": [1],
            "ends": [3],
            "axes": [0],
            "steps": [1],
        },
    )
    assert out.tensors["g"].shape == [3, 4]
    assert out.tensors["y"].shape == [2, 4]


def test_shape_prop_for_concat_unsqueeze_and_flatten() -> None:
    ir = GraphIR(
        name="layout_ops",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [2, 3], "input"),
            "x2": TensorRef("x2", "float32", [2, 3], "input"),
            "axes": TensorRef("axes", "int64", [1], "initializer"),
            "c": TensorRef("c", "float32", [], "intermediate"),
            "u": TensorRef("u", "float32", [], "intermediate"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(id="n0", op="CONCAT", inputs=["x", "x2"], outputs=["c"], attrs={"axis": 1}),
            OpNode(id="n1", op="UNSQUEEZE", inputs=["c", "axes"], outputs=["u"], attrs={}),
            OpNode(id="n2", op="FLATTEN", inputs=["u"], outputs=["y"], attrs={"axis": 1}),
        ],
        inputs=["x", "x2"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir, runtime_values={"axes": [0]})
    assert out.tensors["c"].shape == [2, 6]
    assert out.tensors["u"].shape == [1, 2, 6]
    assert out.tensors["y"].shape == [1, 12]


def test_shape_prop_for_squeeze_and_reciprocal() -> None:
    ir = GraphIR(
        name="squeeze_ops",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 1, 4], "input"),
            "axes": TensorRef("axes", "int64", [1], "initializer"),
            "sq": TensorRef("sq", "float32", [], "intermediate"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(id="n0", op="SQUEEZE", inputs=["x", "axes"], outputs=["sq"], attrs={}),
            OpNode(id="n1", op="RECIPROCAL", inputs=["sq"], outputs=["y"], attrs={}),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir, runtime_values={"axes": [1]})
    assert out.tensors["sq"].shape == [1, 4]
    assert out.tensors["y"].shape == [1, 4]


def test_shape_prop_for_instance_norm() -> None:
    ir = GraphIR(
        name="instance_norm_ops",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 2, 4, 4], "input"),
            "scale": TensorRef("scale", "float32", [2], "initializer"),
            "bias": TensorRef("bias", "float32", [2], "initializer"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="INSTANCENORM",
                inputs=["x", "scale", "bias"],
                outputs=["y"],
                attrs={"epsilon": 1e-5},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir)
    assert out.tensors["y"].shape == [1, 2, 4, 4]


def test_shape_prop_for_pool_batchnorm_resize_and_quantization() -> None:
    ir = GraphIR(
        name="spatial_ops",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 3, 8, 8], "input"),
            "scale": TensorRef("scale", "float32", [3], "initializer"),
            "bias": TensorRef("bias", "float32", [3], "initializer"),
            "mean": TensorRef("mean", "float32", [3], "initializer"),
            "var": TensorRef("var", "float32", [3], "initializer"),
            "sizes": TensorRef("sizes", "int64", [4], "initializer"),
            "q_scale": TensorRef("q_scale", "float32", [1], "initializer"),
            "zero": TensorRef("zero", "uint8", [1], "initializer"),
            "h": TensorRef("h", "float32", [], "intermediate"),
            "p": TensorRef("p", "float32", [], "intermediate"),
            "r": TensorRef("r", "float32", [], "intermediate"),
            "q": TensorRef("q", "uint8", [], "intermediate"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="BATCHNORM",
                inputs=["x", "scale", "bias", "mean", "var"],
                outputs=["h"],
                attrs={"epsilon": 1e-5},
            ),
            OpNode(
                id="n1",
                op="AVGPOOL",
                inputs=["h"],
                outputs=["p"],
                attrs={"kernel_shape": [2, 2], "strides": [2, 2]},
            ),
            OpNode(
                id="n2",
                op="UPSAMPLE",
                inputs=["p", "sizes"],
                outputs=["r"],
                attrs={"mode": "nearest", "input_slots": [0, 3]},
            ),
            OpNode(
                id="n3", op="QUANTIZE", inputs=["r", "q_scale", "zero"], outputs=["q"], attrs={}
            ),
            OpNode(
                id="n4", op="DEQUANTIZE", inputs=["q", "q_scale", "zero"], outputs=["y"], attrs={}
            ),
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir, runtime_values={"sizes": [1, 3, 8, 8]})
    assert out.tensors["h"].shape == [1, 3, 8, 8]
    assert out.tensors["p"].shape == [1, 3, 4, 4]
    assert out.tensors["r"].shape == [1, 3, 8, 8]
    assert out.tensors["q"].shape == [1, 3, 8, 8]
    assert out.tensors["y"].shape == [1, 3, 8, 8]


def test_shape_prop_for_range_where_and_conv1d() -> None:
    ir = GraphIR(
        name="mixed_ops",
        opset=18,
        tensors={
            "start": TensorRef("start", "int64", [], "initializer"),
            "limit": TensorRef("limit", "int64", [], "initializer"),
            "delta": TensorRef("delta", "int64", [], "initializer"),
            "cond": TensorRef("cond", "bool", [5], "input"),
            "alt": TensorRef("alt", "float32", [5], "input"),
            "x": TensorRef("x", "float32", [2, 80, 16], "input"),
            "w": TensorRef("w", "float32", [32, 80, 3], "initializer"),
            "b": TensorRef("b", "float32", [32], "initializer"),
            "rng": TensorRef("rng", "float32", [], "intermediate"),
            "y": TensorRef("y", "float32", [], "intermediate"),
            "z": TensorRef("z", "float32", [], "output"),
        },
        nodes=[
            OpNode(
                id="n0", op="ARANGE", inputs=["start", "limit", "delta"], outputs=["rng"], attrs={}
            ),
            OpNode(id="n1", op="WHERE", inputs=["cond", "rng", "alt"], outputs=["y"], attrs={}),
            OpNode(
                id="n2",
                op="CONV2D",
                inputs=["x", "w", "b"],
                outputs=["z"],
                attrs={"strides": [2], "pads": [1, 1]},
            ),
        ],
        inputs=["cond", "alt", "x"],
        outputs=["z"],
        metadata={},
    )
    out = propagate_shapes(ir, runtime_values={"start": [0], "limit": [5], "delta": [1]})
    assert out.tensors["rng"].shape == [5]
    assert out.tensors["y"].shape == [5]
    assert out.tensors["z"].shape == [2, 32, 8]


def test_shape_prop_for_global_average_pool() -> None:
    ir = GraphIR(
        name="global_pool",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 64, 7, 7], "input"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="AVGPOOL",
                inputs=["x"],
                outputs=["y"],
                attrs={"global": 1},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir)
    assert out.tensors["y"].shape == [1, 64, 1, 1]


def test_shape_prop_for_constant_of_shape() -> None:
    ir = GraphIR(
        name="constant_of_shape",
        opset=18,
        tensors={
            "shape": TensorRef("shape", "int64", [2], "initializer"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="CONSTANT_OF_SHAPE",
                inputs=["shape"],
                outputs=["y"],
                attrs={"value": 1.0},
            )
        ],
        inputs=[],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir, runtime_values={"shape": [2, 3]})
    assert out.tensors["y"].shape == [2, 3]


def test_shape_prop_for_pad() -> None:
    ir = GraphIR(
        name="pad",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 16, 20, 20], "input"),
            "pads": TensorRef("pads", "int64", [8], "initializer"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="PAD",
                inputs=["x", "pads"],
                outputs=["y"],
                attrs={"mode": "constant"},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir, runtime_values={"pads": [0, 0, 1, 2, 0, 0, 3, 4]})
    assert out.tensors["y"].shape == [1, 16, 24, 26]


def test_shape_prop_for_expand_preserves_non_unit_input_dims() -> None:
    ir = GraphIR(
        name="expand_broadcast",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 512], "input"),
            "shape": TensorRef("shape", "int64", [2], "initializer"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[OpNode(id="n0", op="EXPAND", inputs=["x", "shape"], outputs=["y"], attrs={})],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir, runtime_values={"shape": [1, 1]})
    assert out.tensors["y"].shape == [1, 512]


def test_shape_prop_for_resize_uses_scale_slot_when_sizes_missing() -> None:
    ir = GraphIR(
        name="resize_scale_slot",
        opset=18,
        tensors={
            "x": TensorRef("x", "float32", [1, 64, 20, 20], "input"),
            "scales": TensorRef("scales", "float32", [4], "initializer"),
            "y": TensorRef("y", "float32", [], "output"),
        },
        nodes=[
            OpNode(
                id="n0",
                op="UPSAMPLE",
                inputs=["x", "scales"],
                outputs=["y"],
                attrs={"mode": "nearest", "input_slots": [0, 2]},
            )
        ],
        inputs=["x"],
        outputs=["y"],
        metadata={},
    )
    out = propagate_shapes(ir, runtime_values={"scales": [1.0, 1.0, 2.0, 2.0]})
    assert out.tensors["y"].shape == [1, 64, 40, 40]
