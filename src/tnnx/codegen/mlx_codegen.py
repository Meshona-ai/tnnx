# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

from ..ir.types import GraphIR, OpNode
from .common import attr_float as _attr_float
from .common import attr_int as _attr_int
from .common import attr_int_list as _attr_int_list
from .common import order_nodes_for_emission


def _tensor_expr(name: str, ir: GraphIR) -> str:
    tensor = ir.tensors[name]
    if tensor.kind == "initializer":
        return f'params["{name}"]'
    return f'tensors["{name}"]'


def _mlx_conv_weight_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return f"__tnnx_mlx_conv_weight__{safe}"


def _mlx_conv_weight_expr(name: str, ir: GraphIR) -> tuple[str, bool]:
    tensor = ir.tensors.get(name)
    if tensor is None or tensor.kind != "initializer":
        return _tensor_expr(name, ir), False
    if len(tensor.shape) not in {3, 4}:
        return _tensor_expr(name, ir), False
    return f'params["{_mlx_conv_weight_name(name)}"]', True


def _ops_in_graph(ir: GraphIR) -> set[str]:
    return {node.op for node in ir.nodes}


def _input_rank(name: str, ir: GraphIR) -> int:
    tensor = ir.tensors.get(name)
    return len(tensor.shape) if tensor is not None else 0


def _requires_conv1d_helper(ir: GraphIR) -> bool:
    for node in ir.nodes:
        if node.op == "CONV2D" and node.inputs and _input_rank(node.inputs[0], ir) == 3:
            return True
    return False


def _requires_conv2d_helper(ir: GraphIR) -> bool:
    for node in ir.nodes:
        if node.op != "CONV2D" or not node.inputs:
            continue
        if _input_rank(node.inputs[0], ir) != 3:
            return True
    return False


def _mlx_prepacked_conv_weights(ir: GraphIR) -> list[tuple[str, str, int]]:
    weights: dict[str, tuple[str, int]] = {}
    for node in ir.nodes:
        if node.op != "CONV2D" or len(node.inputs) < 2:
            continue
        weight_name = node.inputs[1]
        tensor = ir.tensors.get(weight_name)
        if tensor is None or tensor.kind != "initializer":
            continue
        rank = len(tensor.shape)
        if rank in {3, 4}:
            weights[weight_name] = (_mlx_conv_weight_name(weight_name), rank)
    return [(source, packed, rank) for source, (packed, rank) in sorted(weights.items())]


def _clip_arg(node: OpNode, ins: list[str], input_idx: int, attr_name: str) -> str:
    if len(ins) > input_idx:
        return ins[input_idx]
    if attr_name in node.attrs:
        return repr(node.attrs[attr_name])
    return "None"


def _slot_exprs(node: OpNode, ins: list[str]) -> dict[int, str]:
    slots = _attr_int_list(node.attrs, "input_slots", list(range(len(ins))))
    return {slot: expr for slot, expr in zip(slots, ins, strict=False)}


def _emit_node_lines(node: OpNode, ir: GraphIR) -> list[str]:
    expr = _emit_node_expr(node, ir)
    if len(node.outputs) == 1:
        return [f'    tensors["{node.outputs[0]}"] = {expr}']
    tmp_name = f"_node_{node.id}"
    lines = [f"    {tmp_name} = {expr}"]
    for idx, output_name in enumerate(node.outputs):
        lines.append(f'    tensors["{output_name}"] = {tmp_name}[{idx}]')
    return lines


def _emit_node_expr(node: OpNode, ir: GraphIR) -> str:
    ins = [_tensor_expr(name, ir) for name in node.inputs]
    if node.op == "ADD":
        return f"mx.add({ins[0]}, {ins[1]})"
    if node.op == "AND":
        return f"_onnx_and({ins[0]}, {ins[1]})"
    if node.op == "SUB":
        return f"mx.subtract({ins[0]}, {ins[1]})"
    if node.op == "DIV":
        return f"mx.divide({ins[0]}, {ins[1]})"
    if node.op == "EQUAL":
        return f"mx.equal({ins[0]}, {ins[1]})"
    if node.op == "GREATER":
        return f"mx.greater({ins[0]}, {ins[1]})"
    if node.op == "GREATEROREQUAL":
        return f"mx.greater_equal({ins[0]}, {ins[1]})"
    if node.op == "ERF":
        return f"mx.erf({ins[0]})"
    if node.op == "EXP":
        return f"mx.exp({ins[0]})"
    if node.op == "EXPAND":
        return f"_onnx_expand({ins[0]}, {ins[1]})"
    if node.op == "COS":
        return f"mx.cos({ins[0]})"
    if node.op == "CUMSUM":
        exclusive = _attr_int(node.attrs, "exclusive", 0) == 1
        reverse = _attr_int(node.attrs, "reverse", 0) == 1
        return f"_onnx_cumsum({ins[0]}, {ins[1]}, exclusive={exclusive}, reverse={reverse})"
    if node.op == "SIN":
        return f"mx.sin({ins[0]})"
    if node.op == "SIGMOID":
        return f"(1.0 / (1.0 + mx.exp(-{ins[0]})))"
    if node.op == "SILU":
        return f"({ins[0]} * (1.0 / (1.0 + mx.exp(-{ins[0]}))))"
    if node.op == "SOFTPLUS":
        return f"mx.log(1.0 + mx.exp({ins[0]}))"
    if node.op == "SQRT":
        return f"mx.sqrt({ins[0]})"
    if node.op == "TANH":
        return f"mx.tanh({ins[0]})"
    if node.op == "MISH":
        return f"({ins[0]} * mx.tanh(mx.log(1.0 + mx.exp({ins[0]}))))"
    if node.op == "GATHER":
        axis = _attr_int(node.attrs, "axis", 0)
        return f"mx.take({ins[0]}, mx.array({ins[1]}), axis={axis})"
    if node.op == "GATHERELEMENTS":
        axis = _attr_int(node.attrs, "axis", 0)
        return f"_onnx_gather_elements({ins[0]}, {ins[1]}, axis={axis})"
    if node.op == "GROUPQUERYATTENTION":
        slot_map = _slot_exprs(node, ins)
        attention_bias = slot_map.get(10, "None")
        num_heads = _attr_int(node.attrs, "num_heads", 1)
        kv_num_heads = _attr_int(node.attrs, "kv_num_heads", 1)
        scale = _attr_float(node.attrs, "scale", 1.0)
        softcap = _attr_float(node.attrs, "softcap", 0.0)
        return (
            "_onnx_group_query_attention("
            f"{ins[0]}, {ins[1]}, {ins[2]}, {ins[3]}, {ins[4]}, "
            f"attention_bias={attention_bias}, num_heads={num_heads}, "
            f"kv_num_heads={kv_num_heads}, scale={scale}, softcap={softcap})"
        )
    if node.op == "IDENTITY":
        return ins[0]
    if node.op == "ISNAN":
        return f"mx.isnan({ins[0]})"
    if node.op == "LESS":
        return f"mx.less({ins[0]}, {ins[1]})"
    if node.op == "LESSOREQUAL":
        return f"mx.less_equal({ins[0]}, {ins[1]})"
    if node.op == "LOG":
        return f"mx.log({ins[0]})"
    if node.op == "MUL":
        return f"mx.multiply({ins[0]}, {ins[1]})"
    if node.op == "MOD":
        fmod = _attr_int(node.attrs, "fmod", 0) == 1
        return f"_onnx_mod({ins[0]}, {ins[1]}, fmod={fmod})"
    if node.op == "NEG":
        return f"(-{ins[0]})"
    if node.op == "PAD":
        slot_map = _slot_exprs(node, ins)
        constant_value = slot_map.get(2, "None")
        axes = slot_map.get(3, "None")
        mode = repr(str(node.attrs.get("mode", "constant")).lower())
        return f"_onnx_pad({ins[0]}, {ins[1]}, {constant_value}, {axes}, mode={mode})"
    if node.op == "POW":
        return f"_onnx_pow({ins[0]}, {ins[1]})"
    if node.op == "RECIPROCAL":
        return f"(1.0 / {ins[0]})"
    if node.op == "MATMUL":
        return f"mx.matmul({ins[0]}, {ins[1]})"
    if node.op == "GEMM":
        left = ins[0]
        right = ins[1]
        if _attr_int(node.attrs, "transA", 0) == 1:
            left = f"mx.swapaxes({left}, -1, -2)"
        if _attr_int(node.attrs, "transB", 0) == 1:
            right = f"mx.swapaxes({right}, -1, -2)"
        out = f"mx.matmul({left}, {right})"
        if len(ins) > 2:
            out = f"({out} + {ins[2]})"
        return out
    if node.op == "REDUCEMEAN":
        axes_expr = "None"
        if isinstance(node.attrs.get("axes"), list):
            axes_expr = repr(tuple(_attr_int_list(node.attrs, "axes", [])))
        elif len(ins) > 1:
            axes_expr = ins[1]
        keepdims = _attr_int(node.attrs, "keepdims", 1) == 1
        return f"_onnx_reduce_mean({ins[0]}, {axes_expr}, keepdims={keepdims})"
    if node.op == "REDUCESUM":
        axes_expr = "None"
        if isinstance(node.attrs.get("axes"), list):
            axes_expr = repr(tuple(_attr_int_list(node.attrs, "axes", [])))
        elif len(ins) > 1:
            axes_expr = ins[1]
        keepdims = _attr_int(node.attrs, "keepdims", 1) == 1
        return f"_onnx_reduce_sum({ins[0]}, {axes_expr}, keepdims={keepdims})"
    if node.op == "RELU":
        return f"mx.maximum({ins[0]}, 0)"
    if node.op == "RELU6":
        return f"mx.minimum(mx.maximum({ins[0]}, 0), 6)"
    if node.op == "CLIP":
        return f"_onnx_clip({ins[0]}, {_clip_arg(node, ins, 1, 'min')}, {_clip_arg(node, ins, 2, 'max')})"
    if node.op == "GELU":
        return f"(0.5 * {ins[0]} * (1.0 + mx.erf({ins[0]} / mx.sqrt(2.0))))"
    if node.op == "LAYERNORM":
        axis = _attr_int(node.attrs, "axis", -1)
        eps = _attr_float(node.attrs, "epsilon", 1e-5)
        mean = f"mx.mean({ins[0]}, axis={axis}, keepdims=True)"
        var = f"mx.var({ins[0]}, axis={axis}, keepdims=True)"
        out = f"(({ins[0]} - {mean}) / mx.sqrt({var} + {eps})) * {ins[1]}"
        if len(ins) > 2:
            out = f"({out} + {ins[2]})"
        return out
    if node.op == "RMSNORM":
        axis = _attr_int(node.attrs, "axis", -1)
        eps = _attr_float(node.attrs, "epsilon", 1e-5)
        rms = f"mx.mean({ins[0]} * {ins[0]}, axis={axis}, keepdims=True)"
        out = f"({ins[0]} / mx.sqrt({rms} + {eps})) * {ins[1]}"
        if len(ins) > 2:
            out = f"({out} + {ins[2]})"
        return out
    if node.op == "ROTARYEMBEDDING":
        rotary_dim = _attr_int(node.attrs, "rotary_embedding_dim", 0)
        num_heads = _attr_int(node.attrs, "num_heads", 0)
        interleaved = _attr_int(node.attrs, "interleaved", 0) == 1
        return (
            "_onnx_rotary_embedding("
            f"{ins[0]}, {ins[1]}, {ins[2]}, {ins[3]}, "
            f"rotary_dim={rotary_dim}, num_heads={num_heads}, interleaved={interleaved})"
        )
    if node.op == "BATCHNORM":
        eps = _attr_float(node.attrs, "epsilon", 1e-5)
        return f"_onnx_batchnorm({ins[0]}, {ins[1]}, {ins[2]}, {ins[3]}, {ins[4]}, epsilon={eps})"
    if node.op == "INSTANCENORM":
        eps = _attr_float(node.attrs, "epsilon", 1e-5)
        return f"_onnx_instancenorm({ins[0]}, {ins[1]}, {ins[2]}, epsilon={eps})"
    if node.op == "CAST":
        return f"_onnx_cast({ins[0]}, {_attr_int(node.attrs, 'to', -1)})"
    if node.op == "RESHAPE":
        allowzero = _attr_int(node.attrs, "allowzero", 0) == 1
        return f"_onnx_reshape({ins[0]}, {ins[1]}, allowzero={allowzero})"
    if node.op == "SHAPE":
        start = _attr_int(node.attrs, "start", 0)
        end = node.attrs.get("end")
        end_expr = "None" if end is None else str(_attr_int(node.attrs, "end", 0))
        return f"_onnx_shape({ins[0]}, start={start}, end={end_expr})"
    if node.op == "FLATTEN":
        return f"_onnx_flatten({ins[0]}, axis={_attr_int(node.attrs, 'axis', 1)})"
    if node.op == "CONCAT":
        if len(ins) == 1:
            return ins[0]
        joined = ", ".join(ins)
        return f"mx.concatenate(({joined}), axis={_attr_int(node.attrs, 'axis', 0)})"
    if node.op == "SCATTERND":
        return f"_onnx_scatter_nd({ins[0]}, {ins[1]}, {ins[2]})"
    if node.op == "CONSTANT_OF_SHAPE":
        value = repr(node.attrs.get("value", 0.0))
        return f"_onnx_constant_of_shape({ins[0]}, value={value})"
    if node.op == "UNSQUEEZE":
        if isinstance(node.attrs.get("axes"), list):
            return f"_onnx_unsqueeze({ins[0]}, {tuple(_attr_int_list(node.attrs, 'axes', []))})"
        if len(ins) > 1:
            return f"_onnx_unsqueeze({ins[0]}, {ins[1]})"
        raise ValueError(f"UNSQUEEZE node {node.id} is missing axes.")
    if node.op == "SQUEEZE":
        if isinstance(node.attrs.get("axes"), list):
            return f"_onnx_squeeze({ins[0]}, {tuple(_attr_int_list(node.attrs, 'axes', []))})"
        if len(ins) > 1:
            return f"_onnx_squeeze({ins[0]}, {ins[1]})"
        return f"_onnx_squeeze({ins[0]})"
    if node.op == "SLICE":
        axes = ins[3] if len(ins) > 3 else "None"
        steps = ins[4] if len(ins) > 4 else "None"
        return f"_onnx_slice({ins[0]}, {ins[1]}, {ins[2]}, {axes}, {steps})"
    if node.op == "SPLIT":
        split_expr = "None"
        if len(ins) > 1:
            split_expr = ins[1]
        elif isinstance(node.attrs.get("split"), list):
            split_expr = repr(tuple(_attr_int_list(node.attrs, "split", [])))
        axis = _attr_int(node.attrs, "axis", 0)
        num_outputs = _attr_int(node.attrs, "num_outputs", len(node.outputs))
        return f"_onnx_split({ins[0]}, {split_expr}, axis={axis}, num_outputs={num_outputs})"
    if node.op == "TRANSPOSE":
        perm = node.attrs.get("perm")
        if perm is None:
            return f"mx.transpose({ins[0]})"
        return f"mx.transpose({ins[0]}, axes={tuple(_attr_int_list(node.attrs, 'perm', []))})"
    if node.op == "TRILU":
        diagonal = ins[1] if len(ins) > 1 else "None"
        upper = _attr_int(node.attrs, "upper", 1) == 1
        return f"_onnx_trilu({ins[0]}, {diagonal}, upper={upper})"
    if node.op == "SOFTMAX":
        axis = _attr_int(node.attrs, "axis", -1)
        return (
            f"(lambda t: t / mx.sum(t, axis={axis}, keepdims=True))"
            f"(mx.exp({ins[0]} - mx.max({ins[0]}, axis={axis}, keepdims=True)))"
        )
    if node.op == "SKIPRMSNORM":
        eps = _attr_float(node.attrs, "epsilon", 1e-5)
        return_residual = len(node.outputs) > 1
        return (
            f"_onnx_skip_rmsnorm({ins[0]}, {ins[1]}, {ins[2]}, "
            f"epsilon={eps}, return_residual={return_residual})"
        )
    if node.op in {"AVGPOOL", "MAXPOOL"}:
        rank = _input_rank(node.inputs[0], ir) if node.inputs else 0
        spatial = max(rank - 2, 1)
        kernel = tuple(_attr_int_list(node.attrs, "kernel_shape", [1] * spatial))
        strides = tuple(_attr_int_list(node.attrs, "strides", [1] * spatial))
        pads = tuple(_attr_int_list(node.attrs, "pads", [0] * (spatial * 2)))
        global_pool = _attr_int(node.attrs, "global", 0) == 1
        kind = "avg" if node.op == "AVGPOOL" else "max"
        return (
            f"_onnx_pool({ins[0]}, kind='{kind}', kernel={kernel}, strides={strides}, "
            f"pads={pads}, global_pool={global_pool})"
        )
    if node.op == "UPSAMPLE":
        slot_map = _slot_exprs(node, ins)
        roi = slot_map.get(1, "None")
        scales = slot_map.get(2, "None")
        sizes = slot_map.get(3, "None")
        if scales == "None" and sizes == "None" and len(ins) == 2 and 1 not in slot_map:
            scales = ins[1]
        return f"_onnx_resize_nearest({ins[0]}, roi={roi}, scales={scales}, sizes={sizes})"
    if node.op == "ARANGE":
        return f"_onnx_arange({ins[0]}, {ins[1]}, {ins[2]})"
    if node.op == "WHERE":
        return f"mx.where({ins[0]}, {ins[1]}, {ins[2]})"
    if node.op == "QUANTIZE":
        zero_point = ins[2] if len(ins) > 2 else "None"
        return f"_onnx_quantize({ins[0]}, {ins[1]}, {zero_point})"
    if node.op == "DEQUANTIZE":
        zero_point = ins[2] if len(ins) > 2 else "None"
        return f"_onnx_dequantize({ins[0]}, {ins[1]}, {zero_point})"
    if node.op == "CONV2D":
        rank = _input_rank(node.inputs[0], ir) if node.inputs else 0
        groups = _attr_int(node.attrs, "group", 1)
        weight_expr, weight_prepacked = (
            _mlx_conv_weight_expr(node.inputs[1], ir) if len(node.inputs) > 1 else ("None", False)
        )
        weight_flag = ", weight_prepacked=True" if weight_prepacked else ""
        if rank == 3:
            strides_1d = _attr_int_list(node.attrs, "strides", [1])
            dilations_1d = _attr_int_list(node.attrs, "dilations", [1])
            pads_1d = _attr_int_list(node.attrs, "pads", [0, 0])
            stride = strides_1d[0] if strides_1d else 1
            dilation = dilations_1d[0] if dilations_1d else 1
            if dilation != 1:
                raise ValueError("MLX CONV1D codegen supports dilation=1 only in v0.")
            if len(pads_1d) >= 2:
                pad_l, pad_r = pads_1d[0], pads_1d[1]
            elif len(pads_1d) == 1:
                pad_l = pads_1d[0]
                pad_r = pads_1d[0]
            else:
                pad_l = 0
                pad_r = 0
            if len(ins) > 2:
                return (
                    f"_onnx_conv1d({ins[0]}, {weight_expr}, {ins[2]}, "
                    f"stride={stride}, padding=({pad_l}, {pad_r}), "
                    f"dilation={dilation}, groups={groups}{weight_flag})"
                )
            return (
                f"_onnx_conv1d({ins[0]}, {weight_expr}, None, "
                f"stride={stride}, padding=({pad_l}, {pad_r}), "
                f"dilation={dilation}, groups={groups}{weight_flag})"
            )

        strides_2d = tuple(_attr_int_list(node.attrs, "strides", [1, 1]))
        dilations_2d = tuple(_attr_int_list(node.attrs, "dilations", [1, 1]))
        pads_2d = _attr_int_list(node.attrs, "pads", [0, 0, 0, 0])
        if len(pads_2d) == 4:
            top, pad_left, bottom, pad_right = pads_2d
            if top != bottom or pad_left != pad_right:
                raise ValueError("MLX CONV2D codegen supports symmetric ONNX pads only in v0.")
            padding = (top, pad_left)
        elif len(pads_2d) == 2:
            padding = (pads_2d[0], pads_2d[1])
        else:
            padding = (0, 0)
        if len(ins) > 2:
            return (
                f"_onnx_conv2d({ins[0]}, {weight_expr}, {ins[2]}, "
                f"stride={strides_2d}, padding={padding}, "
                f"dilation={dilations_2d}, groups={groups}{weight_flag})"
            )
        return (
            f"_onnx_conv2d({ins[0]}, {weight_expr}, None, "
            f"stride={strides_2d}, padding={padding}, "
            f"dilation={dilations_2d}, groups={groups}{weight_flag})"
        )
    raise ValueError(f"Unsupported op for MLX codegen: {node.op}")


def render_mlx_module(ir: GraphIR, *, entrypoint: str = "forward") -> str:
    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "import mlx.core as mx",
        "",
    ]

    ops = _ops_in_graph(ir)

    if "SLICE" in ops:
        lines.extend(
            [
                "def _onnx_slice(data, starts, ends, axes=None, steps=None):",
                "    arr = np.asarray(data)",
                "    starts_l = [int(v) for v in np.asarray(starts).tolist()]",
                "    ends_l = [int(v) for v in np.asarray(ends).tolist()]",
                "    if axes is None:",
                "        axes_l = list(range(len(starts_l)))",
                "    else:",
                "        axes_l = [int(v) for v in np.asarray(axes).tolist()]",
                "    if steps is None:",
                "        steps_l = [1] * len(starts_l)",
                "    else:",
                "        steps_l = [int(v) for v in np.asarray(steps).tolist()]",
                "    slices = [slice(None)] * arr.ndim",
                "    for s, e, a, st in zip(starts_l, ends_l, axes_l, steps_l, strict=False):",
                "        slices[a] = slice(s, e, st)",
                "    return mx.array(arr[tuple(slices)])",
                "",
            ]
        )

    if "SPLIT" in ops:
        lines.extend(
            [
                "def _onnx_split(data, split=None, *, axis=0, num_outputs=1):",
                "    arr = np.asarray(data)",
                "    axis_i = int(axis)",
                "    if axis_i < 0:",
                "        axis_i += arr.ndim",
                "    if axis_i < 0 or axis_i >= arr.ndim:",
                "        raise ValueError(f'Split axis out of range: {axis}')",
                "    if split is None:",
                "        count = int(num_outputs)",
                "        if count <= 0:",
                "            raise ValueError('Split requires num_outputs > 0 when split is omitted.')",
                "        parts = np.split(arr, count, axis=axis_i)",
                "        return tuple(mx.array(part) for part in parts)",
                "    split_l = [int(v) for v in np.asarray(split).reshape(-1).tolist()]",
                "    if not split_l:",
                "        count = int(num_outputs)",
                "        if count <= 0:",
                "            raise ValueError('Split requires num_outputs > 0 when split is empty.')",
                "        parts = np.split(arr, count, axis=axis_i)",
                "        return tuple(mx.array(part) for part in parts)",
                "    if len(split_l) == 1:",
                "        return (mx.array(arr),)",
                "    indices = np.cumsum(split_l[:-1], dtype=int).tolist()",
                "    parts = np.split(arr, indices, axis=axis_i)",
                "    return tuple(mx.array(part) for part in parts)",
                "",
            ]
        )

    if "AND" in ops:
        lines.extend(
            [
                "def _onnx_and(lhs, rhs):",
                "    return mx.array(np.logical_and(np.asarray(lhs), np.asarray(rhs)))",
                "",
            ]
        )

    if "CAST" in ops:
        lines.extend(
            [
                "_ONNX_DTYPE_MAP = {",
                "    1: np.float32,",
                "    2: np.uint8,",
                "    3: np.int8,",
                "    4: np.uint16,",
                "    5: np.int16,",
                "    6: np.int32,",
                "    7: np.int64,",
                "    9: np.bool_,",
                "    10: np.float16,",
                "    11: np.float64,",
                "    12: np.uint32,",
                "    13: np.uint64,",
                "}",
                "",
                "def _onnx_cast(data, to):",
                "    dtype = _ONNX_DTYPE_MAP.get(int(to))",
                "    if dtype is None:",
                "        raise ValueError(f'Unsupported ONNX Cast dtype enum: {to}')",
                "    return mx.array(np.asarray(data).astype(dtype))",
                "",
            ]
        )

    if "CUMSUM" in ops:
        lines.extend(
            [
                "def _onnx_cumsum(data, axis, *, exclusive=False, reverse=False):",
                "    arr = np.asarray(data)",
                "    axis_i = int(np.asarray(axis).reshape(-1)[0])",
                "    if reverse:",
                "        arr = np.flip(arr, axis=axis_i)",
                "    out = np.cumsum(arr, axis=axis_i)",
                "    if exclusive:",
                "        shifted = np.zeros_like(out)",
                "        head = [slice(None)] * out.ndim",
                "        tail = [slice(None)] * out.ndim",
                "        head[axis_i] = slice(1, None)",
                "        tail[axis_i] = slice(0, -1)",
                "        shifted[tuple(head)] = out[tuple(tail)]",
                "        out = shifted",
                "    if reverse:",
                "        out = np.flip(out, axis=axis_i)",
                "    return mx.array(out)",
                "",
            ]
        )

    if "RESHAPE" in ops:
        lines.extend(
            [
                "def _onnx_reshape(data, shape, *, allowzero=False):",
                "    target = [int(v) for v in np.asarray(shape).reshape(-1).tolist()]",
                "    if not allowzero:",
                "        data_shape = list(np.asarray(data).shape)",
                "        for idx, dim in enumerate(target):",
                "            if dim == 0:",
                "                if idx >= len(data_shape):",
                "                    raise ValueError('Reshape zero-copy index exceeds input rank.')",
                "                target[idx] = int(data_shape[idx])",
                "    return mx.reshape(data, tuple(target))",
                "",
            ]
        )

    if "UNSQUEEZE" in ops:
        lines.extend(
            [
                "def _onnx_unsqueeze(data, axes):",
                "    axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]",
                "    out_rank = data.ndim + len(axes_l)",
                "    normalized = sorted((axis if axis >= 0 else axis + out_rank) for axis in axes_l)",
                "    out = data",
                "    for axis in normalized:",
                "        out = mx.expand_dims(out, axis)",
                "    return out",
                "",
            ]
        )

    if "SQUEEZE" in ops:
        lines.extend(
            [
                "def _onnx_squeeze(data, axes=None):",
                "    if axes is None:",
                "        return mx.squeeze(data)",
                "    axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]",
                "    if not axes_l:",
                "        return data",
                "    normalized = []",
                "    for axis in axes_l:",
                "        axis_i = axis if axis >= 0 else data.ndim + axis",
                "        if axis_i < 0 or axis_i >= data.ndim:",
                "            raise ValueError(f'Squeeze axis out of range: {axis}')",
                "        normalized.append(axis_i)",
                "    return mx.squeeze(data, axis=tuple(sorted(set(normalized))))",
                "",
            ]
        )

    if "CONSTANT_OF_SHAPE" in ops:
        lines.extend(
            [
                "def _onnx_constant_of_shape(shape, *, value=0.0):",
                "    dims = tuple(int(v) for v in np.asarray(shape).reshape(-1).tolist())",
                "    return mx.array(np.full(dims, value, dtype=np.float32))",
                "",
            ]
        )

    if "EXPAND" in ops:
        lines.extend(
            [
                "def _onnx_expand(data, shape):",
                "    arr = np.asarray(data)",
                "    requested = [int(v) for v in np.asarray(shape).reshape(-1).tolist()]",
                "    data_shape = list(arr.shape)",
                "    if len(requested) < len(data_shape):",
                "        requested = [1] * (len(data_shape) - len(requested)) + requested",
                "    elif len(requested) > len(data_shape):",
                "        expand_rank = len(requested) - len(data_shape)",
                "        arr = np.reshape(arr, (1,) * expand_rank + tuple(data_shape))",
                "        data_shape = [1] * expand_rank + data_shape",
                "    dims = []",
                "    for data_dim, target_dim in zip(data_shape, requested, strict=False):",
                "        if target_dim < 0 or target_dim == 1 and data_dim > 1:",
                "            dims.append(int(data_dim))",
                "            continue",
                "        if data_dim not in {1, target_dim}:",
                "            raise ValueError(",
                "                f'Incompatible shapes for ONNX Expand: {tuple(data_shape)} and {tuple(requested)}'",
                "            )",
                "        dims.append(int(target_dim))",
                "    return mx.array(np.broadcast_to(arr, tuple(dims)))",
                "",
            ]
        )

    if "PAD" in ops:
        lines.extend(
            [
                "def _onnx_pad(data, pads, constant_value=None, axes=None, *, mode='constant'):",
                "    arr = np.asarray(data)",
                "    pads_l = [int(v) for v in np.asarray(pads).reshape(-1).tolist()]",
                "    if len(pads_l) % 2 != 0:",
                "        raise ValueError('Pad helper expects begin/end pad pairs.')",
                "    if axes is None:",
                "        axes_l = list(range(len(pads_l) // 2))",
                "    else:",
                "        axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]",
                "    if len(axes_l) * 2 != len(pads_l):",
                "        raise ValueError('Pad helper received mismatched pads and axes lengths.')",
                "    pad_cfg = [(0, 0)] * arr.ndim",
                "    half = len(pads_l) // 2",
                "    for idx, axis in enumerate(axes_l):",
                "        axis_i = axis if axis >= 0 else arr.ndim + axis",
                "        if axis_i < 0 or axis_i >= arr.ndim:",
                "            raise ValueError(f'Pad axis out of range: {axis}')",
                "        pad_cfg[axis_i] = (pads_l[idx], pads_l[idx + half])",
                "    mode_l = str(mode).lower()",
                "    if mode_l == 'constant':",
                "        cval = 0.0 if constant_value is None else float(np.asarray(constant_value).reshape(-1)[0])",
                "        return mx.array(np.pad(arr, tuple(pad_cfg), mode='constant', constant_values=cval))",
                "    if mode_l in {'edge', 'reflect'}:",
                "        return mx.array(np.pad(arr, tuple(pad_cfg), mode=mode_l))",
                "    raise ValueError(f'Unsupported ONNX Pad mode: {mode}')",
                "",
            ]
        )

    if "POW" in ops:
        lines.extend(
            [
                "def _onnx_pow(lhs, rhs):",
                "    return mx.power(lhs, rhs)",
                "",
            ]
        )

    if "REDUCEMEAN" in ops:
        lines.extend(
            [
                "def _onnx_reduce_mean(data, axes=None, *, keepdims=True):",
                "    if axes is None:",
                "        axes_t = None",
                "    else:",
                "        axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]",
                "        axes_t = tuple(axes_l) if axes_l else None",
                "    return mx.array(np.mean(np.asarray(data), axis=axes_t, keepdims=keepdims))",
                "",
            ]
        )

    if "REDUCESUM" in ops:
        lines.extend(
            [
                "def _onnx_reduce_sum(data, axes=None, *, keepdims=True):",
                "    if axes is None:",
                "        axes_t = None",
                "    else:",
                "        axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]",
                "        axes_t = tuple(axes_l) if axes_l else None",
                "    return mx.array(np.sum(np.asarray(data), axis=axes_t, keepdims=keepdims))",
                "",
            ]
        )

    if "SHAPE" in ops:
        lines.extend(
            [
                "def _onnx_shape(data, *, start=0, end=None):",
                "    dims = list(np.asarray(data).shape)",
                "    return mx.array(np.asarray(dims[slice(int(start), None if end is None else int(end))], dtype=np.int64))",
                "",
            ]
        )

    if "FLATTEN" in ops:
        lines.extend(
            [
                "def _onnx_flatten(data, axis=1):",
                "    axis_i = int(axis)",
                "    if axis_i < 0:",
                "        axis_i += data.ndim",
                "    left = int(np.prod(data.shape[:axis_i], dtype=int)) if axis_i > 0 else 1",
                "    right = int(np.prod(data.shape[axis_i:], dtype=int)) if axis_i < data.ndim else 1",
                "    return mx.reshape(data, (left, right))",
                "",
            ]
        )

    if "GATHERELEMENTS" in ops:
        lines.extend(
            [
                "def _onnx_gather_elements(data, indices, axis=0):",
                "    return mx.array(",
                "        np.take_along_axis(",
                "            np.asarray(data),",
                "            np.asarray(indices, dtype=np.int64),",
                "            axis=int(axis),",
                "        )",
                "    )",
                "",
            ]
        )

    if "ROTARYEMBEDDING" in ops:
        lines.extend(
            [
                "def _onnx_rotary_embedding(data, position_ids, cos_cache, sin_cache, *, rotary_dim, num_heads, interleaved=False):",
                "    del num_heads",
                "    if interleaved:",
                "        raise ValueError('RotaryEmbedding interleaved mode is not supported yet.')",
                "    arr = np.asarray(data)",
                "    rotary_i = int(rotary_dim)",
                "    if rotary_i <= 0:",
                "        return mx.array(arr)",
                "    half = rotary_i // 2",
                "    pos = np.asarray(position_ids, dtype=np.int64)",
                "    cos = np.take(np.asarray(cos_cache), pos, axis=0)",
                "    sin = np.take(np.asarray(sin_cache), pos, axis=0)",
                "    cos = np.expand_dims(cos, axis=1)",
                "    sin = np.expand_dims(sin, axis=1)",
                "    left = arr[..., :half]",
                "    right = arr[..., half:rotary_i]",
                "    rotated = np.concatenate((",
                "        left * cos - right * sin,",
                "        right * cos + left * sin,",
                "    ), axis=-1)",
                "    out = np.concatenate((rotated, arr[..., rotary_i:]), axis=-1)",
                "    return mx.array(out.astype(arr.dtype, copy=False))",
                "",
            ]
        )

    if "SKIPRMSNORM" in ops:
        lines.extend(
            [
                "def _onnx_skip_rmsnorm(data, skip, gamma, epsilon=1e-5, *, return_residual=True):",
                "    data_arr = np.asarray(data)",
                "    skip_arr = np.asarray(skip)",
                "    residual = data_arr + skip_arr",
                "    rms = np.mean(residual * residual, axis=-1, keepdims=True)",
                "    normalized = (residual / np.sqrt(rms + float(epsilon))) * np.asarray(gamma)",
                "    if return_residual:",
                "        return mx.array(normalized), mx.array(residual)",
                "    return mx.array(normalized)",
                "",
            ]
        )

    if "GROUPQUERYATTENTION" in ops:
        lines.extend(
            [
                "def _onnx_group_query_attention(",
                "    query,",
                "    key,",
                "    value,",
                "    past_key,",
                "    past_value,",
                "    attention_bias=None,",
                "    *,",
                "    num_heads=1,",
                "    kv_num_heads=1,",
                "    scale=1.0,",
                "    softcap=0.0,",
                "):",
                "    q = np.asarray(query)",
                "    k = np.asarray(key)",
                "    v = np.asarray(value)",
                "    pk = np.asarray(past_key)",
                "    pv = np.asarray(past_value)",
                "    batch, query_len, q_width = q.shape",
                "    num_heads_i = int(num_heads)",
                "    kv_heads_i = int(kv_num_heads)",
                "    if num_heads_i <= 0 or kv_heads_i <= 0 or q_width % num_heads_i != 0:",
                "        raise ValueError('Invalid GroupQueryAttention head configuration.')",
                "    head_dim = q_width // num_heads_i",
                "    if k.shape[-1] != kv_heads_i * head_dim or v.shape[-1] != kv_heads_i * head_dim:",
                "        raise ValueError('GroupQueryAttention key/value width mismatch.')",
                "    if num_heads_i % kv_heads_i != 0:",
                "        raise ValueError('num_heads must be divisible by kv_num_heads.')",
                "    qh = q.reshape(batch, query_len, num_heads_i, head_dim).transpose(0, 2, 1, 3)",
                "    kh = k.reshape(batch, query_len, kv_heads_i, head_dim).transpose(0, 2, 1, 3)",
                "    vh = v.reshape(batch, query_len, kv_heads_i, head_dim).transpose(0, 2, 1, 3)",
                "    present_key = np.concatenate((pk, kh), axis=2)",
                "    present_value = np.concatenate((pv, vh), axis=2)",
                "    repeats = num_heads_i // kv_heads_i",
                "    full_key = np.repeat(present_key, repeats, axis=1)",
                "    full_value = np.repeat(present_value, repeats, axis=1)",
                "    scores = np.einsum('bhqd,bhkd->bhqk', qh, full_key) * float(scale)",
                "    if float(softcap) > 0.0:",
                "        scores = np.tanh(scores / float(softcap)) * float(softcap)",
                "    if attention_bias is not None:",
                "        scores = scores + np.asarray(attention_bias, dtype=scores.dtype)",
                "    scores = scores - np.max(scores, axis=-1, keepdims=True)",
                "    probs = np.exp(scores)",
                "    probs = probs / np.sum(probs, axis=-1, keepdims=True)",
                "    context = np.einsum('bhqk,bhkd->bhqd', probs, full_value)",
                "    context = context.transpose(0, 2, 1, 3).reshape(batch, query_len, q_width)",
                "    return mx.array(context), mx.array(present_key), mx.array(present_value)",
                "",
            ]
        )

    if "BATCHNORM" in ops:
        lines.extend(
            [
                "def _onnx_batchnorm(data, scale, bias, mean, var, epsilon=1e-5):",
                "    reshape_dims = (1, -1) + (1,) * max(data.ndim - 2, 0)",
                "    scale_r = mx.reshape(scale, reshape_dims)",
                "    bias_r = mx.reshape(bias, reshape_dims)",
                "    mean_r = mx.reshape(mean, reshape_dims)",
                "    var_r = mx.reshape(var, reshape_dims)",
                "    return ((data - mean_r) / mx.sqrt(var_r + epsilon)) * scale_r + bias_r",
                "",
            ]
        )

    if "INSTANCENORM" in ops:
        lines.extend(
            [
                "def _onnx_instancenorm(data, scale, bias, epsilon=1e-5):",
                "    if data.ndim < 3:",
                "        raise ValueError('InstanceNormalization expects rank >= 3.')",
                "    reshape_dims = (1, -1) + (1,) * max(data.ndim - 2, 0)",
                "    scale_r = mx.reshape(scale, reshape_dims)",
                "    bias_r = mx.reshape(bias, reshape_dims)",
                "    reduce_axes = tuple(range(2, data.ndim))",
                "    mean = mx.mean(data, axis=reduce_axes, keepdims=True)",
                "    var = mx.var(data, axis=reduce_axes, keepdims=True)",
                "    normalized = (data - mean) / mx.sqrt(var + epsilon)",
                "    return normalized * scale_r + bias_r",
                "",
            ]
        )

    if ops & {"AVGPOOL", "MAXPOOL"}:
        lines.extend(
            [
                "def _onnx_pool(data, *, kind, kernel, strides, pads, global_pool=False):",
                "    if global_pool:",
                "        axes = tuple(range(2, data.ndim))",
                "        if kind == 'max':",
                "            return mx.max(data, axis=axes, keepdims=True)",
                "        return mx.mean(data, axis=axes, keepdims=True)",
                "    kernel_l = tuple(int(v) for v in kernel)",
                "    strides_l = tuple(int(v) for v in strides)",
                "    pads_l = tuple(int(v) for v in pads)",
                "    if data.ndim == 3:",
                "        pad_l, pad_r = (pads_l[0], pads_l[1]) if len(pads_l) >= 2 else (0, 0)",
                "        pad_cfg = ((0, 0), (0, 0), (pad_l, pad_r))",
                "        if kind == 'max':",
                "            padded = mx.contiguous(mx.pad(data, pad_cfg, constant_values=-float('inf')))",
                "        else:",
                "            padded = mx.contiguous(mx.pad(data, pad_cfg, constant_values=0.0))",
                "        n, c, length = padded.shape",
                "        out_len = ((length - kernel_l[0]) // strides_l[0]) + 1",
                "        base = (c * length, length, 1)",
                "        windows = mx.as_strided(",
                "            padded,",
                "            shape=(data.shape[0], data.shape[1], out_len, kernel_l[0]),",
                "            strides=(base[0], base[1], strides_l[0] * base[2], base[2]),",
                "        )",
                "        if kind == 'max':",
                "            return mx.max(windows, axis=-1)",
                "        ones = mx.ones_like(data).astype(mx.float32)",
                "        counts_padded = mx.contiguous(mx.pad(ones, pad_cfg, constant_values=0.0))",
                "        count_windows = mx.as_strided(",
                "            counts_padded,",
                "            shape=(data.shape[0], data.shape[1], out_len, kernel_l[0]),",
                "            strides=(base[0], base[1], strides_l[0] * base[2], base[2]),",
                "        )",
                "        total = mx.sum(windows, axis=-1)",
                "        counts = mx.sum(count_windows, axis=-1)",
                "        return total / mx.maximum(counts, 1.0)",
                "    if data.ndim != 4:",
                "        raise ValueError('Pooling helper supports NCW and NCHW tensors only.')",
                "    if len(pads_l) == 2:",
                "        pads_l = (pads_l[0], pads_l[1], pads_l[0], pads_l[1])",
                "    elif len(pads_l) < 4:",
                "        pads_l = (0, 0, 0, 0)",
                "    pad_cfg = ((0, 0), (0, 0), (pads_l[0], pads_l[2]), (pads_l[1], pads_l[3]))",
                "    if kind == 'max':",
                "        padded = mx.contiguous(mx.pad(data, pad_cfg, constant_values=-float('inf')))",
                "    else:",
                "        padded = mx.contiguous(mx.pad(data, pad_cfg, constant_values=0.0))",
                "    n, c, h, w = padded.shape",
                "    out_h = ((h - kernel_l[0]) // strides_l[0]) + 1",
                "    out_w = ((w - kernel_l[1]) // strides_l[1]) + 1",
                "    base = (c * h * w, h * w, w, 1)",
                "    windows = mx.as_strided(",
                "        padded,",
                "        shape=(data.shape[0], data.shape[1], out_h, out_w, kernel_l[0], kernel_l[1]),",
                "        strides=(",
                "            base[0],",
                "            base[1],",
                "            strides_l[0] * base[2],",
                "            strides_l[1] * base[3],",
                "            base[2],",
                "            base[3],",
                "        ),",
                "    )",
                "    if kind == 'max':",
                "        return mx.max(windows, axis=(-1, -2))",
                "    ones = mx.ones_like(data).astype(mx.float32)",
                "    counts_padded = mx.contiguous(mx.pad(ones, pad_cfg, constant_values=0.0))",
                "    count_windows = mx.as_strided(",
                "        counts_padded,",
                "        shape=(data.shape[0], data.shape[1], out_h, out_w, kernel_l[0], kernel_l[1]),",
                "        strides=(",
                "            base[0],",
                "            base[1],",
                "            strides_l[0] * base[2],",
                "            strides_l[1] * base[3],",
                "            base[2],",
                "            base[3],",
                "        ),",
                "    )",
                "    total = mx.sum(windows, axis=(-1, -2))",
                "    counts = mx.sum(count_windows, axis=(-1, -2))",
                "    return total / mx.maximum(counts, 1.0)",
                "",
            ]
        )

    if "SCATTERND" in ops:
        lines.extend(
            [
                "def _onnx_scatter_nd(data, indices, updates):",
                "    arr = np.asarray(data).copy()",
                "    idx = np.asarray(indices, dtype=np.int64)",
                "    if idx.ndim == 0:",
                "        raise ValueError('ScatterND indices must have rank >= 1.')",
                "    if idx.shape[-1] == 0:",
                "        raise ValueError('ScatterND indices must include at least one index dimension.')",
                "    flat_idx = idx.reshape(-1, idx.shape[-1])",
                "    slice_shape = tuple(arr.shape[int(idx.shape[-1]) :])",
                "    flat_updates = np.asarray(updates).reshape((flat_idx.shape[0],) + slice_shape)",
                "    if slice_shape:",
                "        for row_idx, row in enumerate(flat_idx):",
                "            arr[tuple(row.tolist())] = flat_updates[row_idx]",
                "    else:",
                "        arr[tuple(flat_idx.T)] = flat_updates.reshape(-1)",
                "    return mx.array(arr)",
                "",
            ]
        )

    if "TRILU" in ops:
        lines.extend(
            [
                "def _onnx_trilu(data, diagonal=None, *, upper=True):",
                "    diag = 0 if diagonal is None else int(np.asarray(diagonal).reshape(-1)[0])",
                "    arr = np.asarray(data)",
                "    if upper:",
                "        return mx.array(np.triu(arr, k=diag))",
                "    return mx.array(np.tril(arr, k=diag))",
                "",
            ]
        )

    if "UPSAMPLE" in ops:
        lines.extend(
            [
                "def _resolve_resize_target(data, scales=None, sizes=None):",
                "    arr = np.asarray(data)",
                "    if sizes is not None:",
                "        candidate = np.asarray(sizes).reshape(-1)",
                "        if candidate.size == arr.ndim and np.all(candidate > 0) and np.all(np.equal(candidate, np.round(candidate))):",
                "            return tuple(int(v) for v in candidate.tolist())",
                "    if scales is not None:",
                "        candidate = np.asarray(scales).reshape(-1)",
                "        if candidate.size == arr.ndim:",
                "            return tuple(",
                "                max(1, int(round(dim * float(scale))))",
                "                for dim, scale in zip(arr.shape, candidate.tolist(), strict=False)",
                "            )",
                "    raise ValueError('Resize helper needs scales or sizes with rank-matched values.')",
                "",
                "def _onnx_resize_nearest(data, roi=None, scales=None, sizes=None):",
                "    del roi",
                "    if scales is None and sizes is None:",
                "        return data",
                "    arr = np.asarray(data)",
                "    target_shape = _resolve_resize_target(arr, scales=scales, sizes=sizes)",
                "    out = arr",
                "    for axis, out_dim in enumerate(target_shape):",
                "        in_dim = out.shape[axis]",
                "        if out_dim == in_dim:",
                "            continue",
                "        scale = float(in_dim) / float(out_dim)",
                "        idx = np.floor(np.arange(out_dim) * scale).astype(np.int64)",
                "        idx = np.clip(idx, 0, max(in_dim - 1, 0))",
                "        out = np.take(out, idx, axis=axis)",
                "    return mx.array(out)",
                "",
            ]
        )

    if "CLIP" in ops:
        lines.extend(
            [
                "def _onnx_clip(data, min_val=None, max_val=None):",
                "    a_min = None if min_val is None else float(np.asarray(min_val).reshape(-1)[0])",
                "    a_max = None if max_val is None else float(np.asarray(max_val).reshape(-1)[0])",
                "    return mx.array(np.clip(np.asarray(data), a_min=a_min, a_max=a_max))",
                "",
            ]
        )

    if "MOD" in ops:
        lines.extend(
            [
                "def _onnx_mod(left, right, *, fmod=False):",
                "    if fmod:",
                "        return mx.array(np.fmod(np.asarray(left), np.asarray(right)))",
                "    return mx.remainder(left, right)",
                "",
            ]
        )

    if "ARANGE" in ops:
        lines.extend(
            [
                "def _onnx_arange(start, limit, delta):",
                "    start_v = np.asarray(start).reshape(-1)[0]",
                "    limit_v = np.asarray(limit).reshape(-1)[0]",
                "    delta_v = np.asarray(delta).reshape(-1)[0]",
                "    return mx.array(np.arange(start_v, limit_v, delta_v))",
                "",
            ]
        )

    if ops & {"QUANTIZE", "DEQUANTIZE"}:
        lines.extend(
            [
                "def _onnx_quantize(data, scale, zero_point=None):",
                "    arr = np.asarray(data)",
                "    scale_arr = np.asarray(scale)",
                "    if zero_point is None:",
                "        zp = 0.0",
                "        dtype = np.uint8",
                "    else:",
                "        zp_arr = np.asarray(zero_point)",
                "        zp = zp_arr",
                "        dtype = zp_arr.dtype",
                "    out = np.round(arr / scale_arr) + zp",
                "    return mx.array(out.astype(dtype))",
                "",
                "def _onnx_dequantize(data, scale, zero_point=None):",
                "    arr = np.asarray(data, dtype=np.float32)",
                "    scale_arr = np.asarray(scale, dtype=np.float32)",
                "    if zero_point is None:",
                "        zp = 0.0",
                "    else:",
                "        zp = np.asarray(zero_point, dtype=np.float32)",
                "    return mx.array((arr - zp) * scale_arr)",
                "",
            ]
        )

    if _requires_conv1d_helper(ir):
        lines.extend(
            [
                (
                    "def _onnx_conv1d("
                    "data, weight, bias=None, *, stride=1, padding=(0, 0), dilation=1, "
                    "groups=1, weight_prepacked=False"
                    "):"
                ),
                "    pad_l = int(padding[0])",
                "    pad_r = int(padding[1])",
                "    if pad_l != pad_r:",
                "        raise ValueError(",
                '            "MLX CONV1D helper supports symmetric ONNX pads only in v0."',
                "        )",
                "    x = mx.transpose(data, axes=(0, 2, 1))",
                "    w = weight if weight_prepacked else mx.transpose(weight, axes=(0, 2, 1))",
                (
                    "    out = mx.conv1d("
                    "x, w, stride=int(stride), padding=pad_l, dilation=int(dilation), "
                    "groups=int(groups)"
                    ")"
                ),
                "    out = mx.transpose(out, axes=(0, 2, 1))",
                "    if bias is not None:",
                "        out = out + bias.reshape(1, -1, 1)",
                "    return out",
                "",
            ]
        )

    if _requires_conv2d_helper(ir):
        lines.extend(
            [
                (
                    "def _onnx_conv2d("
                    "data, weight, bias=None, *, stride=(1, 1), padding=(0, 0), "
                    "dilation=(1, 1), groups=1, weight_prepacked=False"
                    "):"
                ),
                "    x = mx.transpose(data, axes=(0, 2, 3, 1))",
                "    w = weight if weight_prepacked else mx.transpose(weight, axes=(0, 2, 3, 1))",
                "    out = mx.conv2d(",
                "        x,",
                "        w,",
                "        stride=(int(stride[0]), int(stride[1])),",
                "        padding=(int(padding[0]), int(padding[1])),",
                "        dilation=(int(dilation[0]), int(dilation[1])),",
                "        groups=int(groups),",
                "    )",
                "    out = mx.transpose(out, axes=(0, 3, 1, 2))",
                "    if bias is not None:",
                "        out = out + bias.reshape(1, -1, 1, 1)",
                "    return out",
                "",
            ]
        )

    prepacked_weights = _mlx_prepacked_conv_weights(ir)
    lines.extend(
        [
            "def load_weights(path: str) -> dict[str, mx.array]:",
            "    data = np.load(path)",
            "    params = {k: mx.array(data[k]) for k in sorted(data.files)}",
        ]
    )
    for source, packed, rank in prepacked_weights:
        axes = "(0, 2, 1)" if rank == 3 else "(0, 2, 3, 1)"
        lines.append(f'    params["{packed}"] = mx.transpose(params["{source}"], axes={axes})')
    if prepacked_weights:
        lines.extend(
            [
                "    mx.eval(*params.values())",
                "    mx.synchronize()",
            ]
        )
    lines.extend(
        [
            "    return params",
            "",
            "def forward(",
            "    params: dict[str, mx.array],",
            "    inputs: dict[str, mx.array],",
            ") -> dict[str, mx.array]:",
            "    tensors: dict[str, mx.array] = {}",
        ]
    )
    for input_name in ir.inputs:
        lines.append(f'    tensors["{input_name}"] = mx.asarray(inputs["{input_name}"])')
    for node in order_nodes_for_emission(ir):
        lines.extend(_emit_node_lines(node, ir))
    outputs_literal = ", ".join([f'"{o}": tensors["{o}"]' for o in ir.outputs])
    lines.append(f"    return {{{outputs_literal}}}")
    if entrypoint != "forward":
        lines.append(f"{entrypoint} = forward")
    lines.append("")
    return "\n".join(lines)


def emit_mlx_module(ir: GraphIR, out_dir: str | Path, *, entrypoint: str = "forward") -> Path:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "model_mlx.py"
    out_path.write_text(render_mlx_module(ir, entrypoint=entrypoint), encoding="utf-8")
    return out_path
