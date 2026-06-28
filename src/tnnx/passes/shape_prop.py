from __future__ import annotations

from copy import deepcopy
from math import ceil, floor
from typing import Any

from ..ir.types import Dim, GraphIR, OpNode

type RuntimeScalar = int | float

type RuntimeValues = dict[str, list[RuntimeScalar]]

_SAME_SHAPE_OPS = {
    "BATCHNORM",
    "CAST",
    "CLIP",
    "COS",
    "CUMSUM",
    "ERF",
    "EXP",
    "GELU",
    "IDENTITY",
    "ISNAN",
    "INSTANCENORM",
    "LAYERNORM",
    "LOG",
    "MISH",
    "NEG",
    "RECIPROCAL",
    "RELU",
    "RELU6",
    "RMSNORM",
    "ROTARYEMBEDDING",
    "SCATTERND",
    "SIGMOID",
    "SILU",
    "SIN",
    "SKIPRMSNORM",
    "SOFTMAX",
    "SOFTPLUS",
    "SQRT",
    "TANH",
    "TRILU",
}

_BROADCAST_OPS = {
    "ADD",
    "AND",
    "DEQUANTIZE",
    "DIV",
    "EQUAL",
    "GREATER",
    "GREATEROREQUAL",
    "LESS",
    "LESSOREQUAL",
    "MOD",
    "MUL",
    "POW",
    "QUANTIZE",
    "SUB",
    "WHERE",
}


def _shape(ir: GraphIR, name: str) -> list[Dim]:
    tensor = ir.tensors.get(name)
    return list(tensor.shape) if tensor else []


def _set_shape(ir: GraphIR, name: str, shape: list[Dim]) -> None:
    if name in ir.tensors and shape:
        ir.tensors[name].shape = list(shape)


def _as_int(value: Dim) -> int | None:
    return value if isinstance(value, int) else None


def _attr_int(attrs: dict[str, Any], key: str, default: int) -> int:
    value = attrs.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        return int(value)
    return default


def _attr_float(attrs: dict[str, Any], key: str, default: float) -> float:
    value = attrs.get(key, default)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float | str):
        return float(value)
    return default


def _attr_int_list(attrs: dict[str, Any], key: str, default: list[int]) -> list[int]:
    value = attrs.get(key, default)
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            if isinstance(item, bool):
                out.append(int(item))
            elif isinstance(item, int | float | str):
                out.append(int(item))
        if out:
            return out
    return default


def _runtime_values(values: RuntimeValues, key: str) -> list[RuntimeScalar] | None:
    return values.get(key)


def _runtime_scalar(values: RuntimeValues, key: str) -> RuntimeScalar | None:
    raw = values.get(key)
    if raw:
        return raw[0]
    return None


def _normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def _prod_dim(dims: list[Dim], fallback: str) -> Dim:
    if not dims:
        return 1
    if all(isinstance(dim, int) for dim in dims):
        total = 1
        for dim in dims:
            assert isinstance(dim, int)
            total *= dim
        return total
    return fallback


def _merge_broadcast_dim(dim_a: Dim, dim_b: Dim) -> Dim | None:
    if dim_a == 1:
        return dim_b
    if dim_b == 1:
        return dim_a
    if dim_a == dim_b:
        return dim_a
    if isinstance(dim_a, str) and isinstance(dim_b, str):
        return dim_a if dim_a == dim_b else None
    if isinstance(dim_a, str):
        return dim_a
    if isinstance(dim_b, str):
        return dim_b
    return None


def _infer_broadcast_shape(shapes: list[list[Dim]]) -> list[Dim]:
    if not shapes:
        return []
    if len(shapes) == 1:
        return list(shapes[0])

    max_rank = max(len(shape) for shape in shapes)
    out: list[Dim] = []
    for offset in range(1, max_rank + 1):
        merged: Dim = 1
        for shape in shapes:
            dim = shape[-offset] if offset <= len(shape) else 1
            next_dim = _merge_broadcast_dim(merged, dim)
            if next_dim is None:
                return []
            merged = next_dim
        out.append(merged)
    out.reverse()
    return out


def _infer_matmul_shape(left: list[Dim], right: list[Dim]) -> list[Dim]:
    if len(left) == 2 and len(right) == 2:
        return [left[0], right[1]]
    if left and right:
        return [*left[:-1], right[-1]]
    return []


def _infer_gemm_shape(node: OpNode, left: list[Dim], right: list[Dim]) -> list[Dim]:
    if len(left) != 2 or len(right) != 2:
        return _infer_matmul_shape(left, right)

    left_m = left[1] if _attr_int(node.attrs, "transA", 0) == 1 else left[0]
    left_k = left[0] if _attr_int(node.attrs, "transA", 0) == 1 else left[1]
    right_k = right[1] if _attr_int(node.attrs, "transB", 0) == 1 else right[0]
    right_n = right[0] if _attr_int(node.attrs, "transB", 0) == 1 else right[1]

    if isinstance(left_k, int) and isinstance(right_k, int) and left_k != right_k:
        return []
    return [left_m, right_n]


def _infer_transpose_shape(node: OpNode, input_shape: list[Dim]) -> list[Dim]:
    perm = _attr_int_list(node.attrs, "perm", [])
    if not perm:
        return list(reversed(input_shape))
    if len(perm) != len(input_shape):
        return []
    return [input_shape[i] for i in perm]


def _infer_conv_shape(node: OpNode, input_shape: list[Dim], weight_shape: list[Dim]) -> list[Dim]:
    if len(input_shape) == 3 and len(weight_shape) == 3:
        n, _, length = input_shape
        oc, _, kernel = weight_shape
        strides = _attr_int_list(node.attrs, "strides", [1])
        pads = _attr_int_list(node.attrs, "pads", [0, 0])
        dilations = _attr_int_list(node.attrs, "dilations", [1])
        stride = strides[0] if strides else 1
        dilation = dilations[0] if dilations else 1
        if len(pads) >= 2:
            pad_l, pad_r = pads[0], pads[1]
        elif len(pads) == 1:
            pad_l = pads[0]
            pad_r = pads[0]
        else:
            pad_l = 0
            pad_r = 0

        length_i = _as_int(length)
        kernel_i = _as_int(kernel)
        if any(v is None for v in [length_i, kernel_i]):
            return [n, oc, "L_out"]
        assert length_i is not None and kernel_i is not None
        length_out = floor((length_i + pad_l + pad_r - dilation * (kernel_i - 1) - 1) / stride + 1)
        return [n, oc, length_out]

    if len(input_shape) == 4 and len(weight_shape) == 4:
        n, _, h, w = input_shape
        oc, _, kh, kw = weight_shape
        strides = _attr_int_list(node.attrs, "strides", [1, 1])
        pads = _attr_int_list(node.attrs, "pads", [0, 0, 0, 0])
        dilations = _attr_int_list(node.attrs, "dilations", [1, 1])

        h_i = _as_int(h)
        w_i = _as_int(w)
        kh_i = _as_int(kh)
        kw_i = _as_int(kw)
        if any(v is None for v in [h_i, w_i, kh_i, kw_i]):
            return [n, oc, "H_out", "W_out"]

        assert h_i is not None and w_i is not None and kh_i is not None and kw_i is not None
        h_out = floor((h_i + pads[0] + pads[2] - dilations[0] * (kh_i - 1) - 1) / strides[0] + 1)
        w_out = floor((w_i + pads[1] + pads[3] - dilations[1] * (kw_i - 1) - 1) / strides[1] + 1)
        return [n, oc, h_out, w_out]

    return []


def _infer_pool_shape(node: OpNode, input_shape: list[Dim]) -> list[Dim]:
    if _attr_int(node.attrs, "global", 0) == 1:
        if len(input_shape) >= 3:
            return [*input_shape[:2], *([1] * (len(input_shape) - 2))]
        return list(input_shape)

    if len(input_shape) == 3:
        n, c, length = input_shape
        kernel = _attr_int_list(node.attrs, "kernel_shape", [1])
        strides = _attr_int_list(node.attrs, "strides", [1])
        pads = _attr_int_list(node.attrs, "pads", [0, 0])
        dilations = _attr_int_list(node.attrs, "dilations", [1])
        stride = strides[0] if strides else 1
        dilation = dilations[0] if dilations else 1
        kernel_i = kernel[0] if kernel else 1
        if len(pads) >= 2:
            pad_l, pad_r = pads[0], pads[1]
        elif len(pads) == 1:
            pad_l = pads[0]
            pad_r = pads[0]
        else:
            pad_l = 0
            pad_r = 0
        length_i = _as_int(length)
        if length_i is None:
            return [n, c, "L_out"]
        length_out = floor((length_i + pad_l + pad_r - dilation * (kernel_i - 1) - 1) / stride + 1)
        return [n, c, length_out]

    if len(input_shape) == 4:
        n, c, h, w = input_shape
        kernel = _attr_int_list(node.attrs, "kernel_shape", [1, 1])
        strides = _attr_int_list(node.attrs, "strides", [1, 1])
        pads = _attr_int_list(node.attrs, "pads", [0, 0, 0, 0])
        dilations = _attr_int_list(node.attrs, "dilations", [1, 1])
        kh = kernel[0] if kernel else 1
        kw = kernel[1] if len(kernel) > 1 else kh
        sh = strides[0] if strides else 1
        sw = strides[1] if len(strides) > 1 else sh
        dh = dilations[0] if dilations else 1
        dw = dilations[1] if len(dilations) > 1 else dh
        pads4 = pads if len(pads) >= 4 else [0, 0, 0, 0]
        h_i = _as_int(h)
        w_i = _as_int(w)
        if h_i is None or w_i is None:
            return [n, c, "H_out", "W_out"]
        h_out = floor((h_i + pads4[0] + pads4[2] - dh * (kh - 1) - 1) / sh + 1)
        w_out = floor((w_i + pads4[1] + pads4[3] - dw * (kw - 1) - 1) / sw + 1)
        return [n, c, h_out, w_out]

    return []


def _infer_gather_shape(node: OpNode, data_shape: list[Dim], indices_shape: list[Dim]) -> list[Dim]:
    if not data_shape:
        return []
    rank = len(data_shape)
    axis = _normalize_axis(_attr_int(node.attrs, "axis", 0), rank)
    if axis < 0 or axis >= rank:
        return []
    return [*data_shape[:axis], *indices_shape, *data_shape[axis + 1 :]]


def _infer_gather_elements_shape(indices_shape: list[Dim]) -> list[Dim]:
    return list(indices_shape)


def _infer_concat_shape(node: OpNode, input_shapes: list[list[Dim]]) -> list[Dim]:
    if not input_shapes:
        return []
    rank = len(input_shapes[0])
    if any(len(shape) != rank for shape in input_shapes):
        return []
    axis = _normalize_axis(_attr_int(node.attrs, "axis", 0), rank)
    if axis < 0 or axis >= rank:
        return []

    out = list(input_shapes[0])
    for dim_idx in range(rank):
        dims = [shape[dim_idx] for shape in input_shapes]
        if dim_idx == axis:
            if all(isinstance(dim, int) for dim in dims):
                out[dim_idx] = sum(int(dim) for dim in dims)
            else:
                out[dim_idx] = f"concat_{axis}"
            continue
        first = dims[0]
        if any(dim != first for dim in dims[1:]):
            return []
        out[dim_idx] = first
    return out


def _infer_unsqueeze_shape(
    node: OpNode,
    input_shape: list[Dim],
    runtime_values: RuntimeValues,
) -> list[Dim]:
    axes_raw: list[int] = []
    if isinstance(node.attrs.get("axes"), list):
        axes_raw = _attr_int_list(node.attrs, "axes", [])
    elif len(node.inputs) > 1:
        axes_values = _runtime_values(runtime_values, node.inputs[1])
        if axes_values is not None:
            axes_raw = [int(v) for v in axes_values]
    if not axes_raw:
        return []

    out_rank = len(input_shape) + len(axes_raw)
    axes = sorted(_normalize_axis(axis, out_rank) for axis in axes_raw)
    if any(axis < 0 or axis >= out_rank for axis in axes):
        return []
    if len(set(axes)) != len(axes):
        return []

    out: list[Dim] = []
    input_idx = 0
    axis_set = set(axes)
    for dim_idx in range(out_rank):
        if dim_idx in axis_set:
            out.append(1)
        else:
            out.append(input_shape[input_idx])
            input_idx += 1
    return out


def _infer_squeeze_shape(
    node: OpNode,
    input_shape: list[Dim],
    runtime_values: RuntimeValues,
) -> list[Dim]:
    if not input_shape:
        return []

    axes = node.attrs.get("axes")
    if isinstance(axes, list):
        axes_l = [int(v) for v in axes]
    elif len(node.inputs) > 1:
        runtime_axes = runtime_values.get(node.inputs[1])
        axes_l = [int(v) for v in runtime_axes] if runtime_axes is not None else []
    else:
        axes_l = []

    if not axes_l:
        return [dim for dim in input_shape if dim != 1]

    rank = len(input_shape)
    normalized = {_normalize_axis(int(axis), rank) for axis in axes_l}
    if any(axis < 0 or axis >= rank for axis in normalized):
        return []

    out: list[Dim] = []
    for idx, dim in enumerate(input_shape):
        if idx not in normalized:
            out.append(dim)
            continue
        if dim == 1 or not isinstance(dim, int):
            continue
        return []
    return out


def _infer_flatten_shape(node: OpNode, input_shape: list[Dim]) -> list[Dim]:
    if not input_shape:
        return [1, 1]
    rank = len(input_shape)
    axis = _normalize_axis(_attr_int(node.attrs, "axis", 1), rank)
    if axis < 0 or axis > rank:
        return []
    return [
        _prod_dim(input_shape[:axis], "flatten_left"),
        _prod_dim(input_shape[axis:], "flatten_right"),
    ]


def _infer_slice_shape(
    data_shape: list[Dim],
    node: OpNode,
    runtime_values: RuntimeValues,
) -> list[Dim]:
    if len(node.inputs) < 3:
        return data_shape
    starts = _runtime_values(runtime_values, node.inputs[1])
    ends = _runtime_values(runtime_values, node.inputs[2])
    if starts is None or ends is None:
        return data_shape

    if len(node.inputs) > 3:
        axes_values = _runtime_values(runtime_values, node.inputs[3])
        axes = (
            [int(v) for v in axes_values] if axes_values is not None else list(range(len(starts)))
        )
    else:
        axes = list(range(len(starts)))

    if len(node.inputs) > 4:
        steps_values = _runtime_values(runtime_values, node.inputs[4])
        steps = [int(v) for v in steps_values] if steps_values is not None else [1] * len(starts)
    else:
        steps = [1] * len(starts)

    out = list(data_shape)
    rank = len(out)
    for start, end, axis, step in zip(starts, ends, axes, steps, strict=False):
        if step == 0:
            continue
        axis_idx = axis if axis >= 0 else rank + axis
        if axis_idx < 0 or axis_idx >= rank:
            continue
        dim = out[axis_idx]
        if not isinstance(dim, int):
            continue
        start_i = int(start)
        end_i = int(end)
        step_i = int(step)
        start_idx = start_i if start_i >= 0 else dim + start_i
        end_idx = end_i if end_i >= 0 else dim + end_i
        start_idx = max(0, min(dim, start_idx))
        end_idx = max(0, min(dim, end_idx))
        if step_i > 0:
            if end_idx <= start_idx:
                out_len = 0
            else:
                out_len = ((end_idx - start_idx - 1) // step_i) + 1
        else:
            if start_idx <= end_idx:
                out_len = 0
            else:
                out_len = ((start_idx - end_idx - 1) // (-step_i)) + 1
        out[axis_idx] = out_len
    return out


def _infer_range_shape(node: OpNode, runtime_values: RuntimeValues) -> list[Dim]:
    if len(node.inputs) < 3:
        return []
    start = _runtime_scalar(runtime_values, node.inputs[0])
    limit = _runtime_scalar(runtime_values, node.inputs[1])
    delta = _runtime_scalar(runtime_values, node.inputs[2])
    if start is None or limit is None or delta is None:
        return ["range"]
    delta_f = float(delta)
    if delta_f == 0.0:
        return []
    steps = (float(limit) - float(start)) / delta_f
    count = max(0, int(ceil(steps)))
    return [count]


def _infer_resize_shape(
    node: OpNode,
    input_shape: list[Dim],
    runtime_values: RuntimeValues,
) -> list[Dim]:
    if not input_shape or len(node.inputs) < 2:
        return []

    slot_values: dict[int, list[RuntimeScalar]] = {}
    slots = _attr_int_list(node.attrs, "input_slots", list(range(len(node.inputs))))
    for slot, name in zip(slots, node.inputs, strict=False):
        values = _runtime_values(runtime_values, name)
        if values is not None:
            slot_values[slot] = values

    sizes = slot_values.get(3)
    if sizes is not None and len(sizes) == len(input_shape):
        if all(float(v).is_integer() and int(v) > 0 for v in sizes):
            return [int(v) for v in sizes]

    scales = slot_values.get(2)
    if scales is None and 1 in slot_values and 2 not in slots and 3 not in slots:
        scales = slot_values[1]
    if scales is not None and len(scales) == len(input_shape):
        out: list[Dim] = []
        for dim, scale in zip(input_shape, scales, strict=False):
            if isinstance(dim, int):
                out.append(max(1, int(round(dim * float(scale)))))
            else:
                out.append(dim)
        return out

    mode = str(node.attrs.get("mode", "nearest")).lower()
    if mode != "nearest":
        return []
    return list(input_shape)


def _infer_constant_of_shape_shape(node: OpNode, runtime_values: RuntimeValues) -> list[Dim]:
    if not node.inputs:
        return []
    shape_values = _runtime_values(runtime_values, node.inputs[0])
    if shape_values is None:
        return []
    return [int(v) for v in shape_values]


def _infer_expand_shape(
    node: OpNode,
    input_shape: list[Dim],
    runtime_values: RuntimeValues,
) -> list[Dim]:
    if len(node.inputs) < 2:
        return []
    shape_values = _runtime_values(runtime_values, node.inputs[1])
    if shape_values is None:
        return []
    target = [int(v) for v in shape_values]
    merged = _infer_broadcast_shape([input_shape, target])
    return merged or target


def _infer_shape_op_shape(node: OpNode, input_shape: list[Dim]) -> list[Dim]:
    rank = len(input_shape)
    start = _attr_int(node.attrs, "start", 0)
    end = _attr_int(node.attrs, "end", rank)
    if start < 0:
        start += rank
    if end < 0:
        end += rank
    start = max(0, min(rank, start))
    end = max(start, min(rank, end))
    return [end - start]


def _infer_pad_shape(
    node: OpNode,
    input_shape: list[Dim],
    runtime_values: RuntimeValues,
) -> list[Dim]:
    if len(node.inputs) < 2:
        return list(input_shape)

    pads_values = _runtime_values(runtime_values, node.inputs[1])
    if pads_values is None or len(pads_values) % 2 != 0:
        return list(input_shape)

    pad_rank = len(pads_values) // 2
    if len(node.inputs) > 3:
        axes_values = _runtime_values(runtime_values, node.inputs[3])
        if axes_values is None:
            return list(input_shape)
        axes = [int(v) for v in axes_values]
    else:
        axes = list(range(pad_rank))

    if len(axes) != pad_rank:
        return list(input_shape)

    out = list(input_shape)
    rank = len(out)
    for idx, axis in enumerate(axes):
        axis_idx = _normalize_axis(axis, rank)
        if axis_idx < 0 or axis_idx >= rank:
            return list(input_shape)
        dim = out[axis_idx]
        if isinstance(dim, int):
            out[axis_idx] = dim + int(pads_values[idx]) + int(pads_values[idx + pad_rank])
    return out


def _infer_reduce_shape(
    node: OpNode,
    input_shape: list[Dim],
    runtime_values: RuntimeValues,
) -> list[Dim]:
    rank = len(input_shape)
    if rank == 0:
        return [1] if _attr_int(node.attrs, "keepdims", 1) == 1 else []

    if isinstance(node.attrs.get("axes"), list):
        axes_raw = _attr_int_list(node.attrs, "axes", [])
    elif len(node.inputs) > 1:
        axes_values = _runtime_values(runtime_values, node.inputs[1])
        axes_raw = [int(v) for v in axes_values] if axes_values is not None else []
    else:
        axes_raw = []

    if axes_raw:
        axes = sorted({_normalize_axis(axis, rank) for axis in axes_raw})
        if any(axis < 0 or axis >= rank for axis in axes):
            return []
    else:
        axes = list(range(rank))

    keepdims = _attr_int(node.attrs, "keepdims", 1) == 1
    if keepdims:
        out = list(input_shape)
        for axis in axes:
            out[axis] = 1
        return out
    reduced = set(axes)
    return [dim for idx, dim in enumerate(input_shape) if idx not in reduced]


def _resolve_split_sizes(node: OpNode, runtime_values: RuntimeValues) -> list[int] | None:
    if len(node.inputs) > 1:
        split_values = _runtime_values(runtime_values, node.inputs[1])
        if split_values:
            return [int(v) for v in split_values]
    split_attr = node.attrs.get("split")
    if isinstance(split_attr, list):
        return [int(v) for v in split_attr if isinstance(v, int | float | str | bool)]
    return None


def _infer_split_shapes(
    node: OpNode,
    input_shape: list[Dim],
    runtime_values: RuntimeValues,
) -> list[list[Dim]]:
    if not input_shape or not node.outputs:
        return []
    axis = _normalize_axis(_attr_int(node.attrs, "axis", 0), len(input_shape))
    if axis < 0 or axis >= len(input_shape):
        return []

    axis_dim = input_shape[axis]
    split_sizes = _resolve_split_sizes(node, runtime_values)
    if split_sizes is None:
        axis_int = _as_int(axis_dim)
        output_count = len(node.outputs)
        if output_count <= 0:
            return []
        if axis_int is None:
            split_sizes = [1] * output_count
        else:
            if axis_int % output_count != 0:
                return []
            split_sizes = [axis_int // output_count] * output_count

    if len(split_sizes) != len(node.outputs):
        return []

    split_sizes_were_explicit = _resolve_split_sizes(node, runtime_values) is not None
    out_shapes: list[list[Dim]] = []
    for idx, size in enumerate(split_sizes):
        shape = list(input_shape)
        if isinstance(axis_dim, str) and not split_sizes_were_explicit:
            shape[axis] = f"{axis_dim}_split{idx}"
        else:
            shape[axis] = int(size)
        out_shapes.append(shape)
    return out_shapes


def propagate_shapes(
    ir: GraphIR,
    *,
    runtime_values: RuntimeValues | None = None,
) -> GraphIR:
    out = deepcopy(ir)
    values = runtime_values or {}

    for node in out.nodes:
        if not node.outputs:
            continue
        input_shapes = [_shape(out, name) for name in node.inputs]

        if node.op == "GROUPQUERYATTENTION" and input_shapes:
            _set_shape(out, node.outputs[0], input_shapes[0])
            continue

        if node.op == "SKIPRMSNORM" and input_shapes:
            for output_name in node.outputs:
                _set_shape(out, output_name, input_shapes[0])
            continue

        if node.op == "SPLIT" and input_shapes:
            split_shapes = _infer_split_shapes(node, input_shapes[0], values)
            if len(split_shapes) == len(node.outputs):
                for output_name, shape in zip(node.outputs, split_shapes, strict=False):
                    _set_shape(out, output_name, shape)
            continue

        output = node.outputs[0]

        if node.op in _SAME_SHAPE_OPS and input_shapes:
            _set_shape(out, output, input_shapes[0])
        elif node.op in _BROADCAST_OPS and input_shapes:
            _set_shape(out, output, _infer_broadcast_shape(input_shapes))
        elif node.op == "GATHER" and len(input_shapes) >= 2:
            _set_shape(out, output, _infer_gather_shape(node, input_shapes[0], input_shapes[1]))
        elif node.op == "GATHERELEMENTS" and len(input_shapes) >= 2:
            _set_shape(out, output, _infer_gather_elements_shape(input_shapes[1]))
        elif node.op == "CONCAT" and input_shapes:
            _set_shape(out, output, _infer_concat_shape(node, input_shapes))
        elif node.op == "UNSQUEEZE" and input_shapes:
            _set_shape(out, output, _infer_unsqueeze_shape(node, input_shapes[0], values))
        elif node.op == "SQUEEZE" and input_shapes:
            _set_shape(out, output, _infer_squeeze_shape(node, input_shapes[0], values))
        elif node.op == "FLATTEN" and input_shapes:
            _set_shape(out, output, _infer_flatten_shape(node, input_shapes[0]))
        elif node.op == "SLICE" and input_shapes:
            _set_shape(out, output, _infer_slice_shape(input_shapes[0], node, values))
        elif node.op == "MATMUL" and len(input_shapes) >= 2:
            _set_shape(out, output, _infer_matmul_shape(input_shapes[0], input_shapes[1]))
        elif node.op == "GEMM" and len(input_shapes) >= 2:
            _set_shape(out, output, _infer_gemm_shape(node, input_shapes[0], input_shapes[1]))
        elif node.op == "TRANSPOSE" and input_shapes:
            _set_shape(out, output, _infer_transpose_shape(node, input_shapes[0]))
        elif node.op == "RESHAPE" and len(node.inputs) >= 2:
            reshape_key = node.inputs[1]
            if reshape_key in values:
                _set_shape(out, output, [int(v) for v in values[reshape_key]])
        elif node.op == "CONV2D" and len(input_shapes) >= 2:
            _set_shape(out, output, _infer_conv_shape(node, input_shapes[0], input_shapes[1]))
        elif node.op in {"AVGPOOL", "MAXPOOL"} and input_shapes:
            _set_shape(out, output, _infer_pool_shape(node, input_shapes[0]))
        elif node.op == "ARANGE":
            _set_shape(out, output, _infer_range_shape(node, values))
        elif node.op == "UPSAMPLE" and input_shapes:
            _set_shape(out, output, _infer_resize_shape(node, input_shapes[0], values))
        elif node.op == "CONSTANT_OF_SHAPE":
            _set_shape(out, output, _infer_constant_of_shape_shape(node, values))
        elif node.op == "EXPAND":
            input_shape = input_shapes[0] if input_shapes else []
            _set_shape(out, output, _infer_expand_shape(node, input_shape, values))
        elif node.op == "SHAPE" and input_shapes:
            _set_shape(out, output, _infer_shape_op_shape(node, input_shapes[0]))
        elif node.op == "PAD" and input_shapes:
            _set_shape(out, output, _infer_pad_shape(node, input_shapes[0], values))
        elif node.op in {"REDUCEMEAN", "REDUCESUM"} and input_shapes:
            _set_shape(out, output, _infer_reduce_shape(node, input_shapes[0], values))

    return out
