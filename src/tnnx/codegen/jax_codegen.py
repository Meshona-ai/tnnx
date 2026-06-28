# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ir.types import GraphIR, OpNode


def _tensor_expr(name: str, ir: GraphIR) -> str:
    tensor = ir.tensors[name]
    if tensor.kind == "input":
        return f'tensors["{name}"]'
    if tensor.kind == "initializer":
        return f'params["{name}"]'
    return f'tensors["{name}"]'


def _attr_int(attrs: dict[str, Any], key: str, default: int) -> int:
    value = attrs.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        return int(value)
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


def _attr_float(attrs: dict[str, Any], key: str, default: float) -> float:
    value = attrs.get(key, default)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float | str):
        return float(value)
    return default


def _ops_in_graph(ir: GraphIR) -> set[str]:
    return {node.op for node in ir.nodes}


def _input_rank(name: str, ir: GraphIR) -> int:
    tensor = ir.tensors.get(name)
    return len(tensor.shape) if tensor is not None else 0


def _requires_conv1d_helper(ir: GraphIR) -> bool:
    return any(
        node.op == "CONV2D" and node.inputs and _input_rank(node.inputs[0], ir) == 3
        for node in ir.nodes
    )


def _requires_conv2d_helper(ir: GraphIR) -> bool:
    return any(
        node.op == "CONV2D" and node.inputs and _input_rank(node.inputs[0], ir) != 3
        for node in ir.nodes
    )


def _clip_arg(node: OpNode, ins: list[str], input_idx: int, attr_name: str) -> str:
    if len(ins) > input_idx:
        return ins[input_idx]
    if attr_name in node.attrs:
        return repr(node.attrs[attr_name])
    return "None"


def _slot_exprs(node: OpNode, ins: list[str]) -> dict[int, str]:
    slots = _attr_int_list(node.attrs, "input_slots", list(range(len(ins))))
    return {slot: expr for slot, expr in zip(slots, ins, strict=False)}


def _scheduled_nodes(ir: GraphIR) -> list[OpNode]:
    remaining = list(ir.nodes)
    available = set(ir.inputs)
    available.update(name for name, tensor in ir.tensors.items() if tensor.kind == "initializer")
    ordered: list[OpNode] = []

    while remaining:
        progressed = False
        deferred: list[OpNode] = []
        for node in remaining:
            if all(name in available for name in node.inputs):
                ordered.append(node)
                available.update(node.outputs)
                progressed = True
            else:
                deferred.append(node)
        if not progressed:
            ordered.extend(remaining)
            break
        remaining = deferred
    return ordered


_STATIC_META_INPUT_SLOTS: dict[str, tuple[int, ...]] = {
    "CONSTANT_OF_SHAPE": (0,),
    "CUMSUM": (1,),
    "EXPAND": (1,),
    "PAD": (1, 3),
    "REDUCEMEAN": (1,),
    "REDUCESUM": (1,),
    "RESHAPE": (1,),
    "SLICE": (1, 2, 3, 4),
    "SPLIT": (1,),
    "SQUEEZE": (1,),
    "TRILU": (1,),
    "UNSQUEEZE": (1,),
    "UPSAMPLE": (2, 3),
}


_STATIC_META_OPS = {
    "ADD",
    "AND",
    "CAST",
    "CONCAT",
    "CONSTANT_OF_SHAPE",
    "DIV",
    "EQUAL",
    "EXPAND",
    "GATHER",
    "GATHERELEMENTS",
    "GREATER",
    "GREATEROREQUAL",
    "LESS",
    "LESSOREQUAL",
    "MUL",
    "RESHAPE",
    "SHAPE",
    "SLICE",
    "SQUEEZE",
    "SUB",
    "UNSQUEEZE",
    "WHERE",
}


def _static_meta_analysis(ir: GraphIR) -> tuple[set[str], set[str]]:
    demanded: set[str] = set()
    for node in ir.nodes:
        for slot in _STATIC_META_INPUT_SLOTS.get(node.op, ()):
            if slot < len(node.inputs):
                demanded.add(node.inputs[slot])

    static_outputs: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in reversed(ir.nodes):
            if node.op not in _STATIC_META_OPS:
                continue
            if not any(output in demanded for output in node.outputs):
                continue
            for output in node.outputs:
                if output not in static_outputs:
                    static_outputs.add(output)
                    changed = True
            if node.op == "SHAPE":
                continue
            for input_name in node.inputs:
                if input_name not in demanded:
                    demanded.add(input_name)
                    changed = True
    static_params = {
        name
        for name in demanded
        if (tensor := ir.tensors.get(name)) is not None and tensor.kind == "initializer"
    }
    return static_outputs, static_params


def _emit_node_lines(node: OpNode, ir: GraphIR, static_meta: set[str]) -> list[str]:
    expr = _emit_node_expr(node, ir, static_meta)
    if len(node.outputs) == 1:
        return [f'    tensors["{node.outputs[0]}"] = {expr}']
    tmp_name = f"_node_{node.id}"
    lines = [f"    {tmp_name} = {expr}"]
    for idx, output_name in enumerate(node.outputs):
        lines.append(f'    tensors["{output_name}"] = {tmp_name}[{idx}]')
    return lines


def _emit_node_expr(node: OpNode, ir: GraphIR, static_meta: set[str]) -> str:
    ins = [_tensor_expr(name, ir) for name in node.inputs]
    if any(output in static_meta for output in node.outputs):
        static_expr = _emit_static_meta_expr(node, ins)
        if static_expr is not None:
            return static_expr
    if node.op == "ADD":
        return f"jnp.add({ins[0]}, {ins[1]})"
    if node.op == "AND":
        return f"jnp.logical_and({ins[0]}, {ins[1]})"
    if node.op == "SUB":
        return f"jnp.subtract({ins[0]}, {ins[1]})"
    if node.op == "DIV":
        return f"jnp.divide({ins[0]}, {ins[1]})"
    if node.op == "EQUAL":
        return f"jnp.equal({ins[0]}, {ins[1]})"
    if node.op == "GREATER":
        return f"jnp.greater({ins[0]}, {ins[1]})"
    if node.op == "GREATEROREQUAL":
        return f"jnp.greater_equal({ins[0]}, {ins[1]})"
    if node.op == "ERF":
        return f"jax.lax.erf({ins[0]})"
    if node.op == "EXP":
        return f"jnp.exp({ins[0]})"
    if node.op == "EXPAND":
        return f"_onnx_expand({ins[0]}, {ins[1]})"
    if node.op == "COS":
        return f"jnp.cos({ins[0]})"
    if node.op == "CUMSUM":
        exclusive = _attr_int(node.attrs, "exclusive", 0) == 1
        reverse = _attr_int(node.attrs, "reverse", 0) == 1
        return f"_onnx_cumsum({ins[0]}, {ins[1]}, exclusive={exclusive}, reverse={reverse})"
    if node.op == "SIN":
        return f"jnp.sin({ins[0]})"
    if node.op == "SIGMOID":
        return f"jnn.sigmoid({ins[0]})"
    if node.op == "SILU":
        return f"({ins[0]} * jnn.sigmoid({ins[0]}))"
    if node.op == "SOFTPLUS":
        return f"jnn.softplus({ins[0]})"
    if node.op == "SQRT":
        return f"jnp.sqrt({ins[0]})"
    if node.op == "TANH":
        return f"jnp.tanh({ins[0]})"
    if node.op == "MISH":
        return f"({ins[0]} * jnp.tanh(jnp.log1p(jnp.exp({ins[0]}))))"
    if node.op == "GATHER":
        axis = _attr_int(node.attrs, "axis", 0)
        return f"jnp.take({ins[0]}, jnp.asarray({ins[1]}, dtype=jnp.int32), axis={axis})"
    if node.op == "GATHERELEMENTS":
        axis = _attr_int(node.attrs, "axis", 0)
        return f"jnp.take_along_axis({ins[0]}, jnp.asarray({ins[1]}, dtype=jnp.int32), axis={axis})"
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
        return f"jnp.isnan({ins[0]})"
    if node.op == "LESS":
        return f"jnp.less({ins[0]}, {ins[1]})"
    if node.op == "LESSOREQUAL":
        return f"jnp.less_equal({ins[0]}, {ins[1]})"
    if node.op == "LOG":
        return f"jnp.log({ins[0]})"
    if node.op == "MUL":
        return f"jnp.multiply({ins[0]}, {ins[1]})"
    if node.op == "MOD":
        fmod = _attr_int(node.attrs, "fmod", 0) == 1
        return f"_onnx_mod({ins[0]}, {ins[1]}, fmod={fmod})"
    if node.op == "NEG":
        return f"jnp.negative({ins[0]})"
    if node.op == "PAD":
        slot_map = _slot_exprs(node, ins)
        constant_value = slot_map.get(2, "None")
        axes = slot_map.get(3, "None")
        mode = repr(str(node.attrs.get("mode", "constant")).lower())
        return f"_onnx_pad({ins[0]}, {ins[1]}, {constant_value}, {axes}, mode={mode})"
    if node.op == "POW":
        return f"jnp.power({ins[0]}, {ins[1]})"
    if node.op == "RECIPROCAL":
        return f"jnp.reciprocal({ins[0]})"
    if node.op == "MATMUL":
        return f"jnp.matmul({ins[0]}, {ins[1]})"
    if node.op == "GEMM":
        left = ins[0]
        right = ins[1]
        if _attr_int(node.attrs, "transA", 0) == 1:
            left = f"jnp.swapaxes({left}, -1, -2)"
        if _attr_int(node.attrs, "transB", 0) == 1:
            right = f"jnp.swapaxes({right}, -1, -2)"
        out = f"jnp.matmul({left}, {right})"
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
        return f"jnp.maximum({ins[0]}, 0)"
    if node.op == "RELU6":
        return f"jnp.clip({ins[0]}, a_min=0, a_max=6)"
    if node.op == "CLIP":
        return f"_onnx_clip({ins[0]}, {_clip_arg(node, ins, 1, 'min')}, {_clip_arg(node, ins, 2, 'max')})"
    if node.op == "GELU":
        return f"jnn.gelu({ins[0]})"
    if node.op == "LAYERNORM":
        axis = _attr_int(node.attrs, "axis", -1)
        eps = _attr_float(node.attrs, "epsilon", 1e-5)
        mean = f"jnp.mean({ins[0]}, axis={axis}, keepdims=True)"
        var = f"jnp.var({ins[0]}, axis={axis}, keepdims=True)"
        out = f"(({ins[0]} - {mean}) / jnp.sqrt({var} + {eps})) * {ins[1]}"
        if len(ins) > 2:
            out = f"({out} + {ins[2]})"
        return out
    if node.op == "RMSNORM":
        axis = _attr_int(node.attrs, "axis", -1)
        eps = _attr_float(node.attrs, "epsilon", 1e-5)
        rms = f"jnp.mean(jnp.square({ins[0]}), axis={axis}, keepdims=True)"
        out = f"({ins[0]} / jnp.sqrt({rms} + {eps})) * {ins[1]}"
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
        return f"jnp.concatenate(({joined}), axis={_attr_int(node.attrs, 'axis', 0)})"
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
            return f"jnp.transpose({ins[0]})"
        return f"jnp.transpose({ins[0]}, axes={tuple(_attr_int_list(node.attrs, 'perm', []))})"
    if node.op == "TRILU":
        diagonal = ins[1] if len(ins) > 1 else "None"
        upper = _attr_int(node.attrs, "upper", 1) == 1
        return f"_onnx_trilu({ins[0]}, {diagonal}, upper={upper})"
    if node.op == "SOFTMAX":
        axis = _attr_int(node.attrs, "axis", -1)
        return f"jnn.softmax({ins[0]}, axis={axis})"
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
        return f"jnp.where(jnp.asarray({ins[0]}, dtype=bool), {ins[1]}, {ins[2]})"
    if node.op == "QUANTIZE":
        zero_point = ins[2] if len(ins) > 2 else "None"
        return f"_onnx_quantize({ins[0]}, {ins[1]}, {zero_point})"
    if node.op == "DEQUANTIZE":
        zero_point = ins[2] if len(ins) > 2 else "None"
        return f"_onnx_dequantize({ins[0]}, {ins[1]}, {zero_point})"
    if node.op == "CONV2D":
        rank = _input_rank(node.inputs[0], ir) if node.inputs else 0
        groups = _attr_int(node.attrs, "group", 1)
        if rank == 3:
            strides_l = _attr_int_list(node.attrs, "strides", [1])
            dilations_l = _attr_int_list(node.attrs, "dilations", [1])
            pads_l = _attr_int_list(node.attrs, "pads", [0, 0])
            stride = strides_l[0] if strides_l else 1
            dilation = dilations_l[0] if dilations_l else 1
            if len(pads_l) >= 2:
                pad_l, pad_r = pads_l[0], pads_l[1]
            elif len(pads_l) == 1:
                pad_l = pads_l[0]
                pad_r = pads_l[0]
            else:
                pad_l = 0
                pad_r = 0
            conv = (
                "jax.lax.conv_general_dilated("
                f"{ins[0]}, {ins[1]}, window_strides=({stride},), padding=(({pad_l}, {pad_r}),), "
                f"lhs_dilation={(1,)}, rhs_dilation=({dilation},), "
                'dimension_numbers=("NCW","OIW","NCW"), '
                f"feature_group_count={groups})"
            )
            if len(ins) > 2:
                return f"({conv} + {ins[2]}.reshape(1, -1, 1))"
            return conv

        strides_2d = tuple(_attr_int_list(node.attrs, "strides", [1, 1]))
        pads_2d = _attr_int_list(node.attrs, "pads", [0, 0, 0, 0])
        dilations_2d = tuple(_attr_int_list(node.attrs, "dilations", [1, 1]))
        padding = ((pads_2d[0], pads_2d[2]), (pads_2d[1], pads_2d[3]))
        conv = (
            "jax.lax.conv_general_dilated("
            f"{ins[0]}, {ins[1]}, window_strides={strides_2d}, padding={padding}, "
            f"lhs_dilation={(1, 1)}, rhs_dilation={dilations_2d}, "
            'dimension_numbers=("NCHW","OIHW","NCHW"), '
            f"feature_group_count={groups})"
        )
        if len(ins) > 2:
            return f"({conv} + {ins[2]}.reshape(1, -1, 1, 1))"
        return conv
    raise ValueError(f"Unsupported op for JAX codegen: {node.op}")


def _emit_static_meta_expr(node: OpNode, ins: list[str]) -> str | None:
    if node.op == "ADD":
        return f"np.add({ins[0]}, {ins[1]})"
    if node.op == "AND":
        return f"np.logical_and({ins[0]}, {ins[1]})"
    if node.op == "SUB":
        return f"np.subtract({ins[0]}, {ins[1]})"
    if node.op == "MUL":
        return f"np.multiply({ins[0]}, {ins[1]})"
    if node.op == "DIV":
        return f"np.divide({ins[0]}, {ins[1]})"
    if node.op == "EQUAL":
        return f"np.equal({ins[0]}, {ins[1]})"
    if node.op == "GREATER":
        return f"np.greater({ins[0]}, {ins[1]})"
    if node.op == "GREATEROREQUAL":
        return f"np.greater_equal({ins[0]}, {ins[1]})"
    if node.op == "LESS":
        return f"np.less({ins[0]}, {ins[1]})"
    if node.op == "LESSOREQUAL":
        return f"np.less_equal({ins[0]}, {ins[1]})"
    if node.op == "WHERE":
        return f"np.where(np.asarray({ins[0]}, dtype=bool), {ins[1]}, {ins[2]})"
    if node.op == "CONCAT":
        axis = _attr_int(node.attrs, "axis", 0)
        if len(ins) == 1:
            return f"np.asarray({ins[0]})"
        joined = ", ".join(f"np.asarray({expr})" for expr in ins)
        return f"np.concatenate(({joined}), axis={axis})"
    if node.op == "CONSTANT_OF_SHAPE":
        value = repr(node.attrs.get("value", 0.0))
        return f"np.full(tuple(int(v) for v in np.asarray({ins[0]}).reshape(-1).tolist()), {value})"
    if node.op == "CAST":
        dtype = _attr_int(node.attrs, "to", -1)
        return f"_onnx_numpy_cast({ins[0]}, {dtype})"
    if node.op == "SHAPE":
        start = _attr_int(node.attrs, "start", 0)
        end = node.attrs.get("end")
        end_expr = "None" if end is None else str(_attr_int(node.attrs, "end", 0))
        return (
            "np.asarray("
            f"list({ins[0]}.shape)[slice({start}, {end_expr})], "
            "dtype=np.int64 if jax.config.x64_enabled else np.int32)"
        )
    if node.op == "RESHAPE":
        allowzero = _attr_int(node.attrs, "allowzero", 0) == 1
        return f"_onnx_numpy_reshape({ins[0]}, {ins[1]}, allowzero={allowzero})"
    if node.op == "UNSQUEEZE":
        if isinstance(node.attrs.get("axes"), list):
            axes = tuple(_attr_int_list(node.attrs, "axes", []))
            return f"_onnx_numpy_unsqueeze({ins[0]}, {axes})"
        if len(ins) > 1:
            return f"_onnx_numpy_unsqueeze({ins[0]}, {ins[1]})"
        return None
    if node.op == "SQUEEZE":
        if isinstance(node.attrs.get("axes"), list):
            axes = tuple(_attr_int_list(node.attrs, "axes", []))
            return f"_onnx_numpy_squeeze({ins[0]}, {axes})"
        if len(ins) > 1:
            return f"_onnx_numpy_squeeze({ins[0]}, {ins[1]})"
        return f"np.squeeze({ins[0]})"
    if node.op == "SLICE":
        axes = ins[3] if len(ins) > 3 else "None"
        steps = ins[4] if len(ins) > 4 else "None"
        return f"_onnx_numpy_slice({ins[0]}, {ins[1]}, {ins[2]}, {axes}, {steps})"
    if node.op == "GATHER":
        axis = _attr_int(node.attrs, "axis", 0)
        return f"np.take({ins[0]}, np.asarray({ins[1]}, dtype=np.int64), axis={axis})"
    if node.op == "GATHERELEMENTS":
        axis = _attr_int(node.attrs, "axis", 0)
        return f"np.take_along_axis({ins[0]}, np.asarray({ins[1]}, dtype=np.int64), axis={axis})"
    if node.op == "EXPAND":
        return f"_onnx_numpy_expand({ins[0]}, {ins[1]})"
    return None


def render_jax_module(ir: GraphIR, *, entrypoint: str = "forward") -> str:
    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "import jax",
        "import jax.numpy as jnp",
        "import jax.nn as jnn",
        "",
    ]

    ops = _ops_in_graph(ir)
    static_meta, static_meta_params = _static_meta_analysis(ir)
    static_meta_ops = {
        node.op for node in ir.nodes if any(output in static_meta for output in node.outputs)
    }

    if static_meta_ops:
        lines.extend(
            [
                "_ONNX_NUMPY_DTYPE_MAP = {",
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
                "def _onnx_numpy_cast(data, to):",
                "    dtype = _ONNX_NUMPY_DTYPE_MAP.get(int(to))",
                "    if dtype is None:",
                "        raise ValueError(f'Unsupported ONNX Cast dtype enum for shape metadata: {to}')",
                "    return np.asarray(data).astype(dtype)",
                "",
                "def _onnx_numpy_reshape(data, shape, *, allowzero=False):",
                "    target = [int(v) for v in np.asarray(shape).reshape(-1).tolist()]",
                "    if not allowzero:",
                "        data_shape = list(np.asarray(data).shape)",
                "        for idx, dim in enumerate(target):",
                "            if dim == 0:",
                "                if idx >= len(data_shape):",
                "                    raise ValueError('Reshape zero-copy index exceeds input rank.')",
                "                target[idx] = int(data_shape[idx])",
                "    return np.reshape(data, tuple(target))",
                "",
                "def _onnx_numpy_unsqueeze(data, axes):",
                "    axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]",
                "    out_rank = np.asarray(data).ndim + len(axes_l)",
                "    normalized = sorted((axis if axis >= 0 else axis + out_rank) for axis in axes_l)",
                "    out = np.asarray(data)",
                "    for axis in normalized:",
                "        out = np.expand_dims(out, axis)",
                "    return out",
                "",
                "def _onnx_numpy_squeeze(data, axes=None):",
                "    if axes is None:",
                "        return np.squeeze(data)",
                "    axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]",
                "    if not axes_l:",
                "        return np.asarray(data)",
                "    return np.squeeze(data, axis=tuple(sorted(set(axes_l))))",
                "",
                "def _onnx_numpy_slice(data, starts, ends, axes=None, steps=None):",
                "    starts_l = [int(v) for v in np.asarray(starts).tolist()]",
                "    ends_l = [int(v) for v in np.asarray(ends).tolist()]",
                "    axes_l = list(range(len(starts_l))) if axes is None else [int(v) for v in np.asarray(axes).tolist()]",
                "    steps_l = [1] * len(starts_l) if steps is None else [int(v) for v in np.asarray(steps).tolist()]",
                "    slices = [slice(None)] * np.asarray(data).ndim",
                "    for s, e, a, st in zip(starts_l, ends_l, axes_l, steps_l, strict=False):",
                "        slices[a] = slice(s, e, st)",
                "    return np.asarray(data)[tuple(slices)]",
                "",
                "def _onnx_numpy_expand(data, shape):",
                "    requested = [int(v) for v in np.asarray(shape).reshape(-1).tolist()]",
                "    arr = np.asarray(data)",
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
                "    return np.broadcast_to(arr, tuple(dims))",
                "",
            ]
        )

    if "SLICE" in ops:
        lines.extend(
            [
                "def _onnx_slice(data, starts, ends, axes=None, steps=None):",
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
                "    slices = [slice(None)] * data.ndim",
                "    for s, e, a, st in zip(starts_l, ends_l, axes_l, steps_l, strict=False):",
                "        slices[a] = slice(s, e, st)",
                "    return data[tuple(slices)]",
                "",
            ]
        )

    if "SPLIT" in ops:
        lines.extend(
            [
                "def _onnx_split(data, split=None, *, axis=0, num_outputs=1):",
                "    axis_i = int(axis)",
                "    if axis_i < 0:",
                "        axis_i += data.ndim",
                "    if axis_i < 0 or axis_i >= data.ndim:",
                "        raise ValueError(f'Split axis out of range: {axis}')",
                "    if split is None:",
                "        count = int(num_outputs)",
                "        if count <= 0:",
                "            raise ValueError('Split requires num_outputs > 0 when split is omitted.')",
                "        return tuple(jnp.split(data, count, axis=axis_i))",
                "    split_l = [int(v) for v in np.asarray(split).reshape(-1).tolist()]",
                "    if not split_l:",
                "        count = int(num_outputs)",
                "        if count <= 0:",
                "            raise ValueError('Split requires num_outputs > 0 when split is empty.')",
                "        return tuple(jnp.split(data, count, axis=axis_i))",
                "    if len(split_l) == 1:",
                "        return (data,)",
                "    indices = np.cumsum(split_l[:-1], dtype=int).tolist()",
                "    return tuple(jnp.split(data, indices, axis=axis_i))",
                "",
            ]
        )

    if "CAST" in ops:
        lines.extend(
            [
                "_ONNX_DTYPE_MAP = {",
                "    1: jnp.float32,",
                "    2: jnp.uint8,",
                "    3: jnp.int8,",
                "    4: jnp.uint16,",
                "    5: jnp.int16,",
                "    6: jnp.int32,",
                "    7: jnp.int64,",
                "    9: jnp.bool_,",
                "    10: jnp.float16,",
                "    11: jnp.float64,",
                "    12: jnp.uint32,",
                "    13: jnp.uint64,",
                "    16: jnp.bfloat16,",
                "}",
                "for _onnx_dtype, _jnp_dtype_name in (",
                "    (17, 'float8_e4m3fn'),",
                "    (18, 'float8_e4m3fnuz'),",
                "    (19, 'float8_e5m2'),",
                "    (20, 'float8_e5m2fnuz'),",
                "    (24, 'float8_e8m0fnu'),",
                "):",
                "    _jnp_dtype = getattr(jnp, _jnp_dtype_name, None)",
                "    if _jnp_dtype is not None:",
                "        _ONNX_DTYPE_MAP[_onnx_dtype] = _jnp_dtype",
                "del _onnx_dtype, _jnp_dtype_name, _jnp_dtype",
                "",
                "def _onnx_cast(data, to):",
                "    dtype = _ONNX_DTYPE_MAP.get(int(to))",
                "    if dtype is None:",
                "        raise ValueError(f'Unsupported ONNX Cast dtype enum: {to}')",
                "    return data.astype(dtype)",
                "",
            ]
        )

    if "CUMSUM" in ops:
        lines.extend(
            [
                "def _onnx_cumsum(data, axis, *, exclusive=False, reverse=False):",
                "    axis_i = int(np.asarray(axis).reshape(-1)[0])",
                "    arr = data",
                "    if reverse:",
                "        arr = jnp.flip(arr, axis=axis_i)",
                "    out = jnp.cumsum(arr, axis=axis_i)",
                "    if exclusive:",
                "        head = jnp.take(out, jnp.asarray([0], dtype=jnp.int32), axis=axis_i)",
                "        zeros = jnp.zeros_like(head)",
                "        tail_slices = [slice(None)] * out.ndim",
                "        tail_slices[axis_i] = slice(0, -1)",
                "        out = jnp.concatenate((zeros, out[tuple(tail_slices)]), axis=axis_i)",
                "    if reverse:",
                "        out = jnp.flip(out, axis=axis_i)",
                "    return out",
                "",
            ]
        )

    if "RESHAPE" in ops:
        lines.extend(
            [
                "def _onnx_reshape(data, shape, *, allowzero=False):",
                "    target = [int(v) for v in np.asarray(shape).reshape(-1).tolist()]",
                "    if not allowzero:",
                "        data_shape = list(data.shape)",
                "        for idx, dim in enumerate(target):",
                "            if dim == 0:",
                "                if idx >= len(data_shape):",
                "                    raise ValueError('Reshape zero-copy index exceeds input rank.')",
                "                target[idx] = int(data_shape[idx])",
                "    return jnp.reshape(data, tuple(target))",
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
                "        out = jnp.expand_dims(out, axis)",
                "    return out",
                "",
            ]
        )

    if "SQUEEZE" in ops:
        lines.extend(
            [
                "def _onnx_squeeze(data, axes=None):",
                "    if axes is None:",
                "        return jnp.squeeze(data)",
                "    axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]",
                "    if not axes_l:",
                "        return data",
                "    normalized = []",
                "    for axis in axes_l:",
                "        axis_i = axis if axis >= 0 else data.ndim + axis",
                "        if axis_i < 0 or axis_i >= data.ndim:",
                "            raise ValueError(f'Squeeze axis out of range: {axis}')",
                "        normalized.append(axis_i)",
                "    return jnp.squeeze(data, axis=tuple(sorted(set(normalized))))",
                "",
            ]
        )

    if "CONSTANT_OF_SHAPE" in ops:
        lines.extend(
            [
                "def _onnx_constant_of_shape(shape, *, value=0.0):",
                "    dims = tuple(int(v) for v in np.asarray(shape).reshape(-1).tolist())",
                "    return jnp.full(dims, value)",
                "",
            ]
        )

    if "EXPAND" in ops:
        lines.extend(
            [
                "def _onnx_expand(data, shape):",
                "    requested = [int(v) for v in np.asarray(shape).reshape(-1).tolist()]",
                "    data_shape = list(data.shape)",
                "    if len(requested) < len(data_shape):",
                "        requested = [1] * (len(data_shape) - len(requested)) + requested",
                "    elif len(requested) > len(data_shape):",
                "        expand_rank = len(requested) - len(data_shape)",
                "        data = jnp.reshape(data, (1,) * expand_rank + tuple(data_shape))",
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
                "    return jnp.broadcast_to(data, tuple(dims))",
                "",
            ]
        )

    if "PAD" in ops:
        lines.extend(
            [
                "def _onnx_pad(data, pads, constant_value=None, axes=None, *, mode='constant'):",
                "    pads_l = [int(v) for v in np.asarray(pads).reshape(-1).tolist()]",
                "    if len(pads_l) % 2 != 0:",
                "        raise ValueError('Pad helper expects begin/end pad pairs.')",
                "    if axes is None:",
                "        axes_l = list(range(len(pads_l) // 2))",
                "    else:",
                "        axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]",
                "    if len(axes_l) * 2 != len(pads_l):",
                "        raise ValueError('Pad helper received mismatched pads and axes lengths.')",
                "    pad_cfg = [(0, 0)] * data.ndim",
                "    half = len(pads_l) // 2",
                "    for idx, axis in enumerate(axes_l):",
                "        axis_i = axis if axis >= 0 else data.ndim + axis",
                "        if axis_i < 0 or axis_i >= data.ndim:",
                "            raise ValueError(f'Pad axis out of range: {axis}')",
                "        pad_cfg[axis_i] = (pads_l[idx], pads_l[idx + half])",
                "    mode_l = str(mode).lower()",
                "    if mode_l == 'constant':",
                "        cval = 0.0 if constant_value is None else float(np.asarray(constant_value).reshape(-1)[0])",
                "        return jnp.pad(data, tuple(pad_cfg), mode='constant', constant_values=cval)",
                "    if mode_l in {'edge', 'reflect'}:",
                "        return jnp.pad(data, tuple(pad_cfg), mode=mode_l)",
                "    raise ValueError(f'Unsupported ONNX Pad mode: {mode}')",
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
                "    return jnp.mean(data, axis=axes_t, keepdims=keepdims)",
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
                "    return jnp.sum(data, axis=axes_t, keepdims=keepdims)",
                "",
            ]
        )

    if "SHAPE" in ops:
        lines.extend(
            [
                "_ONNX_SHAPE_DTYPE = jnp.int64 if jax.config.x64_enabled else jnp.int32",
                "",
                "def _onnx_shape(data, *, start=0, end=None):",
                "    dims = list(data.shape)",
                "    return jnp.asarray(",
                "        dims[slice(int(start), None if end is None else int(end))],",
                "        dtype=_ONNX_SHAPE_DTYPE,",
                "    )",
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
                "    return jnp.reshape(data, (left, right))",
                "",
            ]
        )

    if "BATCHNORM" in ops:
        lines.extend(
            [
                "def _onnx_batchnorm(data, scale, bias, mean, var, epsilon=1e-5):",
                "    reshape_dims = (1, -1) + (1,) * max(data.ndim - 2, 0)",
                "    scale_r = jnp.reshape(scale, reshape_dims)",
                "    bias_r = jnp.reshape(bias, reshape_dims)",
                "    mean_r = jnp.reshape(mean, reshape_dims)",
                "    var_r = jnp.reshape(var, reshape_dims)",
                "    return ((data - mean_r) / jnp.sqrt(var_r + epsilon)) * scale_r + bias_r",
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
                "    scale_r = jnp.reshape(scale, reshape_dims)",
                "    bias_r = jnp.reshape(bias, reshape_dims)",
                "    reduce_axes = tuple(range(2, data.ndim))",
                "    mean = jnp.mean(data, axis=reduce_axes, keepdims=True)",
                "    var = jnp.var(data, axis=reduce_axes, keepdims=True)",
                "    normalized = (data - mean) / jnp.sqrt(var + epsilon)",
                "    return normalized * scale_r + bias_r",
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
                "    rotary_i = int(rotary_dim)",
                "    if rotary_i <= 0:",
                "        return data",
                "    half = rotary_i // 2",
                "    pos = jnp.asarray(position_ids, dtype=jnp.int32)",
                "    cos = jnp.take(jnp.asarray(cos_cache), pos, axis=0)",
                "    sin = jnp.take(jnp.asarray(sin_cache), pos, axis=0)",
                "    cos = jnp.expand_dims(cos, axis=1)",
                "    sin = jnp.expand_dims(sin, axis=1)",
                "    left = data[..., :half]",
                "    right = data[..., half:rotary_i]",
                "    rotated = jnp.concatenate((",
                "        left * cos - right * sin,",
                "        right * cos + left * sin,",
                "    ), axis=-1)",
                "    return jnp.concatenate((rotated, data[..., rotary_i:]), axis=-1)",
                "",
            ]
        )

    if "SKIPRMSNORM" in ops:
        lines.extend(
            [
                "def _onnx_skip_rmsnorm(data, skip, gamma, epsilon=1e-5, *, return_residual=True):",
                "    residual = data + skip",
                "    rms = jnp.mean(jnp.square(residual), axis=-1, keepdims=True)",
                "    normalized = (residual / jnp.sqrt(rms + epsilon)) * gamma",
                "    if return_residual:",
                "        return normalized, residual",
                "    return normalized",
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
                "    batch, query_len, q_width = query.shape",
                "    num_heads_i = int(num_heads)",
                "    kv_heads_i = int(kv_num_heads)",
                "    if num_heads_i <= 0 or kv_heads_i <= 0 or q_width % num_heads_i != 0:",
                "        raise ValueError('Invalid GroupQueryAttention head configuration.')",
                "    head_dim = q_width // num_heads_i",
                "    if key.shape[-1] != kv_heads_i * head_dim or value.shape[-1] != kv_heads_i * head_dim:",
                "        raise ValueError('GroupQueryAttention key/value width mismatch.')",
                "    if num_heads_i % kv_heads_i != 0:",
                "        raise ValueError('num_heads must be divisible by kv_num_heads.')",
                "    query_h = jnp.transpose(jnp.reshape(query, (batch, query_len, num_heads_i, head_dim)), (0, 2, 1, 3))",
                "    key_h = jnp.transpose(jnp.reshape(key, (batch, query_len, kv_heads_i, head_dim)), (0, 2, 1, 3))",
                "    value_h = jnp.transpose(jnp.reshape(value, (batch, query_len, kv_heads_i, head_dim)), (0, 2, 1, 3))",
                "    present_key = jnp.concatenate((past_key, key_h), axis=2)",
                "    present_value = jnp.concatenate((past_value, value_h), axis=2)",
                "    repeats = num_heads_i // kv_heads_i",
                "    full_key = jnp.repeat(present_key, repeats, axis=1)",
                "    full_value = jnp.repeat(present_value, repeats, axis=1)",
                "    scores = jnp.einsum('bhqd,bhkd->bhqk', query_h, full_key) * float(scale)",
                "    if float(softcap) > 0.0:",
                "        scores = jnp.tanh(scores / float(softcap)) * float(softcap)",
                "    if attention_bias is not None:",
                "        scores = scores + jnp.asarray(attention_bias, dtype=scores.dtype)",
                "    probs = jnn.softmax(scores, axis=-1)",
                "    context = jnp.einsum('bhqk,bhkd->bhqd', probs, full_value)",
                "    context = jnp.reshape(jnp.transpose(context, (0, 2, 1, 3)), (batch, query_len, q_width))",
                "    return context, present_key, present_value",
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
                "            return jnp.max(data, axis=axes, keepdims=True)",
                "        return jnp.mean(data, axis=axes, keepdims=True)",
                "    kernel_l = tuple(int(v) for v in kernel)",
                "    strides_l = tuple(int(v) for v in strides)",
                "    pads_l = tuple(int(v) for v in pads)",
                "    if data.ndim == 3:",
                "        pad_l, pad_r = (pads_l[0], pads_l[1]) if len(pads_l) >= 2 else (0, 0)",
                "        pad_cfg = ((0, 0), (0, 0), (pad_l, pad_r))",
                "        window = (1, 1, kernel_l[0])",
                "        stride_cfg = (1, 1, strides_l[0])",
                "    elif data.ndim == 4:",
                "        if len(pads_l) == 2:",
                "            pads_l = (pads_l[0], pads_l[1], pads_l[0], pads_l[1])",
                "        elif len(pads_l) < 4:",
                "            pads_l = (0, 0, 0, 0)",
                "        pad_cfg = ((0, 0), (0, 0), (pads_l[0], pads_l[2]), (pads_l[1], pads_l[3]))",
                "        window = (1, 1, kernel_l[0], kernel_l[1])",
                "        stride_cfg = (1, 1, strides_l[0], strides_l[1])",
                "    else:",
                "        raise ValueError('Pooling helper supports NCW and NCHW tensors only.')",
                "    if kind == 'max':",
                "        padded = jnp.pad(data, pad_cfg, constant_values=-jnp.inf)",
                "        return jax.lax.reduce_window(padded, -jnp.inf, jax.lax.max, window, stride_cfg, 'VALID')",
                "    padded = jnp.pad(data, pad_cfg)",
                "    total = jax.lax.reduce_window(padded, 0.0, jax.lax.add, window, stride_cfg, 'VALID')",
                "    counts = jax.lax.reduce_window(",
                "        jnp.pad(jnp.ones_like(data), pad_cfg),",
                "        0.0,",
                "        jax.lax.add,",
                "        window,",
                "        stride_cfg,",
                "        'VALID',",
                "    )",
                "    return total / jnp.maximum(counts, 1.0)",
                "",
            ]
        )

    if "UPSAMPLE" in ops:
        lines.extend(
            [
                "def _resolve_resize_target(data, scales=None, sizes=None):",
                "    if sizes is not None:",
                "        arr = np.asarray(sizes).reshape(-1)",
                "        if arr.size == data.ndim and np.all(arr > 0) and np.all(np.equal(arr, np.round(arr))):",
                "            return tuple(int(v) for v in arr.tolist())",
                "    if scales is not None:",
                "        arr = np.asarray(scales).reshape(-1)",
                "        if arr.size == data.ndim:",
                "            return tuple(",
                "                max(1, int(round(dim * float(scale))))",
                "                for dim, scale in zip(data.shape, arr.tolist(), strict=False)",
                "            )",
                "    raise ValueError('Resize helper needs scales or sizes with rank-matched values.')",
                "",
                "def _onnx_resize_nearest(data, roi=None, scales=None, sizes=None):",
                "    del roi",
                "    if scales is None and sizes is None:",
                "        return data",
                "    target_shape = _resolve_resize_target(data, scales=scales, sizes=sizes)",
                "    out = data",
                "    for axis, out_dim in enumerate(target_shape):",
                "        in_dim = out.shape[axis]",
                "        if out_dim == in_dim:",
                "            continue",
                "        scale = float(in_dim) / float(out_dim)",
                "        idx = np.floor(np.arange(out_dim) * scale).astype(np.int32)",
                "        idx = np.clip(idx, 0, max(in_dim - 1, 0))",
                "        out = jnp.take(out, jnp.asarray(idx, dtype=jnp.int32), axis=axis)",
                "    return out",
                "",
            ]
        )

    if "CLIP" in ops:
        lines.extend(
            [
                "def _onnx_clip(data, min_val=None, max_val=None):",
                "    a_min = None if min_val is None else float(np.asarray(min_val).reshape(-1)[0])",
                "    a_max = None if max_val is None else float(np.asarray(max_val).reshape(-1)[0])",
                "    return jnp.clip(data, a_min=a_min, a_max=a_max)",
                "",
            ]
        )

    if "MOD" in ops:
        lines.extend(
            [
                "def _onnx_mod(left, right, *, fmod=False):",
                "    if fmod:",
                "        return jnp.fmod(left, right)",
                "    return jnp.mod(left, right)",
                "",
            ]
        )

    if "ARANGE" in ops:
        lines.extend(
            [
                "def _onnx_arange(start, limit, delta):",
                "    start_v = float(np.asarray(start).reshape(-1)[0])",
                "    limit_v = float(np.asarray(limit).reshape(-1)[0])",
                "    delta_v = float(np.asarray(delta).reshape(-1)[0])",
                "    return jnp.arange(start_v, limit_v, delta_v)",
                "",
            ]
        )

    if "SCATTERND" in ops:
        lines.extend(
            [
                "def _onnx_scatter_nd(data, indices, updates):",
                "    idx = jnp.asarray(indices, dtype=jnp.int32)",
                "    if idx.ndim == 0:",
                "        raise ValueError('ScatterND indices must have rank >= 1.')",
                "    if idx.shape[-1] == 0:",
                "        raise ValueError('ScatterND indices must include at least one index dimension.')",
                "    flat_idx = jnp.reshape(idx, (-1, idx.shape[-1]))",
                "    slice_shape = tuple(data.shape[int(idx.shape[-1]) :])",
                "    flat_updates = jnp.reshape(jnp.asarray(updates), (-1,) + slice_shape)",
                "    index_tuple = tuple(flat_idx[:, axis] for axis in range(int(idx.shape[-1])))",
                "    if slice_shape:",
                "        return data.at[index_tuple].set(flat_updates)",
                "    return data.at[index_tuple].set(jnp.reshape(flat_updates, (-1,)))",
                "",
            ]
        )

    if "TRILU" in ops:
        lines.extend(
            [
                "def _onnx_trilu(data, diagonal=None, *, upper=True):",
                "    diag = 0 if diagonal is None else int(np.asarray(diagonal).reshape(-1)[0])",
                "    if upper:",
                "        return jnp.triu(data, k=diag)",
                "    return jnp.tril(data, k=diag)",
                "",
            ]
        )

    if ops & {"QUANTIZE", "DEQUANTIZE"}:
        lines.extend(
            [
                "def _onnx_quantize(data, scale, zero_point=None):",
                "    zp = 0 if zero_point is None else jnp.asarray(zero_point)",
                "    out = jnp.round(jnp.asarray(data) / jnp.asarray(scale)) + zp",
                "    if zero_point is not None:",
                "        return out.astype(jnp.asarray(zero_point).dtype)",
                "    return out.astype(jnp.uint8)",
                "",
                "def _onnx_dequantize(data, scale, zero_point=None):",
                "    zp = 0.0 if zero_point is None else jnp.asarray(zero_point, dtype=jnp.float32)",
                "    return (jnp.asarray(data, dtype=jnp.float32) - zp) * jnp.asarray(scale, dtype=jnp.float32)",
                "",
            ]
        )

    lines.extend(
        [
            "def _load_weight_value(value):",
            "    if value.dtype.kind in {'i', 'u', 'b'}:",
            "        return value",
            "    if value.dtype.kind == 'V' and value.dtype.itemsize == 2:",
            "        bits = value.view(np.uint16).reshape(value.shape)",
            "        return jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16)",
            "    return jnp.asarray(value)",
            "",
            "def load_weights(path: str) -> dict[str, jnp.ndarray]:",
            "    data = np.load(path)",
            "    return {",
            "        k: _load_weight_value(data[k])",
            "        for k in sorted(data.files)",
            "    }",
            "",
            "def forward(",
            "    params: dict[str, jnp.ndarray],",
            "    inputs: dict[str, jnp.ndarray],",
            ") -> dict[str, jnp.ndarray]:",
            "    tensors: dict[str, jnp.ndarray] = {}",
        ]
    )

    for input_name in ir.inputs:
        lines.append(f'    tensors["{input_name}"] = jnp.asarray(inputs["{input_name}"])')
    for node in _scheduled_nodes(ir):
        lines.extend(_emit_node_lines(node, ir, static_meta))
    outputs_literal = ", ".join([f'"{o}": tensors["{o}"]' for o in ir.outputs])
    lines.append(f"    return {{{outputs_literal}}}")
    lines.append("")
    if static_meta_params:
        lines.extend(
            [
                f"_STATIC_META_PARAM_NAMES = frozenset({tuple(sorted(static_meta_params))!r})",
                "_FORWARD_JIT_CACHE = {}",
                "",
                "def _static_meta_cache_key(static_params):",
                "    return tuple(",
                "        (key, np.asarray(value).dtype.str, tuple(np.asarray(value).shape), np.asarray(value).tobytes())",
                "        for key, value in sorted(static_params.items())",
                "    )",
                "",
                "def forward_jit(params: dict[str, jnp.ndarray], inputs: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:",
                "    static_params = {",
                "        key: params[key]",
                "        for key in _STATIC_META_PARAM_NAMES",
                "        if key in params",
                "    }",
                "    dynamic_params = {",
                "        key: value",
                "        for key, value in params.items()",
                "        if key not in _STATIC_META_PARAM_NAMES",
                "    }",
                "    cache_key = _static_meta_cache_key(static_params)",
                "    compiled = _FORWARD_JIT_CACHE.get(cache_key)",
                "    if compiled is None:",
                "        def _closed_forward(actual_params, actual_inputs):",
                "            merged = dict(static_params)",
                "            merged.update(actual_params)",
                "            return forward(merged, actual_inputs)",
                "",
                "        compiled = jax.jit(_closed_forward)",
                "        _FORWARD_JIT_CACHE[cache_key] = compiled",
                "    return compiled(dynamic_params, inputs)",
            ]
        )
    else:
        lines.append("forward_jit = jax.jit(forward)")
    if entrypoint != "forward":
        lines.append(f"{entrypoint} = forward")
    lines.append("")
    return "\n".join(lines)


def emit_jax_module(ir: GraphIR, out_dir: str | Path, *, entrypoint: str = "forward") -> Path:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "model_jax.py"
    out_path.write_text(render_jax_module(ir, entrypoint=entrypoint), encoding="utf-8")
    return out_path
