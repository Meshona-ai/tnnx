from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper

from ..ir.schema import validate_graph
from ..ir.types import GraphIR, OpNode, TensorKind, TensorRef
from .dtypes import onnx_dtype_to_str
from .op_map import ONNX_TO_SEMANTIC
from .shape_hints import shape_from_value_info


class UnsupportedOpError(ValueError):
    pass


def extract_initializers(model: ModelProto) -> dict[str, np.ndarray[Any, Any]]:
    graph = model.graph
    initializers: dict[str, np.ndarray[Any, Any]] = {}
    for initializer in sorted(graph.initializer, key=lambda item: item.name):
        initializers[initializer.name] = numpy_helper.to_array(initializer)
    return initializers


def _collect_value_info(model: ModelProto) -> dict[str, tuple[str, list[int | str]]]:
    graph = model.graph
    infos = [*graph.input, *graph.output, *graph.value_info]
    mapping: dict[str, tuple[str, list[int | str]]] = {}
    for info in infos:
        tensor_type = info.type.tensor_type
        if tensor_type.elem_type == TensorProto.UNDEFINED:
            continue
        dtype = onnx_dtype_to_str(tensor_type.elem_type)
        mapping[info.name] = (dtype, shape_from_value_info(info))
    return mapping


def _normalize_attr(value: Any) -> Any:
    if isinstance(value, TensorProto):
        array = numpy_helper.to_array(value)
        if array.size == 1:
            return _normalize_attr(array.item())
        return _normalize_attr(array.reshape(-1).tolist())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_normalize_attr(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tensor_ref_for(
    name: str,
    kind: TensorKind,
    info_map: dict[str, tuple[str, list[int | str]]],
    initializers: dict[str, np.ndarray[Any, Any]],
) -> TensorRef:
    if name in initializers:
        arr = initializers[name]
        return TensorRef(name=name, dtype=str(arr.dtype), shape=list(arr.shape), kind=kind)
    dtype, shape = info_map.get(name, ("float32", []))
    return TensorRef(name=name, dtype=dtype, shape=shape, kind=kind)


def load_onnx_to_ir(
    onnx_path: str | Path,
    *,
    infer_shapes: bool = True,
) -> tuple[GraphIR, dict[str, np.ndarray[Any, Any]]]:
    path = Path(onnx_path)
    model = onnx.load(path)
    if infer_shapes:
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:
            pass

    graph = model.graph
    initializers = extract_initializers(model)
    info_map = _collect_value_info(model)
    onnx_inputs = [inp.name for inp in graph.input if inp.name not in initializers]
    onnx_outputs = [out.name for out in graph.output]

    tensors: dict[str, TensorRef] = {}
    for input_name in onnx_inputs:
        tensors[input_name] = _tensor_ref_for(input_name, "input", info_map, initializers)
    for output_name in onnx_outputs:
        tensors[output_name] = _tensor_ref_for(output_name, "output", info_map, initializers)
    for init_name in sorted(initializers.keys()):
        tensors[init_name] = _tensor_ref_for(init_name, "initializer", info_map, initializers)

    nodes: list[OpNode] = []
    for index, node in enumerate(graph.node):
        if node.op_type == "Constant":
            value_attr = next((a for a in node.attribute if a.name == "value"), None)
            if value_attr is not None:
                tensor_proto = helper.get_attribute_value(value_attr)
                array = numpy_helper.to_array(tensor_proto)
                output_name = node.output[0]
                initializers[output_name] = array
                tensors[output_name] = _tensor_ref_for(
                    output_name, "initializer", info_map, initializers
                )
            continue

        semantic = ONNX_TO_SEMANTIC.get(node.op_type)
        if semantic is None:
            node_name = node.name or f"node_{index}"
            raise UnsupportedOpError(f"Unsupported ONNX op '{node.op_type}' at node '{node_name}'.")

        attrs = {
            attr.name: _normalize_attr(helper.get_attribute_value(attr)) for attr in node.attribute
        }
        if node.op_type in {"GlobalAveragePool", "GlobalMaxPool"}:
            attrs["global"] = 1
        raw_inputs = [str(i) for i in node.input]
        filtered_inputs = [name for name in raw_inputs if name]
        if node.op_type in {"GroupQueryAttention", "Pad", "Resize"} and len(filtered_inputs) != len(
            raw_inputs
        ):
            attrs["input_slots"] = [idx for idx, name in enumerate(raw_inputs) if name]

        op_node = OpNode(
            id=f"n{len(nodes)}",
            op=semantic,
            inputs=filtered_inputs,
            outputs=[str(o) for o in node.output if o],
            attrs=attrs,
        )
        for input_name in op_node.inputs:
            if input_name not in tensors:
                tensors[input_name] = _tensor_ref_for(
                    input_name, "intermediate", info_map, initializers
                )
        for output_name in op_node.outputs:
            if output_name in onnx_outputs:
                tensors[output_name] = _tensor_ref_for(
                    output_name, "output", info_map, initializers
                )
            else:
                tensors[output_name] = _tensor_ref_for(
                    output_name, "intermediate", info_map, initializers
                )
        nodes.append(op_node)

    ir = GraphIR(
        name=graph.name or path.stem,
        opset=model.opset_import[0].version if model.opset_import else 18,
        tensors=tensors,
        nodes=nodes,
        inputs=onnx_inputs,
        outputs=onnx_outputs,
        metadata={"source": "onnx", "deterministic": True},
    )
    validate_graph(ir)
    return ir, initializers
