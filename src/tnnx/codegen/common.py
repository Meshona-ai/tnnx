from __future__ import annotations

from typing import Any

from ..ir.types import GraphIR, OpNode


def attr_int(attrs: dict[str, Any], key: str, default: int) -> int:
    value = attrs.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        return int(value)
    return default


def attr_int_list(attrs: dict[str, Any], key: str, default: list[int]) -> list[int]:
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


def attr_float(attrs: dict[str, Any], key: str, default: float) -> float:
    value = attrs.get(key, default)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float | str):
        return float(value)
    return default


def order_nodes_for_emission(ir: GraphIR) -> list[OpNode]:
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
