from __future__ import annotations

from dataclasses import dataclass

from ..ir.types import GraphIR, OpNode


def order_nodes_for_emission(ir: GraphIR) -> list[OpNode]:
    # Nodes are expected in ONNX topological order; keep stable deterministic order.
    return list(ir.nodes)


@dataclass(slots=True)
class NameGenerator:
    _counter: int = 0

    def next_temp(self) -> str:
        name = f"_v{self._counter}"
        self._counter += 1
        return name
