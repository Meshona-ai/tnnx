from __future__ import annotations

from ..ir.types import GraphIR


def normalize_graph(ir: GraphIR) -> GraphIR:
    # Keep ordering deterministic and normalize op names.
    for node in ir.nodes:
        node.op = node.op.upper()
    return ir
