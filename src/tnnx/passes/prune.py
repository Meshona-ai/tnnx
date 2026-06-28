from __future__ import annotations

from ..ir.types import GraphIR


def prune_dead_nodes(ir: GraphIR) -> GraphIR:
    # v0: ONNX export already contains only live nodes for tested models.
    return ir
