from __future__ import annotations

import json

from .schema import validate_graph
from .types import GraphIR


def dumps_graph_json(ir: GraphIR) -> str:
    validate_graph(ir)
    return json.dumps(ir.to_dict(), sort_keys=True, indent=2)


def loads_graph_json(payload: str) -> GraphIR:
    raw = json.loads(payload)
    ir = GraphIR.from_dict(raw)
    validate_graph(ir)
    return ir
