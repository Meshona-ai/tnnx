from __future__ import annotations

from ..ir.types import GraphIR, TensorRef


def prune_dead_nodes(ir: GraphIR) -> GraphIR:
    live_tensors = set(ir.outputs)
    live_nodes = []

    for node in reversed(ir.nodes):
        if not any(output in live_tensors for output in node.outputs):
            continue
        live_nodes.append(node)
        live_tensors.update(node.inputs)

    live_nodes.reverse()
    retained_tensors: dict[str, TensorRef] = {}
    for name, tensor in ir.tensors.items():
        if (
            name in live_tensors
            or name in ir.inputs
            or name in ir.outputs
            or tensor.kind == "initializer"
        ):
            retained_tensors[name] = tensor

    return GraphIR(
        name=ir.name,
        opset=ir.opset,
        tensors=retained_tensors,
        nodes=live_nodes,
        inputs=list(ir.inputs),
        outputs=list(ir.outputs),
        metadata=dict(ir.metadata),
    )
