from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

type AttrScalar = int | float | str | bool
type AttrValue = AttrScalar | list[int] | list[float] | list[str]
type Dim = int | str
type TensorKind = Literal["input", "output", "initializer", "intermediate"]


@dataclass(slots=True)
class TensorRef:
    name: str
    dtype: str
    shape: list[Dim]
    kind: TensorKind

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": self.shape,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> TensorRef:
        shape = payload.get("shape", [])
        if not isinstance(shape, list):
            raise TypeError(f"Tensor shape for '{name}' must be a list.")
        return cls(
            name=name,
            dtype=str(payload.get("dtype", "float32")),
            shape=[int(v) if isinstance(v, int) else str(v) for v in shape],
            kind=str(payload.get("kind", "intermediate")),  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class OpNode:
    id: str
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, AttrValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OpNode:
        return cls(
            id=str(payload["id"]),
            op=str(payload["op"]).upper(),
            inputs=[str(x) for x in payload.get("inputs", [])],
            outputs=[str(x) for x in payload.get("outputs", [])],
            attrs=dict(payload.get("attrs", {})),
        )


@dataclass(slots=True)
class GraphIR:
    name: str
    opset: int
    tensors: dict[str, TensorRef]
    nodes: list[OpNode]
    inputs: list[str]
    outputs: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        ordered_tensors = {key: self.tensors[key].to_dict() for key in sorted(self.tensors.keys())}
        return {
            "name": self.name,
            "opset": self.opset,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "tensors": ordered_tensors,
            "nodes": [node.to_dict() for node in self.nodes],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphIR:
        tensor_payload = payload.get("tensors", {})
        if not isinstance(tensor_payload, dict):
            raise TypeError("GraphIR 'tensors' must be a dictionary.")
        tensors = {key: TensorRef.from_dict(key, value) for key, value in tensor_payload.items()}
        nodes = [OpNode.from_dict(node) for node in payload.get("nodes", [])]
        return cls(
            name=str(payload.get("name", "graph")),
            opset=int(payload.get("opset", 18)),
            tensors=tensors,
            nodes=nodes,
            inputs=[str(x) for x in payload.get("inputs", [])],
            outputs=[str(x) for x in payload.get("outputs", [])],
            metadata=dict(payload.get("metadata", {})),
        )
