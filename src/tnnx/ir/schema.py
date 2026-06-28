from __future__ import annotations

from dataclasses import dataclass, field

from .types import GraphIR, OpNode


@dataclass(frozen=True, slots=True)
class OpSchema:
    op: str
    min_inputs: int
    max_inputs: int | None
    attr_types: dict[str, type | tuple[type, ...]] = field(default_factory=dict)


SEMANTIC_SCHEMAS: dict[str, OpSchema] = {
    "ADD": OpSchema("ADD", min_inputs=2, max_inputs=2),
    "AND": OpSchema("AND", min_inputs=2, max_inputs=2),
    "ARANGE": OpSchema("ARANGE", min_inputs=3, max_inputs=3),
    "AVGPOOL": OpSchema(
        "AVGPOOL",
        min_inputs=1,
        max_inputs=1,
        attr_types={"kernel_shape": list, "pads": list, "strides": list, "global": int},
    ),
    "BATCHNORM": OpSchema(
        "BATCHNORM",
        min_inputs=5,
        max_inputs=5,
        attr_types={"epsilon": (int, float)},
    ),
    "CAST": OpSchema("CAST", min_inputs=1, max_inputs=1, attr_types={"to": int}),
    "CLIP": OpSchema(
        "CLIP",
        min_inputs=1,
        max_inputs=3,
        attr_types={"min": (int, float), "max": (int, float)},
    ),
    "CONCAT": OpSchema("CONCAT", min_inputs=1, max_inputs=None, attr_types={"axis": int}),
    "CONSTANT_OF_SHAPE": OpSchema(
        "CONSTANT_OF_SHAPE",
        min_inputs=1,
        max_inputs=1,
        attr_types={"value": (int, float, bool)},
    ),
    "CONV2D": OpSchema("CONV2D", min_inputs=2, max_inputs=3),
    "COS": OpSchema("COS", min_inputs=1, max_inputs=1),
    "CUMSUM": OpSchema(
        "CUMSUM",
        min_inputs=2,
        max_inputs=2,
        attr_types={"exclusive": int, "reverse": int},
    ),
    "DEQUANTIZE": OpSchema("DEQUANTIZE", min_inputs=2, max_inputs=3),
    "DIV": OpSchema("DIV", min_inputs=2, max_inputs=2),
    "EQUAL": OpSchema("EQUAL", min_inputs=2, max_inputs=2),
    "ERF": OpSchema("ERF", min_inputs=1, max_inputs=1),
    "EXP": OpSchema("EXP", min_inputs=1, max_inputs=1),
    "EXPAND": OpSchema("EXPAND", min_inputs=2, max_inputs=2),
    "FLATTEN": OpSchema("FLATTEN", min_inputs=1, max_inputs=1, attr_types={"axis": int}),
    "GATHER": OpSchema("GATHER", min_inputs=2, max_inputs=2, attr_types={"axis": int}),
    "GATHERELEMENTS": OpSchema(
        "GATHERELEMENTS",
        min_inputs=2,
        max_inputs=2,
        attr_types={"axis": int},
    ),
    "GEMM": OpSchema(
        "GEMM",
        min_inputs=2,
        max_inputs=3,
        attr_types={"transA": int, "transB": int},
    ),
    "GELU": OpSchema("GELU", min_inputs=1, max_inputs=1),
    "GREATER": OpSchema("GREATER", min_inputs=2, max_inputs=2),
    "GREATEROREQUAL": OpSchema("GREATEROREQUAL", min_inputs=2, max_inputs=2),
    "GROUPQUERYATTENTION": OpSchema(
        "GROUPQUERYATTENTION",
        min_inputs=5,
        max_inputs=8,
        attr_types={
            "do_rotary": int,
            "input_slots": list,
            "kv_num_heads": int,
            "local_window_size": int,
            "num_heads": int,
            "rotary_interleaved": int,
            "scale": (int, float),
            "softcap": (int, float),
        },
    ),
    "IDENTITY": OpSchema("IDENTITY", min_inputs=1, max_inputs=1),
    "ISNAN": OpSchema("ISNAN", min_inputs=1, max_inputs=1),
    "INSTANCENORM": OpSchema(
        "INSTANCENORM",
        min_inputs=3,
        max_inputs=3,
        attr_types={"epsilon": (int, float)},
    ),
    "LAYERNORM": OpSchema(
        "LAYERNORM",
        min_inputs=2,
        max_inputs=3,
        attr_types={"axis": int, "epsilon": (int, float)},
    ),
    "LESS": OpSchema("LESS", min_inputs=2, max_inputs=2),
    "LESSOREQUAL": OpSchema("LESSOREQUAL", min_inputs=2, max_inputs=2),
    "LOG": OpSchema("LOG", min_inputs=1, max_inputs=1),
    "MATMUL": OpSchema("MATMUL", min_inputs=2, max_inputs=2),
    "MAXPOOL": OpSchema(
        "MAXPOOL",
        min_inputs=1,
        max_inputs=1,
        attr_types={"kernel_shape": list, "pads": list, "strides": list, "global": int},
    ),
    "MISH": OpSchema("MISH", min_inputs=1, max_inputs=1),
    "MOD": OpSchema("MOD", min_inputs=2, max_inputs=2, attr_types={"fmod": int}),
    "MUL": OpSchema("MUL", min_inputs=2, max_inputs=2),
    "NEG": OpSchema("NEG", min_inputs=1, max_inputs=1),
    "PAD": OpSchema(
        "PAD",
        min_inputs=2,
        max_inputs=4,
        attr_types={"mode": str, "input_slots": list},
    ),
    "POW": OpSchema("POW", min_inputs=2, max_inputs=2),
    "QUANTIZE": OpSchema("QUANTIZE", min_inputs=2, max_inputs=3),
    "RECIPROCAL": OpSchema("RECIPROCAL", min_inputs=1, max_inputs=1),
    "REDUCEMEAN": OpSchema(
        "REDUCEMEAN",
        min_inputs=1,
        max_inputs=2,
        attr_types={"axes": list, "keepdims": int},
    ),
    "REDUCESUM": OpSchema(
        "REDUCESUM",
        min_inputs=1,
        max_inputs=2,
        attr_types={"axes": list, "keepdims": int},
    ),
    "RELU": OpSchema("RELU", min_inputs=1, max_inputs=1),
    "RELU6": OpSchema("RELU6", min_inputs=1, max_inputs=1),
    "RESHAPE": OpSchema("RESHAPE", min_inputs=2, max_inputs=2),
    "RMSNORM": OpSchema(
        "RMSNORM",
        min_inputs=2,
        max_inputs=3,
        attr_types={"axis": int, "epsilon": (int, float)},
    ),
    "ROTARYEMBEDDING": OpSchema(
        "ROTARYEMBEDDING",
        min_inputs=4,
        max_inputs=4,
        attr_types={"interleaved": int, "num_heads": int, "rotary_embedding_dim": int},
    ),
    "SCATTERND": OpSchema("SCATTERND", min_inputs=3, max_inputs=3),
    "SHAPE": OpSchema(
        "SHAPE",
        min_inputs=1,
        max_inputs=1,
        attr_types={"start": int, "end": int},
    ),
    "SIGMOID": OpSchema("SIGMOID", min_inputs=1, max_inputs=1),
    "SILU": OpSchema("SILU", min_inputs=1, max_inputs=1),
    "SIN": OpSchema("SIN", min_inputs=1, max_inputs=1),
    "SLICE": OpSchema("SLICE", min_inputs=3, max_inputs=5),
    "SOFTMAX": OpSchema("SOFTMAX", min_inputs=1, max_inputs=1, attr_types={"axis": int}),
    "SOFTPLUS": OpSchema("SOFTPLUS", min_inputs=1, max_inputs=1),
    "SKIPRMSNORM": OpSchema(
        "SKIPRMSNORM",
        min_inputs=3,
        max_inputs=3,
        attr_types={"epsilon": (int, float)},
    ),
    "SPLIT": OpSchema(
        "SPLIT",
        min_inputs=1,
        max_inputs=2,
        attr_types={"axis": int, "num_outputs": int, "split": list},
    ),
    "SUB": OpSchema("SUB", min_inputs=2, max_inputs=2),
    "SQUEEZE": OpSchema("SQUEEZE", min_inputs=1, max_inputs=2, attr_types={"axes": list}),
    "SQRT": OpSchema("SQRT", min_inputs=1, max_inputs=1),
    "TANH": OpSchema("TANH", min_inputs=1, max_inputs=1),
    "TRANSPOSE": OpSchema("TRANSPOSE", min_inputs=1, max_inputs=1, attr_types={"perm": list}),
    "TRILU": OpSchema("TRILU", min_inputs=1, max_inputs=2, attr_types={"upper": int}),
    "UNSQUEEZE": OpSchema("UNSQUEEZE", min_inputs=1, max_inputs=2, attr_types={"axes": list}),
    "UPSAMPLE": OpSchema(
        "UPSAMPLE",
        min_inputs=1,
        max_inputs=4,
        attr_types={"mode": str, "input_slots": list},
    ),
    "WHERE": OpSchema("WHERE", min_inputs=3, max_inputs=3),
}


def _is_supported_attr_value(value: object) -> bool:
    if isinstance(value, int | float | str | bool):
        return True
    if isinstance(value, list):
        if not value:
            return True
        first = value[0]
        if isinstance(first, int):
            return all(isinstance(v, int) for v in value)
        if isinstance(first, float):
            return all(isinstance(v, float) for v in value)
        if isinstance(first, str):
            return all(isinstance(v, str) for v in value)
    return False


def validate_node(node: OpNode) -> None:
    schema = SEMANTIC_SCHEMAS.get(node.op)
    if schema is None:
        raise ValueError(f"Unsupported semantic op: {node.op}")

    input_count = len(node.inputs)
    if input_count < schema.min_inputs:
        raise ValueError(
            f"Op '{node.op}' expects at least {schema.min_inputs} inputs, got {input_count}."
        )
    if schema.max_inputs is not None and input_count > schema.max_inputs:
        raise ValueError(
            f"Op '{node.op}' expects at most {schema.max_inputs} inputs, got {input_count}."
        )
    for name, value in node.attrs.items():
        if not _is_supported_attr_value(value):
            raise TypeError(
                f"Node '{node.id}' has unsupported attribute type for '{name}': {type(value)}."
            )
        expected_type = schema.attr_types.get(name)
        if expected_type and not isinstance(value, expected_type):
            raise TypeError(
                f"Node '{node.id}' attr '{name}' expects {expected_type}, got {type(value)}."
            )


def validate_graph(ir: GraphIR) -> None:
    graph_sources = set(ir.inputs)
    graph_sources.update(
        name for name, tensor in ir.tensors.items() if tensor.kind == "initializer"
    )
    produced: dict[str, str] = {}

    for node in ir.nodes:
        validate_node(node)
        node_outputs: set[str] = set()
        for output_name in node.outputs:
            if output_name not in ir.tensors:
                raise KeyError(f"Node '{node.id}' references unknown output tensor: {output_name}")
            if output_name in node_outputs:
                raise ValueError(f"Node '{node.id}' lists duplicate output tensor: {output_name}")
            if output_name in produced:
                raise ValueError(
                    f"Tensor '{output_name}' is produced by both '{produced[output_name]}' "
                    f"and '{node.id}'."
                )
            if output_name in ir.inputs or ir.tensors[output_name].kind == "initializer":
                raise ValueError(
                    f"Node '{node.id}' output '{output_name}' conflicts with graph input "
                    "or initializer provenance."
                )
            node_outputs.add(output_name)
            produced[output_name] = node.id

    provenance = graph_sources | set(produced)
    for node in ir.nodes:
        for input_name in node.inputs:
            if input_name not in ir.tensors:
                raise KeyError(f"Node '{node.id}' references unknown input tensor: {input_name}")
            if input_name not in provenance:
                raise ValueError(
                    f"Node '{node.id}' input tensor '{input_name}' has no producer, "
                    "graph input, or initializer."
                )

    for key in ir.inputs + ir.outputs:
        if key not in ir.tensors:
            raise KeyError(f"Graph key is missing in tensor table: {key}")
    for output_name in ir.outputs:
        if output_name not in provenance:
            raise ValueError(
                f"Graph output '{output_name}' has no producer, graph input, or initializer."
            )
