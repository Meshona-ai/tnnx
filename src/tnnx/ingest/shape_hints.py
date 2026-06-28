from __future__ import annotations

from onnx import ValueInfoProto


def shape_from_value_info(value_info: ValueInfoProto) -> list[int | str]:
    dims: list[int | str] = []
    tensor_type = value_info.type.tensor_type
    for idx, dim in enumerate(tensor_type.shape.dim):
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
            continue
        if dim.HasField("dim_param") and dim.dim_param:
            dims.append(dim.dim_param)
            continue
        dims.append(f"dim_{idx}")
    return dims
