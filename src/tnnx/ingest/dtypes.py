from __future__ import annotations

from onnx import TensorProto

ONNX_DTYPE_MAP: dict[int, str] = {
    TensorProto.FLOAT: "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.BFLOAT16: "bfloat16",
    TensorProto.DOUBLE: "float64",
    TensorProto.FLOAT8E4M3FN: "float8_e4m3fn",
    TensorProto.FLOAT8E4M3FNUZ: "float8_e4m3fnuz",
    TensorProto.FLOAT8E5M2: "float8_e5m2",
    TensorProto.FLOAT8E5M2FNUZ: "float8_e5m2fnuz",
    TensorProto.FLOAT8E8M0: "float8_e8m0fnu",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.BOOL: "bool",
}


def onnx_dtype_to_str(dtype: int) -> str:
    mapped = ONNX_DTYPE_MAP.get(dtype)
    if mapped is None:
        raise ValueError(f"Unsupported ONNX dtype: {dtype}")
    return mapped
