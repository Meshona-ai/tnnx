from __future__ import annotations

import pytest
from onnx import TensorProto

from tnnx.ingest.dtypes import onnx_dtype_to_str


def test_onnx_dtype_mapping_supported() -> None:
    assert onnx_dtype_to_str(TensorProto.FLOAT) == "float32"
    assert onnx_dtype_to_str(TensorProto.BFLOAT16) == "bfloat16"
    assert onnx_dtype_to_str(TensorProto.FLOAT8E4M3FN) == "float8_e4m3fn"
    assert onnx_dtype_to_str(TensorProto.FLOAT8E4M3FNUZ) == "float8_e4m3fnuz"
    assert onnx_dtype_to_str(TensorProto.FLOAT8E5M2) == "float8_e5m2"
    assert onnx_dtype_to_str(TensorProto.FLOAT8E5M2FNUZ) == "float8_e5m2fnuz"
    assert onnx_dtype_to_str(TensorProto.FLOAT8E8M0) == "float8_e8m0fnu"
    assert onnx_dtype_to_str(TensorProto.INT64) == "int64"


def test_onnx_dtype_mapping_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported ONNX dtype"):
        onnx_dtype_to_str(TensorProto.STRING)
