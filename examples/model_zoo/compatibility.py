from __future__ import annotations

from dataclasses import dataclass

from .catalog import ORDERED_MODEL_SPECS

CURRENT_BACKEND_OPS = frozenset(
    {
        "ADD",
        "AND",
        "ARANGE",
        "AVGPOOL",
        "BATCHNORM",
        "CAST",
        "CLIP",
        "CONCAT",
        "CONV2D",
        "COS",
        "DEQUANTIZE",
        "DIV",
        "ERF",
        "EXP",
        "FLATTEN",
        "GATHER",
        "GEMM",
        "GELU",
        "IDENTITY",
        "LAYERNORM",
        "LOG",
        "MATMUL",
        "MAXPOOL",
        "MISH",
        "MUL",
        "NEG",
        "PAD",
        "POW",
        "QUANTIZE",
        "REDUCEMEAN",
        "RELU",
        "RELU6",
        "RESHAPE",
        "RMSNORM",
        "SIGMOID",
        "SILU",
        "SIN",
        "SLICE",
        "SOFTMAX",
        "SUB",
        "SQRT",
        "TRANSPOSE",
        "UNSQUEEZE",
        "UPSAMPLE",
        "WHERE",
    }
)

REQUIRED_EXTRA_OPS = {
    "YOLOv2": frozenset({"CONCAT", "MAXPOOL", "SIGMOID"}),
    "YOLOv3-tiny": frozenset({"CONCAT", "PAD", "UPSAMPLE", "SIGMOID"}),
    "YOLOv4-tiny": frozenset({"CONCAT", "PAD", "UPSAMPLE", "SIGMOID", "MISH"}),
    "YOLOv5s": frozenset({"CONCAT", "PAD", "UPSAMPLE", "SIGMOID", "SILU"}),
    "YOLOv8n": frozenset({"CONCAT", "PAD", "UPSAMPLE", "SIGMOID", "SILU"}),
    "ResNet-18": frozenset({"BATCHNORM", "AVGPOOL", "FLATTEN"}),
    "ResNet-34": frozenset({"BATCHNORM", "AVGPOOL", "FLATTEN"}),
    "ResNet-50": frozenset({"BATCHNORM", "AVGPOOL", "FLATTEN"}),
    "ResNet-50 (quantized)": frozenset(
        {"BATCHNORM", "AVGPOOL", "FLATTEN", "QUANTIZE", "DEQUANTIZE"}
    ),
    "Small BERT variants": frozenset({"CAST", "SUB", "UNSQUEEZE"}),
    "Tiny ViT": frozenset({"CONCAT"}),
    "MobileNetV1/V2": frozenset({"BATCHNORM", "AVGPOOL", "FLATTEN", "RELU6", "CLIP"}),
    "EfficientNet-lite": frozenset({"BATCHNORM", "AVGPOOL", "FLATTEN", "SILU", "SIGMOID"}),
    "Llama 3.1 8B": frozenset(
        {
            "AND",
            "NEG",
            "POW",
            "REDUCEMEAN",
            "RMSNORM",
            "SILU",
            "COS",
            "SIN",
            "SQRT",
            "ARANGE",
            "WHERE",
        }
    ),
}

EXAMPLE_STATUS = {
    "YOLOv2": "planned",
    "YOLOv3-tiny": "experimental",
    "YOLOv4-tiny": "planned",
    "YOLOv5s": "experimental",
    "YOLOv8n": "experimental",
    "ResNet-18": "ready",
    "ResNet-34": "ready",
    "ResNet-50": "ready",
    "ResNet-50 (quantized)": "experimental",
    "Small BERT variants": "ready",
    "Tiny ViT": "ready",
    "MobileNetV1/V2": "experimental",
    "EfficientNet-lite": "ready",
    "Llama 3.1 8B": "experimental",
}


@dataclass(frozen=True, slots=True)
class TranspileReadiness:
    name: str
    missing_ops: tuple[str, ...]
    example_status: str

    @property
    def base_ops_ready(self) -> bool:
        return not self.missing_ops

    @property
    def ready(self) -> bool:
        return self.base_ops_ready and self.example_status == "ready"


def readiness_for(name: str) -> TranspileReadiness:
    required = REQUIRED_EXTRA_OPS.get(name, frozenset())
    blockers = tuple(sorted(required - CURRENT_BACKEND_OPS))
    example_status = EXAMPLE_STATUS.get(name, "planned")
    return TranspileReadiness(name=name, missing_ops=blockers, example_status=example_status)


def readiness_for_all() -> tuple[TranspileReadiness, ...]:
    return tuple(readiness_for(spec.name) for spec in ORDERED_MODEL_SPECS)
