from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FamilySource:
    name: str
    summary: str
    source_url: str
    loader_hint: str


@dataclass(frozen=True, slots=True)
class ModelSourceSpec:
    order: int
    name: str
    family: FamilySource
    upstream_model_id: str
    source_url: str
    loader_hint: str
    notes: str = ""


YOLO_FAMILY = FamilySource(
    name="yolo",
    summary=(
        "Shared detector family. Variants should reuse one YOLO backbone/head "
        "codebase per upstream repo."
    ),
    source_url="https://github.com/ultralytics/ultralytics",
    loader_hint=(
        "Use the upstream repo's model entrypoint/config for each variant; do "
        "not hand-rewrite layers."
    ),
)

RESNET_FAMILY = FamilySource(
    name="resnet",
    summary=(
        "Shared residual backbone family. Variants should come from the same "
        "torchvision ResNet core."
    ),
    source_url="https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py",
    loader_hint="Prefer torchvision ResNet builders from one shared implementation.",
)

BERT_FAMILY = FamilySource(
    name="bert",
    summary=(
        "Small BERT family. Reuse Hugging Face Transformers BERT "
        "implementation for all compact variants."
    ),
    source_url=(
        "https://github.com/huggingface/transformers/blob/main/"
        "src/transformers/models/bert/modeling_bert.py"
    ),
    loader_hint=(
        "Prefer Transformers configs/model classes; vary config/model id, not "
        "the source implementation."
    ),
)

VIT_FAMILY = FamilySource(
    name="vit",
    summary=(
        "Tiny ViT family. Reuse the same ViT implementation and change "
        "patch/width/depth config only."
    ),
    source_url="https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py",
    loader_hint="Prefer timm vision transformer entrypoints.",
)

MOBILENET_FAMILY = FamilySource(
    name="mobilenet",
    summary=(
        "MobileNet family. Reuse one MobileNet implementation per generation "
        "rather than duplicating blocks."
    ),
    source_url="https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py",
    loader_hint=(
        "Prefer torchvision MobileNet builders; V1 can come from a single upstream implementation."
    ),
)

EFFICIENTNET_FAMILY = FamilySource(
    name="efficientnet",
    summary=(
        "EfficientNet-lite family. Keep one EfficientNet-lite implementation "
        "and vary width/depth spec."
    ),
    source_url="https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py",
    loader_hint="Prefer timm EfficientNet-lite entrypoints.",
)

LLAMA_FAMILY = FamilySource(
    name="llama",
    summary=(
        "Llama family. Reuse the shared decoder-only transformer implementation from Transformers."
    ),
    source_url=(
        "https://github.com/huggingface/transformers/blob/main/"
        "src/transformers/models/llama/modeling_llama.py"
    ),
    loader_hint="Prefer Transformers model classes with the correct checkpoint id.",
)


ORDERED_MODEL_SPECS: tuple[ModelSourceSpec, ...] = (
    ModelSourceSpec(
        order=1,
        name="YOLOv2",
        family=YOLO_FAMILY,
        upstream_model_id="yolov2",
        source_url="https://github.com/longcw/yolo2-pytorch",
        loader_hint=(
            "Reference the longcw/yolo2-pytorch implementation for the "
            "original PyTorch YOLOv2 graph."
        ),
        notes="Legacy PyTorch source reference; not in torchvision.",
    ),
    ModelSourceSpec(
        order=2,
        name="YOLOv3-tiny",
        family=YOLO_FAMILY,
        upstream_model_id="yolov3_tiny",
        source_url="https://github.com/ultralytics/yolov3",
        loader_hint=(
            "Use the Ultralytics YOLOv3 repository and select the tiny variant config/weights."
        ),
    ),
    ModelSourceSpec(
        order=3,
        name="YOLOv4-tiny",
        family=YOLO_FAMILY,
        upstream_model_id="yolov4_tiny",
        source_url="https://github.com/WongKinYiu/PyTorch_YOLOv4",
        loader_hint="Use the PyTorch_YOLOv4 repo and the tiny config path.",
    ),
    ModelSourceSpec(
        order=4,
        name="YOLOv5s",
        family=YOLO_FAMILY,
        upstream_model_id="yolov5s",
        source_url="https://github.com/ultralytics/yolov5",
        loader_hint="Use the Ultralytics YOLOv5 shared model code and choose the small variant.",
    ),
    ModelSourceSpec(
        order=5,
        name="YOLOv8n",
        family=YOLO_FAMILY,
        upstream_model_id="yolov8n",
        source_url="https://github.com/ultralytics/ultralytics",
        loader_hint="Use the Ultralytics package/repo and select the nano YOLOv8 model spec.",
    ),
    ModelSourceSpec(
        order=6,
        name="ResNet-18",
        family=RESNET_FAMILY,
        upstream_model_id="resnet18",
        source_url="https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py",
        loader_hint="Use torchvision.models.resnet18 from the shared ResNet implementation.",
    ),
    ModelSourceSpec(
        order=7,
        name="ResNet-34",
        family=RESNET_FAMILY,
        upstream_model_id="resnet34",
        source_url="https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py",
        loader_hint="Use torchvision.models.resnet34 from the shared ResNet implementation.",
    ),
    ModelSourceSpec(
        order=8,
        name="ResNet-50",
        family=RESNET_FAMILY,
        upstream_model_id="resnet50",
        source_url="https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py",
        loader_hint="Use torchvision.models.resnet50 from the shared ResNet implementation.",
    ),
    ModelSourceSpec(
        order=9,
        name="ResNet-50 (quantized)",
        family=RESNET_FAMILY,
        upstream_model_id="quantized_resnet50",
        source_url="https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py",
        loader_hint=(
            "Use torchvision quantized ResNet-50 rather than a separate handwritten ResNet core."
        ),
        notes="Only needed sometimes, per your request.",
    ),
    ModelSourceSpec(
        order=10,
        name="Small BERT variants",
        family=BERT_FAMILY,
        upstream_model_id=(
            "prajjwal1/bert-tiny, prajjwal1/bert-mini, google/bert_uncased_L-4_H-256_A-4"
        ),
        source_url="https://huggingface.co/models?search=bert-tiny",
        loader_hint=(
            "Use the shared Transformers BERT source and swap checkpoint/"
            "config ids for tiny variants."
        ),
    ),
    ModelSourceSpec(
        order=11,
        name="Tiny ViT",
        family=VIT_FAMILY,
        upstream_model_id="vit_tiny_patch16_224",
        source_url="https://github.com/huggingface/pytorch-image-models",
        loader_hint="Use timm's shared VisionTransformer code and the tiny preset.",
    ),
    ModelSourceSpec(
        order=12,
        name="MobileNetV1/V2",
        family=MOBILENET_FAMILY,
        upstream_model_id="mobilenet_v2, mobilenet_v1",
        source_url="https://github.com/pytorch/vision/tree/main/torchvision/models",
        loader_hint=(
            "Use torchvision MobileNetV2 directly and source MobileNetV1 from a single upstream "
            "implementation rather than duplicating depthwise blocks."
        ),
    ),
    ModelSourceSpec(
        order=13,
        name="EfficientNet-lite",
        family=EFFICIENTNET_FAMILY,
        upstream_model_id="efficientnet_lite0",
        source_url="https://github.com/huggingface/pytorch-image-models",
        loader_hint=(
            "Use timm EfficientNet-lite presets from the shared EfficientNet implementation."
        ),
    ),
    ModelSourceSpec(
        order=14,
        name="Llama 3.1 8B",
        family=LLAMA_FAMILY,
        upstream_model_id="meta-llama/Llama-3.1-8B",
        source_url="https://huggingface.co/meta-llama/Llama-3.1-8B",
        loader_hint=(
            "Use the shared Transformers Llama source with the Meta Llama 3.1 8B checkpoint id."
        ),
        notes="Checkpoint access may require acceptance of Meta's license terms.",
    ),
)


def list_model_specs() -> tuple[ModelSourceSpec, ...]:
    return ORDERED_MODEL_SPECS


def get_model_spec(name: str) -> ModelSourceSpec:
    for spec in ORDERED_MODEL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown example model spec: {name}")


def format_catalog_lines() -> list[str]:
    lines: list[str] = []
    for spec in ORDERED_MODEL_SPECS:
        lines.append(f"{spec.order}. {spec.name} [{spec.family.name}]")
        lines.append(f"   source: {spec.source_url}")
        lines.append(f"   loader: {spec.loader_hint}")
        if spec.notes:
            lines.append(f"   notes: {spec.notes}")
    return lines


def family_groups() -> dict[str, tuple[ModelSourceSpec, ...]]:
    grouped: dict[str, list[ModelSourceSpec]] = {}
    for spec in ORDERED_MODEL_SPECS:
        grouped.setdefault(spec.family.name, []).append(spec)
    return {key: tuple(value) for key, value in grouped.items()}


def _optional_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance only
        raise ModuleNotFoundError(
            f"Optional dependency {name!r} is required for this loader. {exc}"
        ) from exc


def build_example_reference(name: str) -> object:
    spec = get_model_spec(name)
    if spec.family.name == "resnet":
        torchvision_models = _optional_import("torchvision.models")
        if spec.upstream_model_id == "resnet18":
            return torchvision_models.resnet18(weights=None)
        if spec.upstream_model_id == "resnet34":
            return torchvision_models.resnet34(weights=None)
        if spec.upstream_model_id == "quantized_resnet50":
            quant_models = _optional_import("torchvision.models.quantization")
            return quant_models.resnet50(weights=None, quantize=True)
    if spec.family.name == "mobilenet":
        torchvision_models = _optional_import("torchvision.models")
        if spec.upstream_model_id.startswith("mobilenet_v2"):
            return torchvision_models.mobilenet_v2(weights=None)
    return spec
