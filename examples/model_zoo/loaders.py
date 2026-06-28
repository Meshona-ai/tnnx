from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .catalog import get_model_spec


@dataclass(frozen=True, slots=True)
class LoaderPlan:
    model_name: str
    source_url: str
    loader_hint: str
    install_hint: str
    smoke_ready: bool
    notes: str = ""


@dataclass(frozen=True, slots=True)
class LoadedFamilyGroup:
    model_name: str
    variants: tuple[str, ...]
    instances: dict[str, Any]


def _plan(model_name: str, *, install_hint: str, smoke_ready: bool, notes: str = "") -> LoaderPlan:
    spec = get_model_spec(model_name)
    return LoaderPlan(
        model_name=model_name,
        source_url=spec.source_url,
        loader_hint=spec.loader_hint,
        install_hint=install_hint,
        smoke_ready=smoke_ready,
        notes=notes or spec.notes,
    )


def _optional_import(name: str) -> Any | None:
    root = name.split(".", 1)[0]
    try:
        return __import__(name, fromlist=["*"])
    except ModuleNotFoundError as exc:
        if exc.name in {root, name}:
            return None
        raise


def load_yolo_variant(name: str) -> Any:
    if name == "YOLOv4-tiny":
        return _plan(
            name,
            install_hint=(
                "Use the upstream PyTorch YOLOv4 repository locally; this variant is not "
                "shipped by the Ultralytics package."
            ),
            smoke_ready=False,
            notes=(
                "The shared YOLO wrapper only covers variants that have a real local model "
                "definition in Ultralytics."
            ),
        )

    ultralytics = _optional_import("ultralytics")
    if ultralytics is None:
        return _plan(
            name,
            install_hint=(
                "Install the 'ultralytics' package to instantiate this upstream YOLO variant."
            ),
            smoke_ready=False,
            notes=(
                "The current compiler is also missing key YOLO ops such as CONCAT/UPSAMPLE/SIGMOID."
            ),
        )

    model_file = {
        "YOLOv3-tiny": "yolov3-tiny.yaml",
        "YOLOv5s": "yolov5s.yaml",
        "YOLOv8n": "yolov8n.yaml",
    }.get(name)
    if model_file is None:
        return _plan(
            name,
            install_hint=(
                "Use the named upstream repo directly for this legacy YOLO implementation."
            ),
            smoke_ready=False,
            notes=(
                "YOLOv2 is intentionally left as a source reference because it "
                "is not in Ultralytics."
            ),
        )
    return ultralytics.YOLO(model_file)


def load_resnet_variant(name: str) -> Any:
    torchvision_models = _optional_import("torchvision.models")
    if torchvision_models is None:
        return _plan(
            name,
            install_hint=(
                "Install 'torchvision' to instantiate the shared torchvision ResNet implementation."
            ),
            smoke_ready=True,
        )

    if name == "ResNet-18":
        return torchvision_models.resnet18(weights=None)
    if name == "ResNet-34":
        return torchvision_models.resnet34(weights=None)
    if name == "ResNet-50":
        return torchvision_models.resnet50(weights=None)
    if name == "ResNet-50 (quantized)":
        quant_models = _optional_import("torchvision.models.quantization")
        if quant_models is None:
            return _plan(
                name,
                install_hint=(
                    "Install a torchvision build with quantization support to instantiate "
                    "quantized ResNet-50."
                ),
                smoke_ready=True,
            )
        return quant_models.resnet50(weights=None, quantize=True)
    raise KeyError(f"Unsupported ResNet example: {name}")


def load_small_bert_variants() -> Any:
    bert_config_mod = _optional_import("transformers.models.bert.configuration_bert")
    bert_model_mod = _optional_import("transformers.models.bert.modeling_bert")
    if bert_config_mod is None or bert_model_mod is None:
        return _plan(
            "Small BERT variants",
            install_hint=(
                "Install 'transformers' to instantiate BERT variants from shared source code."
            ),
            smoke_ready=True,
        )

    BertConfig = bert_config_mod.BertConfig
    BertModel = bert_model_mod.BertModel
    variants = {
        "bert-tiny": BertModel(
            BertConfig(
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=512,
                vocab_size=30522,
            )
        ),
        "bert-mini": BertModel(
            BertConfig(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                vocab_size=30522,
            )
        ),
        "bert-small-L4-H256-A4": BertModel(
            BertConfig(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                vocab_size=30522,
            )
        ),
    }
    return LoadedFamilyGroup(
        model_name="Small BERT variants",
        variants=tuple(variants.keys()),
        instances=variants,
    )


def load_tiny_vit() -> Any:
    vit_config_mod = _optional_import("transformers.models.vit.configuration_vit")
    vit_model_mod = _optional_import("transformers.models.vit.modeling_vit")
    if vit_config_mod is None or vit_model_mod is None:
        return _plan(
            "Tiny ViT",
            install_hint="Install 'transformers' to instantiate a ViT tiny smoke config.",
            smoke_ready=True,
        )

    return vit_model_mod.ViTModel(
        vit_config_mod.ViTConfig(
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
            image_size=224,
            patch_size=16,
        )
    )


def load_mobilenet_variants() -> Any:
    torchvision_models = _optional_import("torchvision.models")
    if torchvision_models is None:
        return _plan(
            "MobileNetV1/V2",
            install_hint=(
                "Install 'torchvision' (and optionally 'timm') for MobileNet family examples."
            ),
            smoke_ready=True,
            notes=(
                "This wrapper instantiates MobileNetV2 and keeps MobileNetV1 as a source reference."
            ),
        )

    variants = {
        "mobilenet_v2": torchvision_models.mobilenet_v2(weights=None),
    }
    return LoadedFamilyGroup(
        model_name="MobileNetV1/V2",
        variants=tuple(variants.keys()),
        instances=variants,
    )


def load_efficientnet_lite() -> Any:
    timm = _optional_import("timm")
    if timm is None:
        return _plan(
            "EfficientNet-lite",
            install_hint=(
                "Install 'timm' to instantiate EfficientNet-lite from the "
                "shared EfficientNet implementation."
            ),
            smoke_ready=True,
        )
    return timm.create_model("efficientnet_lite0", pretrained=False)


def load_llama_3_1_8b(*, smoke: bool = True) -> Any:
    llama_config_mod = _optional_import("transformers.models.llama.configuration_llama")
    llama_model_mod = _optional_import("transformers.models.llama.modeling_llama")
    if llama_config_mod is None or llama_model_mod is None:
        return _plan(
            "Llama 3.1 8B",
            install_hint="Install 'transformers' to instantiate the shared Llama implementation.",
            smoke_ready=smoke,
            notes=(
                "Use a tiny smoke config by default; loading the actual Meta checkpoint requires "
                "access and substantial memory."
            ),
        )

    if smoke:
        return llama_model_mod.LlamaForCausalLM(
            llama_config_mod.LlamaConfig(
                hidden_size=128,
                intermediate_size=512,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                vocab_size=32000,
                max_position_embeddings=256,
            )
        )

    return _plan(
        "Llama 3.1 8B",
        install_hint="Load the actual checkpoint via transformers from 'meta-llama/Llama-3.1-8B'.",
        smoke_ready=False,
        notes="The full 8B checkpoint is intentionally not instantiated in the example workspace.",
    )


def build_reference(name: str) -> Any:
    if name.startswith("YOLO"):
        return load_yolo_variant(name)
    if name.startswith("ResNet"):
        return load_resnet_variant(name)
    if name == "Small BERT variants":
        return load_small_bert_variants()
    if name == "Tiny ViT":
        return load_tiny_vit()
    if name == "MobileNetV1/V2":
        return load_mobilenet_variants()
    if name == "EfficientNet-lite":
        return load_efficientnet_lite()
    if name == "Llama 3.1 8B":
        return load_llama_3_1_8b()
    raise KeyError(f"Unknown model-zoo entry: {name}")
