from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPECTED_ORDER = [
    "YOLOv2",
    "YOLOv3-tiny",
    "YOLOv4-tiny",
    "YOLOv5s",
    "YOLOv8n",
    "ResNet-18",
    "ResNet-34",
    "ResNet-50",
    "ResNet-50 (quantized)",
    "Small BERT variants",
    "Tiny ViT",
    "MobileNetV1/V2",
    "EfficientNet-lite",
    "Llama 3.1 8B",
]


def _catalog_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.model_zoo as model_zoo

    return model_zoo


def _loaders_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.model_zoo.loaders as loaders

    return loaders


def test_model_zoo_catalog_preserves_requested_order() -> None:
    catalog = _catalog_module()
    assert [spec.name for spec in catalog.ORDERED_MODEL_SPECS] == EXPECTED_ORDER
    assert [spec.order for spec in catalog.ORDERED_MODEL_SPECS] == list(range(1, 15))


def test_model_zoo_uses_shared_family_sources() -> None:
    catalog = _catalog_module()
    grouped = catalog.family_groups()

    assert [spec.name for spec in grouped["yolo"]] == EXPECTED_ORDER[:5]
    assert [spec.name for spec in grouped["resnet"]] == EXPECTED_ORDER[5:9]
    assert [spec.name for spec in grouped["bert"]] == ["Small BERT variants"]
    assert [spec.name for spec in grouped["vit"]] == ["Tiny ViT"]
    assert [spec.name for spec in grouped["mobilenet"]] == ["MobileNetV1/V2"]
    assert [spec.name for spec in grouped["efficientnet"]] == ["EfficientNet-lite"]
    assert [spec.name for spec in grouped["llama"]] == ["Llama 3.1 8B"]


def test_model_zoo_default_reference_falls_back_to_metadata_for_heavy_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_zoo = _catalog_module()
    loaders = _loaders_module()
    original = loaders._optional_import

    def _no_transformers(name: str):
        if name.startswith("transformers"):
            return None
        return original(name)

    monkeypatch.setattr(loaders, "_optional_import", _no_transformers)
    llama = model_zoo.build_reference("Llama 3.1 8B")
    assert isinstance(llama, model_zoo.LoaderPlan)
    assert llama.model_name == "Llama 3.1 8B"
    assert llama.smoke_ready is True


def test_model_zoo_loader_plans_do_not_crash_without_optional_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_zoo = _catalog_module()
    loaders = _loaders_module()

    def _missing_optional(_name: str):
        return None

    monkeypatch.setattr(loaders, "_optional_import", _missing_optional)

    yolo = model_zoo.build_reference("YOLOv8n")
    assert isinstance(yolo, model_zoo.LoaderPlan)
    assert "ultralytics" in yolo.install_hint

    bert = model_zoo.build_reference("Small BERT variants")
    assert isinstance(bert, model_zoo.LoaderPlan)
    assert "transformers" in bert.install_hint


def test_model_zoo_readiness_reflects_base_ops_and_example_coverage() -> None:
    model_zoo = _catalog_module()

    yolo = model_zoo.readiness_for("YOLOv8n")
    assert yolo.base_ops_ready is True
    assert yolo.example_status == "experimental"
    assert yolo.ready is False
    assert yolo.missing_ops == ()

    yolo4 = model_zoo.readiness_for("YOLOv4-tiny")
    assert yolo4.base_ops_ready is True
    assert yolo4.example_status == "planned"
    assert yolo4.ready is False

    resnet = model_zoo.readiness_for("ResNet-18")
    assert resnet.base_ops_ready is True
    assert resnet.example_status == "ready"
    assert resnet.ready is True
    assert resnet.missing_ops == ()

    llama = model_zoo.readiness_for("Llama 3.1 8B")
    assert llama.base_ops_ready is True
    assert llama.example_status == "experimental"
    assert llama.ready is False
    assert llama.missing_ops == ()

    mobilenet = model_zoo.readiness_for("MobileNetV1/V2")
    assert mobilenet.base_ops_ready is True
    assert mobilenet.example_status == "experimental"
    assert mobilenet.ready is False

    efficientnet = model_zoo.readiness_for("EfficientNet-lite")
    assert efficientnet.base_ops_ready is True
    assert efficientnet.example_status == "ready"
    assert efficientnet.ready is True
