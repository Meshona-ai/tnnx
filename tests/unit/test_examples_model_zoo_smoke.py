from __future__ import annotations

import sys
from pathlib import Path

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


def _model_zoo():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.model_zoo as model_zoo

    return model_zoo


def test_smoke_jobs_preserve_requested_model_order() -> None:
    model_zoo = _model_zoo()
    jobs = model_zoo.list_smoke_jobs(target="jax")

    seen: list[str] = []
    for job in jobs:
        if not seen or seen[-1] != job.model_name:
            seen.append(job.model_name)

    assert seen == EXPECTED_ORDER
    assert len({job.slug for job in jobs}) == len(jobs)


def test_smoke_jobs_expand_shared_family_variants_cleanly() -> None:
    model_zoo = _model_zoo()
    jobs = model_zoo.list_smoke_jobs(target="mlx", out_root="examples/out/model_zoo")

    bert = [job.variant_name for job in jobs if job.model_name == "Small BERT variants"]
    assert bert == ["bert-tiny", "bert-mini", "bert-small-L4-H256-A4"]

    mobilenet = [job.variant_name for job in jobs if job.model_name == "MobileNetV1/V2"]
    assert mobilenet == ["mobilenet_v1", "mobilenet_v2"]

    assert all("generated_mlx_" in str(job.out_dir) for job in jobs)


def test_smoke_job_lines_are_human_readable() -> None:
    model_zoo = _model_zoo()
    lines = model_zoo.format_smoke_job_lines(target="jax", out_root="/tmp/model_zoo_examples")

    assert any("YOLOv2 :: yolov2 :: planned" in line for line in lines)
    assert any("YOLOv4-tiny :: yolov4_tiny :: planned" in line for line in lines)
    assert any("ResNet-18 :: resnet18 :: ready" in line for line in lines)
    assert any(
        "/tmp/model_zoo_examples/generated_jax_resnet_18__resnet18" in line for line in lines
    )
