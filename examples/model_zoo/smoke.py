from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig

from .loaders import LoadedFamilyGroup, LoaderPlan, build_reference

JobStatus = Literal["ready", "experimental", "planned"]
RunStatus = Literal["transpiled", "planned", "blocked", "failed"]

_OUT_ROOT = Path("examples/out/model_zoo")


@dataclass(frozen=True, slots=True)
class SmokeTranspileJob:
    order: int
    model_name: str
    variant_name: str
    slug: str
    status: JobStatus
    target: str
    onnx_path: Path
    out_dir: Path
    notes: str = ""


@dataclass(frozen=True, slots=True)
class SmokeRunResult:
    job: SmokeTranspileJob
    status: RunStatus
    message: str


@dataclass(slots=True)
class _PreparedExport:
    model: Any
    sample_args: tuple[Any, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


def _optional_import(name: str) -> Any | None:
    try:
        return __import__(name, fromlist=["*"])
    except ModuleNotFoundError:
        return None


def _slug(value: str) -> str:
    out = []
    last_was_sep = False
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
            last_was_sep = False
            continue
        if not last_was_sep:
            out.append("_")
            last_was_sep = True
    return "".join(out).strip("_")


def _job(
    order: int,
    model_name: str,
    variant_name: str,
    status: JobStatus,
    target: str,
    *,
    notes: str = "",
    out_root: Path = _OUT_ROOT,
) -> SmokeTranspileJob:
    base = _slug(model_name)
    variant = _slug(variant_name)
    slug = base if base == variant else f"{base}__{variant}"
    return SmokeTranspileJob(
        order=order,
        model_name=model_name,
        variant_name=variant_name,
        slug=slug,
        status=status,
        target=target,
        onnx_path=out_root / f"{slug}.onnx",
        out_dir=out_root / f"generated_{target}_{slug}",
        notes=notes,
    )


def list_smoke_jobs(
    target: str = "jax", *, out_root: str | Path = _OUT_ROOT
) -> tuple[SmokeTranspileJob, ...]:
    root = Path(out_root)
    jobs = (
        _job(
            1,
            "YOLOv2",
            "yolov2",
            "planned",
            target,
            notes="Legacy upstream repo only.",
            out_root=root,
        ),
        _job(
            2,
            "YOLOv3-tiny",
            "yolov3_tiny",
            "experimental",
            target,
            notes="Uses Ultralytics if installed; export surface is not validated here.",
            out_root=root,
        ),
        _job(
            3,
            "YOLOv4-tiny",
            "yolov4_tiny",
            "planned",
            target,
            notes="Needs the upstream YOLOv4 repo; this variant is not shipped by Ultralytics.",
            out_root=root,
        ),
        _job(
            4,
            "YOLOv5s",
            "yolov5s",
            "experimental",
            target,
            notes="Uses Ultralytics if installed; export surface is not validated here.",
            out_root=root,
        ),
        _job(
            5,
            "YOLOv8n",
            "yolov8n",
            "experimental",
            target,
            notes="Uses Ultralytics if installed; export surface is not validated here.",
            out_root=root,
        ),
        _job(6, "ResNet-18", "resnet18", "ready", target, out_root=root),
        _job(7, "ResNet-34", "resnet34", "ready", target, out_root=root),
        _job(8, "ResNet-50", "resnet50", "ready", target, out_root=root),
        _job(
            9,
            "ResNet-50 (quantized)",
            "quantized_resnet50",
            "experimental",
            target,
            notes=(
                "Quantized ONNX export is often brittle; treat as a probe, not a guaranteed path."
            ),
            out_root=root,
        ),
        _job(10, "Small BERT variants", "bert-tiny", "ready", target, out_root=root),
        _job(10, "Small BERT variants", "bert-mini", "ready", target, out_root=root),
        _job(10, "Small BERT variants", "bert-small-L4-H256-A4", "ready", target, out_root=root),
        _job(11, "Tiny ViT", "vit_tiny_patch16_224", "ready", target, out_root=root),
        _job(
            12,
            "MobileNetV1/V2",
            "mobilenet_v1",
            "experimental",
            target,
            notes="Uses timm if installed; model id may vary by timm release.",
            out_root=root,
        ),
        _job(12, "MobileNetV1/V2", "mobilenet_v2", "ready", target, out_root=root),
        _job(13, "EfficientNet-lite", "efficientnet_lite0", "ready", target, out_root=root),
        _job(
            14,
            "Llama 3.1 8B",
            "llama_3_1_8b_smoke",
            "experimental",
            target,
            notes="Runs the local tiny smoke config, not the real 8B checkpoint.",
            out_root=root,
        ),
    )
    return jobs


def format_smoke_job_lines(
    target: str = "jax",
    *,
    out_root: str | Path = _OUT_ROOT,
) -> tuple[str, ...]:
    lines: list[str] = []
    for job in list_smoke_jobs(target, out_root=out_root):
        lines.append(f"[{job.order:02d}] {job.model_name} :: {job.variant_name} :: {job.status}")
        if job.notes:
            lines.append(f"     {job.notes}")
        lines.append(f"     onnx={job.onnx_path} -> out={job.out_dir}")
    return tuple(lines)


def _require_torch() -> tuple[Any, Any]:
    torch = _optional_import("torch")
    if torch is None:
        raise RuntimeError("Install 'torch' to export model-zoo smoke examples.")
    nn = _optional_import("torch.nn")
    if nn is None:
        raise RuntimeError("Install 'torch' to export model-zoo smoke examples.")
    return torch, nn


def _export_to_onnx(job: SmokeTranspileJob, prepared: _PreparedExport) -> Path:
    torch, _ = _require_torch()
    job.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    args: Any
    if len(prepared.sample_args) == 1:
        args = prepared.sample_args[0]
    else:
        args = prepared.sample_args
    torch.onnx.export(
        prepared.model,
        args,
        job.onnx_path,
        input_names=list(prepared.input_names),
        output_names=list(prepared.output_names),
        opset_version=18,
        dynamo=False,
    )
    return job.onnx_path


def _eval_model(model: Any) -> Any:
    if hasattr(model, "eval"):
        return model.eval()
    return model


def _prepare_image_classifier(model: Any) -> _PreparedExport:
    torch, _ = _require_torch()
    return _PreparedExport(
        model=_eval_model(model),
        sample_args=(torch.randn(1, 3, 224, 224),),
        input_names=("x",),
        output_names=("y",),
    )


def _prepare_resnet(job: SmokeTranspileJob) -> _PreparedExport | LoaderPlan:
    model = build_reference(job.model_name)
    if isinstance(model, LoaderPlan):
        return model
    return _prepare_image_classifier(model)


def _prepare_quantized_resnet(job: SmokeTranspileJob) -> _PreparedExport | LoaderPlan:
    model = build_reference(job.model_name)
    if isinstance(model, LoaderPlan):
        return model
    return _prepare_image_classifier(model)


def _prepare_bert(job: SmokeTranspileJob) -> _PreparedExport | LoaderPlan:
    torch, nn = _require_torch()
    group = build_reference("Small BERT variants")
    if isinstance(group, LoaderPlan):
        return group
    if not isinstance(group, LoadedFamilyGroup):
        raise TypeError("Expected LoadedFamilyGroup for Small BERT variants.")
    model = group.instances[job.variant_name]

    class BertWrapper(nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: Any) -> Any:
            return self.wrapped(input_ids=input_ids).last_hidden_state

    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 30522))
    return _PreparedExport(
        model=_eval_model(BertWrapper(model)),
        sample_args=(torch.randint(0, vocab, (1, 16), dtype=torch.long),),
        input_names=("input_ids",),
        output_names=("hidden",),
    )


def _prepare_vit() -> _PreparedExport | LoaderPlan:
    torch, nn = _require_torch()
    model = build_reference("Tiny ViT")
    if isinstance(model, LoaderPlan):
        return model

    class ViTWrapper(nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: Any) -> Any:
            return self.wrapped(pixel_values=x).last_hidden_state

    return _PreparedExport(
        model=_eval_model(ViTWrapper(model)),
        sample_args=(torch.randn(1, 3, 224, 224),),
        input_names=("x",),
        output_names=("hidden",),
    )


def _prepare_mobilenet(job: SmokeTranspileJob) -> _PreparedExport | LoaderPlan:
    if job.variant_name == "mobilenet_v1":
        timm = _optional_import("timm")
        if timm is None:
            return LoaderPlan(
                model_name=job.model_name,
                source_url="https://github.com/huggingface/pytorch-image-models",
                loader_hint="Use timm.create_model('mobilenetv1_100', pretrained=False).",
                install_hint="Install 'timm' to export the MobileNetV1 smoke example.",
                smoke_ready=True,
                notes=job.notes,
            )
        model = timm.create_model("mobilenetv1_100", pretrained=False)
        return _prepare_image_classifier(model)

    group = build_reference("MobileNetV1/V2")
    if isinstance(group, LoaderPlan):
        return group
    if not isinstance(group, LoadedFamilyGroup):
        raise TypeError("Expected LoadedFamilyGroup for MobileNetV1/V2.")
    model = group.instances[job.variant_name]
    return _prepare_image_classifier(model)


def _prepare_efficientnet() -> _PreparedExport | LoaderPlan:
    model = build_reference("EfficientNet-lite")
    if isinstance(model, LoaderPlan):
        return model
    return _prepare_image_classifier(model)


def _prepare_llama() -> _PreparedExport | LoaderPlan:
    torch, nn = _require_torch()
    model = build_reference("Llama 3.1 8B")
    if isinstance(model, LoaderPlan):
        return model
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"

    class LlamaWrapper(nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits

    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 32000))
    return _PreparedExport(
        model=_eval_model(LlamaWrapper(model)),
        sample_args=(
            torch.randint(0, vocab, (1, 16), dtype=torch.long),
            torch.ones((1, 16), dtype=torch.long),
        ),
        input_names=("input_ids", "attention_mask"),
        output_names=("logits",),
    )


def _prepare_yolo(job: SmokeTranspileJob) -> _PreparedExport | LoaderPlan:
    if job.model_name == "YOLOv2":
        return LoaderPlan(
            model_name=job.model_name,
            source_url="https://github.com/longcw/yolo2-pytorch",
            loader_hint="Export directly from the legacy YOLOv2 repo checkout.",
            install_hint="Use the upstream YOLOv2 repository locally for this example.",
            smoke_ready=False,
            notes=job.notes,
        )

    torch, nn = _require_torch()
    model = build_reference(job.model_name)
    if isinstance(model, LoaderPlan):
        return model
    core = getattr(model, "model", None)
    if core is None:
        return LoaderPlan(
            model_name=job.model_name,
            source_url="https://github.com/ultralytics/ultralytics",
            loader_hint="Use the inner .model torch module from Ultralytics.",
            install_hint="Install 'ultralytics' and verify the selected YAML variant is supported.",
            smoke_ready=False,
            notes="The local wrapper could not find an exportable torch module.",
        )

    class YOLOWrapper(nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: Any) -> Any:
            out = self.wrapped(x)
            if isinstance(out, list | tuple):
                first = out[0]
            else:
                first = out
            if isinstance(first, list | tuple):
                return first[0]
            return first

    return _PreparedExport(
        model=_eval_model(YOLOWrapper(core)),
        sample_args=(torch.randn(1, 3, 640, 640),),
        input_names=("x",),
        output_names=("y",),
    )


def prepare_job(job: SmokeTranspileJob) -> _PreparedExport | LoaderPlan:
    if job.model_name.startswith("YOLO"):
        return _prepare_yolo(job)
    if job.model_name in {"ResNet-18", "ResNet-34", "ResNet-50"}:
        return _prepare_resnet(job)
    if job.model_name == "ResNet-50 (quantized)":
        return _prepare_quantized_resnet(job)
    if job.model_name == "Small BERT variants":
        return _prepare_bert(job)
    if job.model_name == "Tiny ViT":
        return _prepare_vit()
    if job.model_name == "MobileNetV1/V2":
        return _prepare_mobilenet(job)
    if job.model_name == "EfficientNet-lite":
        return _prepare_efficientnet()
    if job.model_name == "Llama 3.1 8B":
        return _prepare_llama()
    raise KeyError(f"Unknown smoke job: {job.model_name}")


def run_job(job: SmokeTranspileJob, *, entrypoint: str = "forward") -> SmokeRunResult:
    if job.status == "planned":
        return SmokeRunResult(job=job, status="planned", message=job.notes or "Planned only.")

    try:
        prepared = prepare_job(job)
    except Exception as exc:
        return SmokeRunResult(job=job, status="failed", message=str(exc))

    return run_prepared_job(job, prepared, entrypoint=entrypoint)


def run_prepared_job(
    job: SmokeTranspileJob,
    prepared: _PreparedExport | LoaderPlan,
    *,
    entrypoint: str = "forward",
) -> SmokeRunResult:
    if job.status == "planned":
        return SmokeRunResult(job=job, status="planned", message=job.notes or "Planned only.")

    if isinstance(prepared, LoaderPlan):
        return SmokeRunResult(job=job, status="blocked", message=prepared.install_hint)

    try:
        _export_to_onnx(job, prepared)
        transpile_onnx(
            str(job.onnx_path),
            job.target,
            str(job.out_dir),
            config=CompileConfig(entrypoint=entrypoint),
        )
    except Exception as exc:
        return SmokeRunResult(job=job, status="failed", message=str(exc))
    return SmokeRunResult(
        job=job, status="transpiled", message="ONNX export + transpile succeeded."
    )


def run_jobs(
    *,
    target: str = "jax",
    model_name: str | None = None,
    include_experimental: bool = False,
    out_root: str | Path = _OUT_ROOT,
) -> tuple[SmokeRunResult, ...]:
    jobs = list_smoke_jobs(target=target, out_root=out_root)
    out: list[SmokeRunResult] = []
    for job in jobs:
        if model_name is not None and job.model_name != model_name:
            continue
        if job.status == "experimental" and not include_experimental:
            out.append(
                SmokeRunResult(
                    job=job,
                    status="planned",
                    message="Skipped experimental job; rerun with --include-experimental.",
                )
            )
            continue
        out.append(run_job(job))
    return tuple(out)
