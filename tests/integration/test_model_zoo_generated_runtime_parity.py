from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_runtime(target: str) -> Any:
    torch = pytest.importorskip("torch")
    if target == "jax":
        pytest.importorskip("jax")
    elif target == "mlx":
        pytest.importorskip("mlx.core")
    else:
        raise ValueError(f"Unsupported target: {target}")
    return torch


def _smoke_module() -> Any:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from examples.model_zoo import smoke

    return smoke


def _select_job(
    *,
    target: str,
    model_name: str,
    variant_name: str,
    out_root: Path,
) -> Any:
    jobs = _smoke_module().list_smoke_jobs(target=target, out_root=out_root)
    for job in jobs:
        if job.model_name == model_name and job.variant_name == variant_name:
            return job
    raise AssertionError(f"Smoke job not found for {target}/{model_name}/{variant_name}")


def _reference_output(
    prepared: Any,
    torch: Any,
) -> tuple[
    dict[str, np.ndarray[Any, Any]],
    str,
    np.ndarray[Any, Any],
]:
    sample_args = tuple(prepared.sample_args)
    inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(prepared.input_names, sample_args, strict=True)
    }
    with torch.no_grad():
        expected = prepared.model(*sample_args)
    if isinstance(expected, list | tuple):
        expected = expected[0]
    output_name = prepared.output_names[0]
    return inputs, output_name, expected.detach().cpu().numpy()


@pytest.mark.parametrize(
    ("target", "model_name", "variant_name"),
    [
        ("jax", "ResNet-18", "resnet18"),
        ("jax", "YOLOv8n", "yolov8n"),
        ("jax", "Small BERT variants", "bert-tiny"),
        ("jax", "Llama 3.1 8B", "llama_3_1_8b_smoke"),
        ("mlx", "ResNet-18", "resnet18"),
        ("mlx", "Small BERT variants", "bert-tiny"),
    ],
)
def test_model_zoo_generated_runtime_parity(
    target: str,
    model_name: str,
    variant_name: str,
    tmp_path: Path,
) -> None:
    torch = _require_runtime(target)
    smoke = _smoke_module()

    job = _select_job(
        target=target,
        model_name=model_name,
        variant_name=variant_name,
        out_root=tmp_path / "model_zoo_runtime",
    )
    prepared = smoke.prepare_job(job)
    if isinstance(prepared, smoke.LoaderPlan):
        pytest.skip(prepared.install_hint)

    inputs, output_name, expected = _reference_output(prepared, torch)
    result = smoke.run_prepared_job(job, prepared)
    assert result.status == "transpiled", result.message

    module_path = job.out_dir / f"model_{target}.py"
    module = _load_module(module_path, f"generated_{target}_{job.slug}")
    params = module.load_weights(str(job.out_dir / "weights.npz"))
    actual = np.asarray(module.forward(params, inputs)[output_name])

    atol = 1e-4 if target == "jax" else 2e-4
    rtol = 1e-4 if target == "jax" else 2e-4
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Runtime parity failed for {target}/{model_name}/{variant_name}: "
        f"max_abs={np.max(np.abs(actual - expected))}"
    )
