from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.model_zoo import smoke  # noqa: E402
from tnnx.api import transpile_onnx  # noqa: E402
from tnnx.config import CompileConfig  # noqa: E402

DEFAULT_MODELS = (
    "ResNet-18::resnet18",
    "ResNet-34::resnet34",
    "ResNet-50::resnet50",
    "MLP smoke::mlp_128x1024",
    "Small BERT variants::bert-tiny",
    "Whisper tiny::whisper_tiny",
    "Llama 3.1 8B::llama_3_1_8b_smoke",
)
DEFAULT_TARGETS = ("jax", "mlx")
_LOCAL_PATH_RE = re.compile(r"(/Users|/home)/[^\s)`>,]+")


def _now_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _optional_import(name: str) -> Any | None:
    try:
        return __import__(name, fromlist=["*"])
    except ModuleNotFoundError:
        return None


def _sanitize_text(value: str) -> str:
    sanitized = _LOCAL_PATH_RE.sub("[local-path]", value)
    sanitized = " ".join(sanitized.split())
    if len(sanitized) > 280:
        return sanitized[:277] + "..."
    return sanitized


def _parse_model(value: str) -> tuple[str, str]:
    if "::" not in value:
        raise argparse.ArgumentTypeError(
            f"Model selection must use 'model name::variant name', got {value!r}."
        )
    model_name, variant_name = value.split("::", 1)
    if not model_name or not variant_name:
        raise argparse.ArgumentTypeError(f"Invalid model selection: {value!r}.")
    return model_name, variant_name


def _select_job(
    *,
    target: str,
    model_name: str,
    variant_name: str,
    out_root: Path,
) -> smoke.SmokeTranspileJob:
    if model_name == "MLP smoke" and variant_name == "mlp_128x1024":
        slug = "mlp_smoke__mlp_128x1024"
        return smoke.SmokeTranspileJob(
            order=15,
            model_name=model_name,
            variant_name=variant_name,
            slug=slug,
            status="ready",
            target=target,
            onnx_path=out_root / f"{slug}.onnx",
            out_dir=out_root / f"generated_{target}_{slug}",
            notes="Batched GEMM/GELU smoke benchmark used for paper evidence.",
        )
    if model_name == "Whisper tiny" and variant_name == "whisper_tiny":
        slug = "whisper_tiny"
        return smoke.SmokeTranspileJob(
            order=16,
            model_name=model_name,
            variant_name=variant_name,
            slug=slug,
            status="ready",
            target=target,
            onnx_path=out_root / f"{slug}.onnx",
            out_dir=out_root / f"generated_{target}_{slug}",
            notes=(
                "Tiny Whisper-style encoder/decoder benchmark from examples/model_whisper_tiny.py."
            ),
        )
    for job in smoke.list_smoke_jobs(target=target, out_root=out_root):
        if job.model_name == model_name and job.variant_name == variant_name:
            return job
    raise KeyError(f"No model-zoo smoke job for {target}/{model_name}/{variant_name}.")


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_npz_contents(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(data.files):
            value = np.ascontiguousarray(data[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(value.tobytes())
    return digest.hexdigest()


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _artifact_paths(target: str, out_dir: Path) -> dict[str, Path]:
    paths = {
        "graph_ir": out_dir / "graph_ir.json",
        "weights": out_dir / "weights.npz",
    }
    if target == "jax":
        paths["source"] = out_dir / "model_jax.py"
    elif target == "mlx":
        paths["source"] = out_dir / "model_mlx.py"
    else:
        raise ValueError(f"Unsupported target: {target}")
    return paths


def _artifact_stats(target: str, out_dir: Path) -> dict[str, Any]:
    paths = _artifact_paths(target, out_dir)
    source_path = paths["source"]
    total_size = sum(_file_size(path) for path in paths.values())
    loc = _count_lines(source_path)
    return {
        "artifact_bytes": total_size,
        "generated_loc": loc,
        "graph_ir_sha256": _sha256_file(paths["graph_ir"]),
        "source_sha256": _sha256_file(source_path),
        "weights_npz_sha256": _sha256_file(paths["weights"]),
        "weights_content_sha256": _sha256_npz_contents(paths["weights"]),
    }


def _determinism(
    *,
    target: str,
    onnx_path: Path,
    first_dir: Path,
    repeat_dir: Path,
) -> dict[str, Any]:
    transpile_onnx(str(onnx_path), target, str(repeat_dir), config=CompileConfig())
    first = _artifact_stats(target, first_dir)
    repeat = _artifact_stats(target, repeat_dir)
    graph_stable = first["graph_ir_sha256"] == repeat["graph_ir_sha256"]
    source_stable = first["source_sha256"] == repeat["source_sha256"]
    weights_byte_stable = first["weights_npz_sha256"] == repeat["weights_npz_sha256"]
    weights_semantic_stable = first["weights_content_sha256"] == repeat["weights_content_sha256"]
    return {
        "graph_ir_byte_stable": graph_stable,
        "source_byte_stable": source_stable,
        "weights_npz_byte_stable": weights_byte_stable,
        "weights_content_stable": weights_semantic_stable,
        "all_byte_stable": graph_stable and source_stable and weights_byte_stable,
        "semantic_stable": graph_stable and source_stable and weights_semantic_stable,
    }


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated module at {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _torch_output_to_numpy(value: Any) -> np.ndarray[Any, Any]:
    if isinstance(value, list | tuple):
        value = value[0]
    return value.detach().cpu().numpy()


def _torch_synchronize(torch: Any, device: str) -> None:
    if device == "mps" and getattr(torch.backends, "mps", None) is not None:
        torch.mps.synchronize()


def _torch_device_available(torch: Any, device: str) -> bool:
    if device == "cpu":
        return True
    if device == "mps":
        return bool(torch.backends.mps.is_available())
    return False


def _torch_args_to_device(args: tuple[Any, ...], device: str) -> tuple[Any, ...]:
    return tuple(arg.to(device) if hasattr(arg, "to") else arg for arg in args)


def _clone_torch_model(model: Any) -> Any:
    cloned = copy.deepcopy(model)
    if hasattr(cloned, "eval"):
        cloned = cloned.eval()
    return cloned


def _prepared_inputs(
    prepared: smoke._PreparedExport,
) -> dict[str, np.ndarray[Any, Any]]:
    return {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(prepared.input_names, prepared.sample_args, strict=True)
    }


def _expected_output(
    prepared: smoke._PreparedExport,
    torch: Any,
) -> tuple[str, np.ndarray[Any, Any]]:
    with torch.no_grad():
        output = prepared.model(*prepared.sample_args)
    return prepared.output_names[0], _torch_output_to_numpy(output)


def _prepare_paper_mlp(torch: Any) -> smoke._PreparedExport:
    nn = _optional_import("torch.nn")
    if nn is None:
        raise RuntimeError("torch.nn is required for the MLP smoke benchmark.")

    class PaperMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.GELU(),
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 512),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return smoke._PreparedExport(
        model=PaperMLP().eval(),
        sample_args=(torch.randn(128, 1024),),
        input_names=("x",),
        output_names=("y",),
    )


def _prepare_whisper_tiny() -> smoke._PreparedExport:
    from examples.model_whisper_tiny import (
        TinyWhisper,
        WhisperTinyConfig,
        build_demo_mel,
        build_demo_tokens,
    )

    cfg = WhisperTinyConfig()
    return smoke._PreparedExport(
        model=TinyWhisper(cfg).eval(),
        sample_args=(build_demo_mel(cfg, seed=7), build_demo_tokens(cfg, seed=11)),
        input_names=("mel", "tokens"),
        output_names=("logits",),
    )


def _prepare_benchmark_job(
    job: smoke.SmokeTranspileJob,
    torch: Any,
) -> smoke._PreparedExport | smoke.LoaderPlan:
    if job.model_name == "MLP smoke":
        return _prepare_paper_mlp(torch)
    if job.model_name == "Whisper tiny":
        return _prepare_whisper_tiny()
    return smoke.prepare_job(job)


def _summarize_times_ms(times: list[float]) -> dict[str, float | None]:
    if not times:
        return {"mean_ms": None, "p50_ms": None, "min_ms": None}
    arr = np.array(times, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "min_ms": float(np.min(arr)),
    }


def _measure_torch(
    prepared: smoke._PreparedExport,
    torch: Any,
    *,
    warmups: int,
    runs: int,
    device: str = "cpu",
    compile_model: bool = False,
) -> dict[str, Any]:
    if not _torch_device_available(torch, device):
        return {
            "status": "unavailable",
            "device": device,
            "message": f"torch device {device!r} is unavailable.",
        }
    if compile_model and not hasattr(torch, "compile"):
        return {
            "status": "unavailable",
            "device": device,
            "message": "torch.compile is unavailable.",
        }

    try:
        model = _clone_torch_model(prepared.model)
        if hasattr(model, "to"):
            model = model.to(device)
        sample_args = _torch_args_to_device(tuple(prepared.sample_args), device)
        label = f"PyTorch {'compile' if compile_model else 'eager'} {device.upper()}"
        if compile_model:
            model = torch.compile(model, mode="reduce-overhead")

        with torch.inference_mode():
            start = time.perf_counter()
            _ = model(*sample_args)
            _torch_synchronize(torch, device)
            first_forward_ms = (time.perf_counter() - start) * 1000.0

            for _ in range(warmups):
                _ = model(*sample_args)
                _torch_synchronize(torch, device)

            times: list[float] = []
            actual = None
            for _ in range(runs):
                start = time.perf_counter()
                actual = model(*sample_args)
                _torch_synchronize(torch, device)
                times.append((time.perf_counter() - start) * 1000.0)

        actual_np = None if actual is None else _torch_output_to_numpy(actual)
        return {
            "status": "ok",
            "label": label,
            "device": device,
            "first_forward_ms": first_forward_ms,
            "output_shape": None if actual_np is None else list(actual_np.shape),
            **_summarize_times_ms(times),
        }
    except Exception as exc:
        return {"status": "failed", "device": device, "message": _sanitize_text(str(exc))}


def _measure_onnxruntime(
    *,
    onnx_path: Path,
    inputs: dict[str, np.ndarray[Any, Any]],
    output_name: str,
    expected: np.ndarray[Any, Any],
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    ort = _optional_import("onnxruntime")
    if ort is None:
        return {"status": "unavailable", "message": "onnxruntime is not installed."}

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        start = time.perf_counter()
        actual = session.run([output_name], inputs)[0]
        first_forward_ms = (time.perf_counter() - start) * 1000.0
        for _ in range(warmups):
            _ = session.run([output_name], inputs)[0]
        times: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            actual = session.run([output_name], inputs)[0]
            times.append((time.perf_counter() - start) * 1000.0)
        parity = None
        if actual is not None and actual.shape == expected.shape:
            parity = float(np.max(np.abs(actual - expected)))
        return {
            "status": "ok",
            "device": "cpu",
            "first_forward_ms": first_forward_ms,
            "max_abs_error": parity,
            **_summarize_times_ms(times),
        }
    except Exception as exc:
        return {"status": "failed", "message": _sanitize_text(str(exc))}


def _force_numpy_output(value: Any, output_name: str) -> np.ndarray[Any, Any]:
    if isinstance(value, dict):
        value = value[output_name]
    return np.asarray(value)


def _runtime_device_summary(target: str) -> str:
    if target == "jax":
        jax = _optional_import("jax")
        if jax is None:
            return "unavailable"
        try:
            return ",".join(str(device) for device in jax.devices())
        except Exception:
            return "unknown"
    if target == "mlx":
        mx = _optional_import("mlx.core")
        if mx is None:
            return "unavailable"
        try:
            return str(mx.default_device())
        except Exception:
            return "unknown"
    return "-"


def _prepare_runtime_inputs(
    target: str,
    inputs: dict[str, np.ndarray[Any, Any]],
) -> dict[str, Any]:
    if target == "jax":
        jnp = _optional_import("jax.numpy")
        if jnp is None:
            return dict(inputs)
        return {name: jnp.asarray(value) for name, value in inputs.items()}
    if target == "mlx":
        mx = _optional_import("mlx.core")
        if mx is None:
            return dict(inputs)
        if hasattr(mx, "set_default_device") and hasattr(mx, "gpu"):
            mx.set_default_device(mx.gpu)
        prepared = {name: mx.asarray(value) for name, value in inputs.items()}
        for value in prepared.values():
            mx.eval(value)
        if hasattr(mx, "synchronize"):
            mx.synchronize()
        return prepared
    return dict(inputs)


def _select_runtime_output(value: Any, output_name: str) -> Any:
    if isinstance(value, dict):
        return value[output_name]
    if isinstance(value, list | tuple):
        return value[0]
    return value


def _block_runtime_output(target: str, value: Any) -> Any:
    if target == "jax" and hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif target == "mlx":
        mx = _optional_import("mlx.core")
        if mx is not None:
            mx.eval(value)
            if hasattr(mx, "synchronize"):
                mx.synchronize()
    return value


def _runtime_output_to_numpy(target: str, value: Any) -> np.ndarray[Any, Any]:
    _block_runtime_output(target, value)
    return np.asarray(value)


def _measure_generated_callable(
    *,
    target: str,
    label: str,
    function: Any,
    params: Any,
    inputs: dict[str, Any],
    output_name: str,
    expected: np.ndarray[Any, Any],
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    try:
        start = time.perf_counter()
        actual_value = _select_runtime_output(function(params, inputs), output_name)
        _block_runtime_output(target, actual_value)
        first_forward_ms = (time.perf_counter() - start) * 1000.0

        for _ in range(warmups):
            value = _select_runtime_output(function(params, inputs), output_name)
            _block_runtime_output(target, value)

        times: list[float] = []
        actual_value = None
        for _ in range(runs):
            start = time.perf_counter()
            actual_value = _select_runtime_output(function(params, inputs), output_name)
            _block_runtime_output(target, actual_value)
            times.append((time.perf_counter() - start) * 1000.0)

        parity = None
        actual = None
        if actual_value is not None:
            actual = _runtime_output_to_numpy(target, actual_value)
        if actual is not None and actual.shape == expected.shape:
            parity = float(np.max(np.abs(actual - expected)))
        return {
            "status": "ok",
            "label": label,
            "device": _runtime_device_summary(target),
            "first_forward_ms": first_forward_ms,
            "max_abs_error": parity,
            **_summarize_times_ms(times),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "label": label,
            "device": _runtime_device_summary(target),
            "message": _sanitize_text(str(exc)),
        }


def _parse_json_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _measure_generated_jax_metal_compat(
    *,
    job: smoke.SmokeTranspileJob,
    inputs_npz_path: Path,
    expected_npz_path: Path,
    output_name: str,
    warmups: int,
    runs: int,
    python_executable: str,
    jax_version: str,
    jaxlib_version: str,
) -> dict[str, Any]:
    probe_path = REPO_ROOT / "scripts" / "research" / "jax_metal_runtime_probe.py"
    command = [
        "uv",
        "run",
        "--python",
        python_executable,
        "--no-project",
        "--with",
        f"jax=={jax_version}",
        "--with",
        f"jaxlib=={jaxlib_version}",
        "--with",
        "jax-metal",
        "python",
        str(probe_path),
        "--artifact-dir",
        str(job.out_dir),
        "--inputs-npz",
        str(inputs_npz_path),
        "--expected-npz",
        str(expected_npz_path),
        "--output-name",
        output_name,
        "--warmups",
        str(warmups),
        "--runs",
        str(runs),
    ]
    env = os.environ.copy()
    env["ENABLE_PJRT_COMPATIBILITY"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except FileNotFoundError as exc:
        return {"status": "unavailable", "message": _sanitize_text(str(exc))}
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "message": "JAX Metal compatibility probe timed out after 900 seconds.",
        }

    payload = _parse_json_payload(completed.stdout)
    if payload is None:
        message = completed.stderr.strip() or completed.stdout.strip()
        return {
            "status": "failed",
            "message": _sanitize_text(message or "JAX Metal probe did not emit JSON."),
        }
    if completed.returncode != 0 and payload.get("status") != "ok":
        stderr = completed.stderr.strip()
        if stderr:
            payload["message"] = _sanitize_text(
                f"{payload.get('message', 'JAX Metal probe failed')} | {stderr}"
            )
    payload["command"] = " ".join(command[:11] + ["python", str(probe_path.name), "..."])
    return payload


def _measure_generated_runtime(
    *,
    target: str,
    job: smoke.SmokeTranspileJob,
    inputs: dict[str, np.ndarray[Any, Any]],
    output_name: str,
    expected: np.ndarray[Any, Any],
    warmups: int,
    runs: int,
    inputs_npz_path: Path,
    expected_npz_path: Path,
    jax_metal_compat_python: str | None,
    jax_metal_compat_jax: str,
    jax_metal_compat_jaxlib: str,
) -> dict[str, Any]:
    if target == "jax" and jax_metal_compat_python:
        return _measure_generated_jax_metal_compat(
            job=job,
            inputs_npz_path=inputs_npz_path,
            expected_npz_path=expected_npz_path,
            output_name=output_name,
            warmups=warmups,
            runs=runs,
            python_executable=jax_metal_compat_python,
            jax_version=jax_metal_compat_jax,
            jaxlib_version=jax_metal_compat_jaxlib,
        )
    if target == "jax" and _optional_import("jax") is None:
        return {"status": "unavailable", "message": "jax is not installed."}
    if target == "mlx" and _optional_import("mlx.core") is None:
        return {"status": "unavailable", "message": "mlx is not installed."}

    module_path = job.out_dir / f"model_{target}.py"
    try:
        start = time.perf_counter()
        module = _load_module(module_path, f"adaptfm_generated_{target}_{job.slug}")
        params = module.load_weights(str(job.out_dir / "weights.npz"))
        cold_start_ms = (time.perf_counter() - start) * 1000.0
        runtime_inputs = _prepare_runtime_inputs(target, inputs)
        variants: list[dict[str, Any]] = []
        if target == "jax":
            jax = _optional_import("jax")
            variants.append(
                _measure_generated_callable(
                    target=target,
                    label="Generated JAX eager",
                    function=module.forward,
                    params=params,
                    inputs=runtime_inputs,
                    output_name=output_name,
                    expected=expected,
                    warmups=warmups,
                    runs=runs,
                )
            )
            if jax is not None:
                jitted_forward = jax.jit(
                    lambda actual_inputs: module.forward(params, actual_inputs)
                )

                def _closed_weight_jit(_params: Any, actual_inputs: dict[str, Any]) -> Any:
                    return jitted_forward(actual_inputs)

                variants.append(
                    _measure_generated_callable(
                        target=target,
                        label="Generated JAX jit",
                        function=_closed_weight_jit,
                        params=params,
                        inputs=runtime_inputs,
                        output_name=output_name,
                        expected=expected,
                        warmups=warmups,
                        runs=runs,
                    )
                )
        else:
            variants.append(
                _measure_generated_callable(
                    target=target,
                    label="Generated MLX Metal",
                    function=module.forward,
                    params=params,
                    inputs=runtime_inputs,
                    output_name=output_name,
                    expected=expected,
                    warmups=warmups,
                    runs=runs,
                )
            )

        primary = next(
            (item for item in reversed(variants) if item.get("status") == "ok"),
            variants[-1],
        )
        return {
            "status": primary.get("status", "ok"),
            "cold_start_ms": cold_start_ms,
            "variants": variants,
            **{k: v for k, v in primary.items() if k not in {"label", "status"}},
        }
    except Exception as exc:
        return {"status": "failed", "message": _sanitize_text(str(exc))}


def _paper_mlp_weights(
    prepared: smoke._PreparedExport,
) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    layers = []
    for layer in getattr(prepared.model, "net", []):
        weight = getattr(layer, "weight", None)
        bias = getattr(layer, "bias", None)
        if weight is None or bias is None:
            continue
        layers.append((weight.detach().cpu().numpy(), bias.detach().cpu().numpy()))
    return layers


def _measure_native_mlp_baselines(
    *,
    prepared: smoke._PreparedExport,
    inputs: dict[str, np.ndarray[Any, Any]],
    expected: np.ndarray[Any, Any],
    warmups: int,
    runs: int,
) -> list[dict[str, Any]]:
    if not hasattr(prepared.model, "net") or "x" not in inputs:
        return []

    layers = _paper_mlp_weights(prepared)
    if len(layers) != 3:
        return []

    rows: list[dict[str, Any]] = []
    if (jax := _optional_import("jax")) is not None and _optional_import("jax.numpy") is not None:
        jnp = _optional_import("jax.numpy")
        try:
            jax_layers = [(jnp.asarray(weight), jnp.asarray(bias)) for weight, bias in layers]
            jax_inputs = {"x": jnp.asarray(inputs["x"])}

            def _jax_forward(_params: Any, actual_inputs: dict[str, Any]) -> dict[str, Any]:
                y = actual_inputs["x"]
                for idx, (weight, bias) in enumerate(jax_layers):
                    y = jnp.matmul(y, jnp.transpose(weight)) + bias
                    if idx < len(jax_layers) - 1:
                        y = 0.5 * y * (1.0 + jax.lax.erf(y / jnp.sqrt(2.0)))
                return {"y": y}

            jitted_jax_forward = jax.jit(lambda actual_inputs: _jax_forward(None, actual_inputs))

            for measurement in (
                _measure_generated_callable(
                    target="jax",
                    label="Native JAX eager MLP",
                    function=_jax_forward,
                    params=None,
                    inputs=jax_inputs,
                    output_name="y",
                    expected=expected,
                    warmups=warmups,
                    runs=runs,
                ),
                _measure_generated_callable(
                    target="jax",
                    label="Native JAX jit MLP",
                    function=lambda _params, actual_inputs: jitted_jax_forward(actual_inputs),
                    params=None,
                    inputs=jax_inputs,
                    output_name="y",
                    expected=expected,
                    warmups=warmups,
                    runs=runs,
                ),
            ):
                rows.append({"baseline": measurement.pop("label"), **measurement})
        except Exception as exc:
            device = _runtime_device_summary("jax")
            message = _sanitize_text(str(exc))
            for label in ("Native JAX eager MLP", "Native JAX jit MLP"):
                rows.append(
                    {
                        "baseline": label,
                        "status": "failed",
                        "device": device,
                        "message": message,
                    }
                )

    if (mx := _optional_import("mlx.core")) is not None:
        if hasattr(mx, "set_default_device") and hasattr(mx, "gpu"):
            mx.set_default_device(mx.gpu)
        mlx_layers = [(mx.array(weight), mx.array(bias)) for weight, bias in layers]
        mlx_inputs = {"x": mx.array(inputs["x"])}

        def _mlx_forward(_params: Any, actual_inputs: dict[str, Any]) -> dict[str, Any]:
            y = actual_inputs["x"]
            for idx, (weight, bias) in enumerate(mlx_layers):
                y = mx.matmul(y, mx.transpose(weight)) + bias
                if idx < len(mlx_layers) - 1:
                    y = 0.5 * y * (1.0 + mx.erf(y / mx.sqrt(mx.array(2.0))))
            return {"y": y}

        measurement = _measure_generated_callable(
            target="mlx",
            label="Native MLX Metal MLP",
            function=_mlx_forward,
            params=None,
            inputs=mlx_inputs,
            output_name="y",
            expected=expected,
            warmups=warmups,
            runs=runs,
        )
        rows.append({"baseline": measurement.pop("label"), **measurement})

    return rows


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "torchvision", "transformers", "jax", "mlx", "onnx", "onnxruntime"):
        module = _optional_import(name)
        versions[name] = None if module is None else str(getattr(module, "__version__", "unknown"))
    return versions


def _runtime_devices() -> dict[str, str]:
    devices: dict[str, str] = {}
    torch = _optional_import("torch")
    if torch is not None:
        devices["torch_cpu"] = "available"
        devices["torch_mps"] = "available" if torch.backends.mps.is_available() else "unavailable"
    jax = _optional_import("jax")
    if jax is not None:
        try:
            devices["jax"] = ", ".join(str(device) for device in jax.devices())
            devices["jax_default_backend"] = str(jax.default_backend())
        except Exception:
            devices["jax"] = "unknown"
    mx = _optional_import("mlx.core")
    if mx is not None:
        try:
            devices["mlx_default"] = str(mx.default_device())
        except Exception:
            devices["mlx_default"] = "unknown"
    return devices


def _support_table() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "order": "15",
            "model": "MLP smoke",
            "variant": "mlp_128x1024",
            "status": "ready",
            "notes": "Batched GEMM/GELU smoke benchmark used for paper evidence.",
        },
        {
            "order": "16",
            "model": "Whisper tiny",
            "variant": "whisper_tiny",
            "status": "ready",
            "notes": "Tiny Whisper-style encoder/decoder benchmark used for paper evidence.",
        },
    ]
    for job in smoke.list_smoke_jobs(target="jax"):
        rows.append(
            {
                "order": str(job.order),
                "model": job.model_name,
                "variant": job.variant_name,
                "status": job.status,
                "notes": job.notes,
            }
        )
    return sorted(rows, key=lambda row: (int(row["order"]), row["model"], row["variant"]))


def _short_hash(value: str | None) -> str:
    return "-" if value is None else value[:12]


def _display_model_name(model_name: str, variant_name: str) -> str:
    if variant_name == "llama_3_1_8b_smoke":
        return "Compact Llama decoder"
    if model_name == "MLP smoke":
        return "Dense MLP"
    return model_name


def _format_ms(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int | float):
        return f"{float(value):.2f}"
    return str(value)


def _format_bytes(value: Any) -> str:
    if not isinstance(value, int | float):
        return "-"
    return f"{float(value) / 1024.0:.1f} KiB"


def _markdown_table(headers: list[str], rows: Iterable[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# TNNX Paper Evidence Report",
        "",
        f"- Generated at: `{result['metadata']['generated_at_utc']}`",
        f"- Artifact root: `{result['metadata']['artifact_root']}`",
        f"- Python: `{result['metadata']['python']}`",
        f"- Platform: `{result['metadata']['platform']}`",
        "",
        "## Runtime Versions",
        "",
    ]
    version_rows = (
        [name, version or "unavailable"] for name, version in result["versions"].items()
    )
    lines.append(_markdown_table(["Dependency", "Version"], version_rows))

    lines.extend(["", "## Runtime Devices", ""])
    device_rows = ([name, value] for name, value in result.get("devices", {}).items())
    lines.append(_markdown_table(["Runtime", "Device"], device_rows))

    lines.extend(["", "## Baseline Timings", ""])
    lines.append(
        "First ms is measured separately; mean and p50 are after warmups and exclude that "
        "first call."
    )
    lines.append("")
    baseline_rows = []
    for row in result["baselines"]:
        baseline_rows.append(
            [
                _display_model_name(row["model"], row["variant"]),
                row["variant"],
                row["baseline"],
                row["status"],
                row.get("device", "-"),
                _format_ms(row.get("first_forward_ms")),
                _format_ms(row.get("mean_ms")),
                _format_ms(row.get("p50_ms")),
                _format_ms(row.get("max_abs_error")),
                row.get("message", ""),
            ]
        )
    lines.append(
        _markdown_table(
            [
                "Model",
                "Variant",
                "Baseline",
                "Status",
                "Device",
                "First ms",
                "Mean ms",
                "P50 ms",
                "Max abs",
                "Notes",
            ],
            baseline_rows,
        )
    )

    lines.extend(["", "## Generated Artifact Evidence", ""])
    artifact_rows = []
    for row in result["artifacts"]:
        runtime = row.get("runtime", {})
        determinism = row.get("determinism", {})
        variants = runtime.get("variants")
        if not variants:
            variants = [runtime]
        for variant in variants:
            artifact_rows.append(
                [
                    _display_model_name(row["model"], row["variant"]),
                    row["variant"],
                    row["target"],
                    row["status"],
                    _format_ms(row.get("compile_ms")),
                    _format_bytes(row.get("artifact_bytes")),
                    str(row.get("generated_loc", "-")),
                    _short_hash(row.get("graph_ir_sha256")),
                    _short_hash(row.get("source_sha256")),
                    _short_hash(row.get("weights_npz_sha256")),
                    str(determinism.get("semantic_stable", "-")),
                    variant.get("label", runtime.get("status", "-")),
                    variant.get("status", runtime.get("status", "-")),
                    variant.get("device", runtime.get("device", "-")),
                    _format_ms(runtime.get("cold_start_ms")),
                    _format_ms(variant.get("first_forward_ms", runtime.get("first_forward_ms"))),
                    _format_ms(variant.get("mean_ms", runtime.get("mean_ms"))),
                    _format_ms(variant.get("max_abs_error", runtime.get("max_abs_error"))),
                    row.get("message", "") or variant.get("message", ""),
                ]
            )
    lines.append(
        _markdown_table(
            [
                "Model",
                "Variant",
                "Target",
                "Status",
                "Compile ms",
                "Size",
                "LOC",
                "Graph hash",
                "Source hash",
                "Weights hash",
                "Stable",
                "Runtime mode",
                "Runtime",
                "Device",
                "Cold ms",
                "First ms",
                "Mean ms",
                "Max abs",
                "Notes",
            ],
            artifact_rows,
        )
    )

    lines.extend(["", "## Model-Zoo Support Matrix", ""])
    support_rows = (
        [
            row["order"],
            _display_model_name(row["model"], row["variant"]),
            row["variant"],
            row["status"],
            row["notes"],
        ]
        for row in result["support_matrix"]
    )
    lines.append(_markdown_table(["Order", "Model", "Variant", "Status", "Notes"], support_rows))

    lines.extend(
        [
            "",
            "## Anonymization Checklist",
            "",
            "- Paper source uses `Anonymous Authors`.",
            "- Do not include local absolute paths in the submitted artifact.",
            "- Do not include non-anonymous repository URLs in the submitted PDF.",
            "- Keep generated evidence JSON/Markdown, but remove machine-specific artifact roots.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = _optional_import("torch")
    if torch is not None:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    artifact_root = Path(args.artifact_root)
    if args.artifact_root_was_default:
        artifact_root = artifact_root / _now_slug()
    artifact_root.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "metadata": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "artifact_root": str(artifact_root),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "runs": args.runs,
            "warmups": args.warmups,
            "seed": args.seed,
        },
        "versions": _runtime_versions(),
        "devices": _runtime_devices(),
        "support_matrix": _support_table(),
        "baselines": [],
        "artifacts": [],
    }

    selected_models = [_parse_model(value) for value in args.model]
    for model_name, variant_name in selected_models:
        model_root = artifact_root / f"{model_name.lower().replace(' ', '_')}__{variant_name}"
        base_job = _select_job(
            target="jax",
            model_name=model_name,
            variant_name=variant_name,
            out_root=model_root,
        )
        if torch is None:
            result["artifacts"].append(
                {
                    "model": model_name,
                    "variant": variant_name,
                    "target": "all",
                    "status": "blocked",
                    "message": "torch is required to export benchmark models.",
                }
            )
            continue
        try:
            prepared = _prepare_benchmark_job(base_job, torch)
        except Exception as exc:
            result["artifacts"].append(
                {
                    "model": model_name,
                    "variant": variant_name,
                    "target": "all",
                    "status": "failed",
                    "message": f"prepare failed: {exc}",
                }
            )
            continue
        if isinstance(prepared, smoke.LoaderPlan):
            result["artifacts"].append(
                {
                    "model": model_name,
                    "variant": variant_name,
                    "target": "all",
                    "status": "blocked",
                    "message": prepared.install_hint,
                }
            )
            continue

        inputs = _prepared_inputs(prepared)
        output_name, expected = _expected_output(prepared, torch)
        model_root.mkdir(parents=True, exist_ok=True)
        inputs_npz_path = model_root / "runtime_inputs.npz"
        expected_npz_path = model_root / "expected_outputs.npz"
        np.savez(inputs_npz_path, **inputs)
        np.savez(expected_npz_path, **{output_name: expected})
        for native_row in _measure_native_mlp_baselines(
            prepared=prepared,
            inputs=inputs,
            expected=expected,
            warmups=args.warmups,
            runs=args.runs,
        ):
            result["baselines"].append(
                {
                    "model": model_name,
                    "variant": variant_name,
                    **native_row,
                }
            )
        for baseline_label, device, compile_model in (
            ("PyTorch eager CPU", "cpu", False),
            ("PyTorch compile CPU", "cpu", True),
            ("PyTorch eager MPS", "mps", False),
            ("PyTorch compile MPS", "mps", True),
        ):
            result["baselines"].append(
                {
                    "model": model_name,
                    "variant": variant_name,
                    "baseline": baseline_label,
                    **_measure_torch(
                        prepared,
                        torch,
                        warmups=args.warmups,
                        runs=args.runs,
                        device=device,
                        compile_model=compile_model,
                    ),
                }
            )

        export_start = time.perf_counter()
        try:
            onnx_path = smoke._export_to_onnx(base_job, prepared)
            export_ms = (time.perf_counter() - export_start) * 1000.0
        except Exception as exc:
            result["artifacts"].append(
                {
                    "model": model_name,
                    "variant": variant_name,
                    "target": "onnx",
                    "status": "failed",
                    "message": f"ONNX export failed: {exc}",
                }
            )
            continue

        ort_result = _measure_onnxruntime(
            onnx_path=onnx_path,
            inputs=inputs,
            output_name=output_name,
            expected=expected,
            warmups=args.warmups,
            runs=args.runs,
        )
        result["baselines"].append(
            {
                "model": model_name,
                "variant": variant_name,
                "baseline": "ONNX Runtime CPU",
                **ort_result,
            }
        )

        for target in args.targets:
            job = _select_job(
                target=target,
                model_name=model_name,
                variant_name=variant_name,
                out_root=model_root,
            )
            compile_start = time.perf_counter()
            try:
                transpile_onnx(str(onnx_path), target, str(job.out_dir), config=CompileConfig())
                compile_ms = (time.perf_counter() - compile_start) * 1000.0
                stats = _artifact_stats(target, job.out_dir)
                determinism = (
                    {}
                    if args.skip_determinism
                    else _determinism(
                        target=target,
                        onnx_path=onnx_path,
                        first_dir=job.out_dir,
                        repeat_dir=job.out_dir.with_name(job.out_dir.name + "__repeat"),
                    )
                )
                runtime = _measure_generated_runtime(
                    target=target,
                    job=job,
                    inputs=inputs,
                    output_name=output_name,
                    expected=expected,
                    warmups=args.warmups,
                    runs=args.runs,
                    inputs_npz_path=inputs_npz_path,
                    expected_npz_path=expected_npz_path,
                    jax_metal_compat_python=args.jax_metal_compat_python,
                    jax_metal_compat_jax=args.jax_metal_compat_jax,
                    jax_metal_compat_jaxlib=args.jax_metal_compat_jaxlib,
                )
                result["artifacts"].append(
                    {
                        "model": model_name,
                        "variant": variant_name,
                        "target": target,
                        "status": "ok",
                        "onnx_export_ms": export_ms,
                        "compile_ms": compile_ms,
                        **stats,
                        "determinism": determinism,
                        "runtime": runtime,
                    }
                )
            except Exception as exc:
                result["artifacts"].append(
                    {
                        "model": model_name,
                        "variant": variant_name,
                        "target": target,
                        "status": "failed",
                        "message": _sanitize_text(str(exc)),
                    }
                )

    json_path = Path(args.result_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(result, Path(args.result_md))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper evidence tables from local tnnx model-zoo jobs."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model selection as 'model name::variant name'. Can be repeated.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=DEFAULT_TARGETS,
        default=list(DEFAULT_TARGETS),
        help="Generated artifact targets to measure.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Timed iterations per baseline.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup iterations before timing.")
    parser.add_argument("--seed", type=int, default=0, help="Torch/NumPy seed for repeatability.")
    parser.add_argument(
        "--artifact-root",
        default="generated/tnnx_evidence",
        help="Root directory for generated ONNX and backend artifacts.",
    )
    parser.add_argument(
        "--result-json",
        default="docs/research_wiki/results/tnnx_evidence.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--result-md",
        default="docs/research_wiki/results/tnnx_evidence.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--skip-determinism",
        action="store_true",
        help="Skip repeat transpilation and deterministic artifact comparison.",
    )
    parser.add_argument(
        "--jax-metal-compat-python",
        default=None,
        help=(
            "Run generated JAX timings in an isolated Apple-compatible JAX Metal "
            "environment using this Python executable, for example "
            "/opt/homebrew/bin/python3.13."
        ),
    )
    parser.add_argument(
        "--jax-metal-compat-jax",
        default="0.4.34",
        help="JAX version for the isolated JAX Metal compatibility probe.",
    )
    parser.add_argument(
        "--jax-metal-compat-jaxlib",
        default="0.4.34",
        help="jaxlib version for the isolated JAX Metal compatibility probe.",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.model is None:
        args.model = list(DEFAULT_MODELS)
    args.artifact_root_was_default = args.artifact_root == parser.get_default("artifact_root")
    result = run(args)
    print(f"Wrote {args.result_json}")
    print(f"Wrote {args.result_md}")
    print(f"Artifact root: {result['metadata']['artifact_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
