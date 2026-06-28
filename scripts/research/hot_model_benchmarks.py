from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples.common import load_generated_module  # noqa: E402
from examples.flux import source as flux_source  # noqa: E402
from examples.flux.transpile_and_generate_jax import (  # noqa: E402
    DEMO_IMAGE_SIZE,
    _save_png,
    _to_image_array,
    build_demo_export_specs,
    build_demo_inputs,
    export_flux_submodule_onnx,
)
from examples.qwen import infer_qwen3_5_from_transformers_jax as qwen_jax  # noqa: E402
from tnnx.api import transpile_onnx  # noqa: E402
from tnnx.config import CompileConfig  # noqa: E402


def _summarize(times: list[float]) -> dict[str, float | int | None]:
    if not times:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "runs": 0}
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "runs": int(len(times)),
    }


def _measure(
    fn: Any,
    *,
    warmups: int,
    runs: int,
    sync: Any | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    first_result = fn()
    if sync is not None:
        sync()
    first_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(warmups):
        _ = fn()
        if sync is not None:
            sync()

    times: list[float] = []
    last_result = first_result
    for _ in range(runs):
        start = time.perf_counter()
        last_result = fn()
        if sync is not None:
            sync()
        times.append((time.perf_counter() - start) * 1000.0)

    return {
        "first_ms": first_ms,
        "last_result": last_result,
        **_summarize(times),
    }


def _max_abs(lhs: Any, rhs: Any) -> float:
    lhs_np = np.asarray(lhs)
    rhs_np = np.asarray(rhs)
    if lhs_np.shape != rhs_np.shape:
        return math.inf
    return float(np.max(np.abs(lhs_np - rhs_np)))


def _runtime_info(target: str) -> dict[str, str]:
    if target == "jax":
        import jax

        return {
            "device": ",".join(str(device) for device in jax.devices()),
            "default_backend": str(jax.default_backend()),
        }
    if target == "mlx":
        import mlx.core as mx

        return {
            "device": str(mx.default_device()),
            "default_backend": "mlx",
        }
    return {"device": "unknown", "default_backend": "unknown"}


def _first_tensor_np(output: Any) -> np.ndarray[Any, Any]:
    if hasattr(output, "detach"):
        return output.detach().cpu().numpy()
    if isinstance(output, dict):
        for value in output.values():
            return _first_tensor_np(value)
    if isinstance(output, tuple | list):
        for value in output:
            return _first_tensor_np(value)
    return np.asarray(output)


def _qwen_prompt_ids(tokenizer: Any, prompt: str) -> tuple[list[int], int]:
    eos_id = int(tokenizer.eos_token_id)
    prompt_ids = [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
    return (prompt_ids or [eos_id]), eos_id


def _qwen_prepare_generated(
    *,
    out_dir: Path,
    backend: str,
    local_files_only: bool,
    force: bool,
) -> dict[str, Any]:
    snapshot_root = qwen_jax._resolve_prebuilt_snapshot(local_files_only=local_files_only)
    tokenizer = qwen_jax._load_prebuilt_tokenizer(snapshot_root)
    embed_onnx_path, decoder_onnx_path = qwen_jax._materialize_prebuilt_onnx_assets(
        out_dir,
        snapshot_root,
    )
    _localize_qwen_external_data(embed_onnx_path.parent)
    decoder_ops, decoder_unsupported = qwen_jax._inspect_onnx(decoder_onnx_path)
    if decoder_unsupported:
        raise ValueError(
            "Prebuilt Qwen decoder still has unsupported ops after decode-only specialization: "
            f"{list(decoder_unsupported)}"
        )

    embed_generated_dir = out_dir / f"generated_qwen_embed_tokens_{backend}"
    decoder_generated_dir = out_dir / f"generated_qwen_decoder_merged_{backend}"
    source_name = "model_jax.py" if backend == "jax" else "model_mlx.py"
    embed_source = embed_generated_dir / source_name
    decoder_source = decoder_generated_dir / source_name
    embed_weights = embed_generated_dir / "weights.npz"
    decoder_weights = decoder_generated_dir / "weights.npz"

    prep_start = time.perf_counter()
    if force or not (embed_source.exists() and embed_weights.exists()):
        embed_manifest = transpile_onnx(
            str(embed_onnx_path),
            backend,
            str(embed_generated_dir),
            config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
        )
        embed_weights = embed_manifest.weights_file
    if force or not (decoder_source.exists() and decoder_weights.exists()):
        decoder_manifest = transpile_onnx(
            str(decoder_onnx_path),
            backend,
            str(decoder_generated_dir),
            config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
        )
        decoder_weights = decoder_manifest.weights_file
    prep_ms = (time.perf_counter() - prep_start) * 1000.0

    load_start = time.perf_counter()
    embed_module = load_generated_module(
        embed_source,
        module_name=f"generated_qwen_embed_tokens_{backend}_bench",
    )
    embed_params = embed_module.load_weights(str(embed_weights))
    decoder_module = load_generated_module(
        decoder_source,
        module_name=f"generated_qwen_decoder_merged_{backend}_bench",
    )
    decoder_params = decoder_module.load_weights(str(decoder_weights))
    load_ms = (time.perf_counter() - load_start) * 1000.0

    runtime = _runtime_info(backend)

    return {
        "snapshot_root": snapshot_root,
        "tokenizer": tokenizer,
        "embed_onnx_path": embed_onnx_path,
        "decoder_onnx_path": decoder_onnx_path,
        "embed_module": embed_module,
        "embed_params": embed_params,
        "decoder_module": decoder_module,
        "decoder_params": decoder_params,
        "decoder_specs": qwen_jax._onnx_input_specs(decoder_onnx_path),
        "decoder_ops": decoder_ops,
        "device": runtime["device"],
        "default_backend": runtime["default_backend"],
        "prep_ms_not_timed": prep_ms,
        "load_ms_not_timed": load_ms,
        "embed_source": embed_source,
        "decoder_source": decoder_source,
    }


def _localize_qwen_external_data(assets_dir: Path) -> None:
    for path in assets_dir.glob("*.onnx_data*"):
        if not path.is_symlink():
            continue
        source = path.resolve()
        path.unlink()
        try:
            os.link(source, path)
        except OSError:
            shutil.copy2(source, path)


def _qwen_generated_once(
    *,
    prepared: dict[str, Any],
    prompt_ids: list[int],
    eos_id: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = qwen_jax._decoder_state_from_specs(prepared["decoder_specs"])
    current_token_id = int(prompt_ids[0])
    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    total_steps = len(prompt_ids) - 1 + max_new_tokens
    position = 0
    last_logits = None

    for step_idx in range(total_steps):
        token_inputs = {"input_ids": np.asarray([[current_token_id]], dtype=np.int64)}
        embed_outputs = prepared["embed_module"].forward(prepared["embed_params"], token_inputs)
        inputs_embeds = np.asarray(embed_outputs["inputs_embeds"])
        decoder_inputs = qwen_jax._build_decode_step_inputs(
            state=state,
            inputs_embeds=inputs_embeds,
            position=position,
        )
        outputs = prepared["decoder_module"].forward(prepared["decoder_params"], decoder_inputs)
        logits = np.asarray(outputs["logits"])
        last_logits = logits
        state = qwen_jax._next_decoder_state(outputs)

        if step_idx < len(prompt_ids) - 1:
            current_token_id = int(prompt_ids[step_idx + 1])
            position += 1
            continue

        next_id = qwen_jax._sample_next_token(
            logits[0, 0],
            temperature=temperature,
            top_k=top_k,
            recent_ids=prompt_ids + generated_ids,
            repetition_penalty=repetition_penalty,
            rng=rng,
        )
        generated_ids.append(next_id)
        position += 1
        if next_id == eos_id:
            stop_reason = "eos"
            break
        current_token_id = next_id

    return {
        "generated_ids": generated_ids,
        "stop_reason": stop_reason,
        "last_logits": last_logits,
    }


def _qwen_generated_error(
    prepared: dict[str, Any],
    *,
    prompt_ids: list[int],
) -> dict[str, float]:
    import onnxruntime as ort

    embed_session = ort.InferenceSession(
        str(prepared["embed_onnx_path"]),
        providers=["CPUExecutionProvider"],
    )
    decoder_session = ort.InferenceSession(
        str(prepared["decoder_onnx_path"]),
        providers=["CPUExecutionProvider"],
    )
    token_inputs = {"input_ids": np.asarray([[int(prompt_ids[0])]], dtype=np.int64)}
    embed_expected = embed_session.run(None, token_inputs)[0]
    embed_actual = prepared["embed_module"].forward(prepared["embed_params"], token_inputs)[
        "inputs_embeds"
    ]
    embed_max_abs = _max_abs(embed_actual, embed_expected)

    decoder_inputs = qwen_jax._build_decode_step_inputs(
        state=qwen_jax._decoder_state_from_specs(prepared["decoder_specs"]),
        inputs_embeds=embed_expected,
        position=0,
    )
    decoder_expected_values = decoder_session.run(None, decoder_inputs)
    decoder_expected = {
        output.name: value
        for output, value in zip(
            decoder_session.get_outputs(),
            decoder_expected_values,
            strict=True,
        )
    }
    decoder_actual = prepared["decoder_module"].forward(
        prepared["decoder_params"],
        decoder_inputs,
    )
    logits_max_abs = _max_abs(decoder_actual["logits"], decoder_expected["logits"])
    state_errors = [
        _max_abs(decoder_actual[name], expected)
        for name, expected in decoder_expected.items()
        if name != "logits" and name in decoder_actual
    ]
    return {
        "embed_max_abs": embed_max_abs,
        "decoder_logits_max_abs": logits_max_abs,
        "decoder_state_max_abs": float(max(state_errors)) if state_errors else 0.0,
    }


def benchmark_qwen_generated(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    prepared = _qwen_prepare_generated(
        out_dir=args.out_dir / f"qwen_generated_{backend}",
        backend=backend,
        local_files_only=True,
        force=args.force,
    )
    prompt_ids, eos_id = _qwen_prompt_ids(prepared["tokenizer"], args.prompt)
    if len(prompt_ids) > args.context_window:
        raise ValueError(
            f"Prompt token count ({len(prompt_ids)}) exceeds context window {args.context_window}."
        )

    errors = _qwen_generated_error(prepared, prompt_ids=prompt_ids)

    def _run() -> dict[str, Any]:
        return _qwen_generated_once(
            prepared=prepared,
            prompt_ids=prompt_ids,
            eos_id=eos_id,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=0,
            repetition_penalty=1.0,
            seed=args.seed,
        )

    timing = _measure(_run, warmups=args.warmups, runs=args.runs)
    last = timing.pop("last_result")
    generated_text = prepared["tokenizer"].decode(
        last["generated_ids"],
        skip_special_tokens=True,
    )
    return {
        "name": f"qwen_generated_{backend}",
        "backend": backend,
        "metric": f"{args.max_new_tokens}-token generated serving loop",
        "timing_excludes": [
            "ONNX materialization",
            "transpilation",
            "generated-module import",
            "weight loading",
        ],
        "prep_ms_not_timed": prepared["prep_ms_not_timed"],
        "load_ms_not_timed": prepared["load_ms_not_timed"],
        "generated_token_count": len(last["generated_ids"]),
        "generated_text": generated_text,
        "stop_reason": last["stop_reason"],
        "generated_module": str(prepared["decoder_source"]),
        "device": prepared["device"],
        "default_backend": prepared["default_backend"],
        "error": errors,
        **timing,
    }


def _torch_mps_sync() -> None:
    import torch

    if hasattr(torch, "mps"):
        torch.mps.synchronize()


def benchmark_qwen_pytorch(args: argparse.Namespace, *, compile_model: bool) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    model_id = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        local_files_only=True,
        dtype=torch.float16,
    ).eval()
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    model = model.to("mps")
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {key: value.to("mps") for key, value in encoded.items()}

    def _run() -> Any:
        with torch.inference_mode():
            return model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

    timing = _measure(_run, warmups=args.warmups, runs=args.runs, sync=_torch_mps_sync)
    last = timing.pop("last_result")
    return {
        "name": "qwen_pytorch_compile_mps" if compile_model else "qwen_pytorch_eager_mps",
        "backend": "pytorch_compile_mps" if compile_model else "pytorch_eager_mps",
        "device": "mps",
        "metric": f"{args.max_new_tokens}-token generation",
        "timing_excludes": ["model loading", "tokenizer loading", "weight loading"],
        "generated_text": tokenizer.decode(last[0], skip_special_tokens=True),
        "error": {"token_reference": "self"},
        **timing,
    }


def _transpile_flux_spec(
    root: Path,
    *,
    target: str,
    submodule: str,
    spec: Any,
) -> tuple[Path, Path, Any, Any]:
    onnx_path = export_flux_submodule_onnx(
        submodule,
        root / f"{submodule}.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
    )
    generated_dir = root / f"generated_{submodule}_{target}"
    manifest = transpile_onnx(
        str(onnx_path),
        target,
        str(generated_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )
    source_name = "model_jax.py" if target == "jax" else "model_mlx.py"
    module = load_generated_module(
        generated_dir / source_name,
        module_name=f"generated_flux_{submodule}_{target}_bench",
    )
    params = module.load_weights(str(manifest.weights_file))
    return onnx_path, generated_dir, module, params


def _flux_generated_once(
    *,
    transformer_module: Any,
    transformer_params: Any,
    vae_module: Any,
    vae_params: Any,
    latents: np.ndarray[Any, Any],
    prompt_embeddings: np.ndarray[Any, Any],
    pooled_prompt: np.ndarray[Any, Any],
    steps: int,
    blend: float,
) -> np.ndarray[Any, Any]:
    current = np.array(latents, copy=True)
    for _ in range(int(steps)):
        denoised = np.asarray(
            transformer_module.forward(
                transformer_params,
                {
                    "latents": current,
                    "prompt_embeddings": prompt_embeddings,
                    "pooled_prompt": pooled_prompt,
                },
            )["denoised"],
            dtype=np.float32,
        )
        current = ((1.0 - blend) * current) + (blend * denoised)
    return np.asarray(vae_module.forward(vae_params, {"latents": current})["image"])


def _flux_spec_inputs_np(spec: Any) -> dict[str, np.ndarray[Any, Any]]:
    return {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(spec.input_names, spec.sample_inputs, strict=True)
    }


def benchmark_flux_reduced(args: argparse.Namespace, target: str) -> dict[str, Any]:
    import torch

    root = args.out_dir / f"flux_reduced_{target}"
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    specs = build_demo_export_specs(seed=args.seed, image_size=DEMO_IMAGE_SIZE)
    transformer_spec = specs["transformer"]
    vae_spec = specs["vae_decoder"]
    demo_inputs = build_demo_inputs(seed=args.seed)
    latents_np = demo_inputs["latents"].detach().cpu().numpy()
    prompt_np = demo_inputs["prompt_embeddings"].detach().cpu().numpy()
    pooled_np = demo_inputs["pooled_prompt"].detach().cpu().numpy()

    prep_start = time.perf_counter()
    _, transformer_dir, transformer_module, transformer_params = _transpile_flux_spec(
        root / "transformer",
        target=target,
        submodule="transformer",
        spec=transformer_spec,
    )
    _, vae_dir, vae_module, vae_params = _transpile_flux_spec(
        root / "vae",
        target=target,
        submodule="vae_decoder",
        spec=vae_spec,
    )
    prep_ms = (time.perf_counter() - prep_start) * 1000.0

    torch_current = demo_inputs["latents"].detach().clone()
    prompt_t = torch.from_numpy(prompt_np.copy())
    pooled_t = torch.from_numpy(pooled_np.copy())
    for _ in range(int(args.flux_steps)):
        torch_denoised = transformer_spec.module(torch_current, prompt_t, pooled_t)
        torch_current = ((1.0 - args.flux_blend) * torch_current) + (
            args.flux_blend * torch_denoised
        )
    expected = vae_spec.module(torch_current).detach().cpu().numpy()

    def _run() -> np.ndarray[Any, Any]:
        return _flux_generated_once(
            transformer_module=transformer_module,
            transformer_params=transformer_params,
            vae_module=vae_module,
            vae_params=vae_params,
            latents=latents_np,
            prompt_embeddings=prompt_np,
            pooled_prompt=pooled_np,
            steps=args.flux_steps,
            blend=args.flux_blend,
        )

    timing = _measure(_run, warmups=args.warmups, runs=args.runs)
    generated = timing.pop("last_result")
    image_path = root / f"flux_reduced_{target}.png"
    _save_png(generated, image_path)
    rgb = _to_image_array(generated)
    diff = np.abs(np.asarray(generated) - expected)
    return {
        "name": f"flux_reduced_{target}",
        "backend": target,
        **_runtime_info(target),
        "metric": f"{args.flux_steps}-step reduced image loop",
        "timing_excludes": [
            "ONNX export",
            "transpilation",
            "generated-module import",
            "weight loading",
        ],
        "prep_ms_not_timed": prep_ms,
        "transformer_module": str(transformer_dir),
        "vae_module": str(vae_dir),
        "image_path": str(image_path),
        "image_size": int(rgb.shape[0]),
        "pixel_std": float(np.std(rgb)),
        "error": {
            "max_abs": float(np.max(diff)),
            "mean_abs": float(np.mean(diff)),
        },
        **timing,
    }


def benchmark_flux_checkpoint_submodule(
    args: argparse.Namespace,
    *,
    target: str,
    submodule: str,
) -> dict[str, Any]:
    if submodule != "vae_decoder":
        raise ValueError(
            "Hot checkpoint timing is currently limited to the real-weight FLUX VAE decoder. "
            "The full transformer/text-encoder checkpoint artifacts are inspected separately."
        )
    import torch

    root = args.out_dir / f"flux_checkpoint_{submodule}_{target}"
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prep_start = time.perf_counter()
    spec = flux_source._real_export_spec(
        submodule,  # pyright: ignore[reportPrivateUsage]
        model_id=args.flux_model_id,
        load_weights=True,
    )
    _, generated_dir, module, params = _transpile_flux_spec(
        root,
        target=target,
        submodule=submodule,
        spec=spec,
    )
    prep_ms = (time.perf_counter() - prep_start) * 1000.0

    inputs = _flux_spec_inputs_np(spec)
    with torch.no_grad():
        expected = _first_tensor_np(spec.module(*spec.sample_inputs))

    def _run() -> np.ndarray[Any, Any]:
        return _first_tensor_np(module.forward(params, inputs))

    timing = _measure(_run, warmups=args.warmups, runs=args.runs)
    generated = timing.pop("last_result")
    diff = np.abs(np.asarray(generated) - expected)
    return {
        "name": f"flux_checkpoint_{submodule}_{target}",
        "backend": target,
        **_runtime_info(target),
        "model_id": args.flux_model_id,
        "submodule": submodule,
        "metric": "real-checkpoint VAE decoder forward",
        "timing_excludes": [
            "checkpoint loading",
            "ONNX export",
            "transpilation",
            "generated-module import",
            "weight loading",
        ],
        "prep_ms_not_timed": prep_ms,
        "generated_module": str(generated_dir),
        "output_shape": [int(dim) for dim in np.asarray(generated).shape],
        "error": {
            "max_abs": float(np.max(diff)),
            "mean_abs": float(np.mean(diff)),
        },
        **timing,
    }


def inspect_flux_full_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    try:
        snapshot = flux_source.resolve_flux_snapshot(args.flux_model_id)
    except FileNotFoundError as exc:
        return {
            "name": "flux_full_checkpoint",
            "status": "missing_snapshot",
            "model_id": args.flux_model_id,
            "message": str(exc),
        }
    submodules = {}
    ready = True
    for submodule in flux_source.real_submodule_export_order(
        snapshot=snapshot,
        model_id=args.flux_model_id,
    ):
        has_weights = flux_source.snapshot_has_submodule_weights(snapshot, submodule)
        submodules[submodule] = {"has_weights": has_weights}
        ready = ready and has_weights
    return {
        "name": "flux_full_checkpoint",
        "status": "ready" if ready else "incomplete_snapshot",
        "model_id": args.flux_model_id,
        "snapshot": f"<hf-cache-snapshot:{snapshot.name}>",
        "submodules": submodules,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for backend in args.qwen_generated_backend:
        results.append(benchmark_qwen_generated(args, backend))
    if args.qwen_pytorch:
        results.append(benchmark_qwen_pytorch(args, compile_model=False))
        results.append(benchmark_qwen_pytorch(args, compile_model=True))
    for target in args.flux_reduced_target:
        results.append(benchmark_flux_reduced(args, target))
    for target in args.flux_checkpoint_target:
        results.append(
            benchmark_flux_checkpoint_submodule(
                args,
                target=target,
                submodule=args.flux_checkpoint_submodule,
            )
        )
    if args.inspect_flux_full:
        results.append(inspect_flux_full_checkpoint(args))

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": args.runs,
        "warmups": args.warmups,
        "results": results,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated hot-model benchmark evidence.")
    parser.add_argument("--out-dir", type=Path, default=Path("/private/tmp/tnnx_hot_models"))
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("docs/research_wiki/results/hot_model_benchmarks.json"),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="Explain ONNX to MLX conversion in one sentence.")
    parser.add_argument("--context-window", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--qwen-generated-backend",
        nargs="*",
        choices=("jax", "mlx"),
        default=["mlx"],
    )
    parser.add_argument("--qwen-pytorch", action="store_true")
    parser.add_argument(
        "--flux-reduced-target",
        nargs="*",
        choices=("jax", "mlx"),
        default=["mlx"],
    )
    parser.add_argument("--flux-steps", type=int, default=2)
    parser.add_argument("--flux-blend", type=float, default=0.4)
    parser.add_argument(
        "--flux-checkpoint-target",
        nargs="*",
        choices=("jax", "mlx"),
        default=[],
    )
    parser.add_argument(
        "--flux-checkpoint-submodule",
        choices=("vae_decoder",),
        default="vae_decoder",
    )
    parser.add_argument("--inspect-flux-full", action="store_true")
    parser.add_argument("--flux-model-id", default=flux_source.DEFAULT_REAL_FLUX_MODEL_ID)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
