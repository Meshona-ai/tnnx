from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from examples.common import load_generated_module
from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig

from . import source as flux_source
from .runtime_env import download_snapshot_from_hub
from .source import (
    DEMO_IMAGE_SIZE,
    DEMO_SEED,
    ExportSpec,
    FluxSubmodule,
    build_demo_export_specs,
    build_demo_inputs,
    build_dummy_flux2_export_specs,
    build_tiny_flux2_export_specs,
    export_flux_submodule_onnx,
)


def _to_image_array(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 4 or data.shape[0] != 1 or data.shape[1] != 3:
        raise ValueError(f"Expected BCHW image tensor, got shape {data.shape}")
    chw = np.clip((data[0] + 1.0) * 0.5, 0.0, 1.0)
    return np.transpose(chw, (1, 2, 0))


def _save_png(image: np.ndarray, path: Path) -> None:
    rgb = (_to_image_array(image) * 255.0).round().astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _write_json_report(report_path: Path, report: dict[str, Any]) -> str | None:
    try:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        return str(exc)
    return None


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _load_prompt_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(Path(path))
    if "prompt_embeddings" not in payload or "pooled_prompt" not in payload:
        raise ValueError("Prompt fixture must contain 'prompt_embeddings' and 'pooled_prompt'.")
    return (
        np.asarray(payload["prompt_embeddings"], dtype=np.float32),
        np.asarray(payload["pooled_prompt"], dtype=np.float32),
    )


def _load_token_ids_npz(path: str | Path) -> np.ndarray:
    payload = np.load(Path(path))
    if "input_ids" not in payload:
        raise ValueError("Token fixture must contain 'input_ids'.")
    return np.asarray(payload["input_ids"], dtype=np.int64)


def _resolve_token_ids(
    default_input_ids: np.ndarray,
    provided_input_ids: np.ndarray | None,
    *,
    name: str,
) -> np.ndarray:
    if provided_input_ids is None:
        return np.array(default_input_ids, copy=True)

    token_ids = np.asarray(provided_input_ids, dtype=np.int64)
    if token_ids.shape != default_input_ids.shape:
        raise ValueError(f"Expected {name} shape {default_input_ids.shape}, got {token_ids.shape}")
    return np.array(token_ids, copy=True)


def _resolve_prompt_inputs(
    default_prompt: np.ndarray,
    default_pooled: np.ndarray,
    *,
    prompt_embeddings: np.ndarray | None,
    pooled_prompt: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if prompt_embeddings is None and pooled_prompt is None:
        return default_prompt, default_pooled, "default"
    if prompt_embeddings is None or pooled_prompt is None:
        raise ValueError("prompt_embeddings and pooled_prompt must be provided together.")

    prompt_np = np.asarray(prompt_embeddings, dtype=np.float32)
    pooled_np = np.asarray(pooled_prompt, dtype=np.float32)
    if prompt_np.shape != default_prompt.shape:
        raise ValueError(
            f"Expected prompt_embeddings shape {default_prompt.shape}, got {prompt_np.shape}"
        )
    if pooled_np.shape != default_pooled.shape:
        raise ValueError(
            f"Expected pooled_prompt shape {default_pooled.shape}, got {pooled_np.shape}"
        )

    return np.array(prompt_np, copy=True), np.array(pooled_np, copy=True), "custom"


def _prompt_inputs_from_single_text_hidden(
    text_hidden: np.ndarray,
    *,
    target_seq_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.asarray(text_hidden, dtype=np.float32)
    if hidden.ndim != 3:
        raise ValueError(f"Expected 3D text hidden tensor, got shape {hidden.shape}")
    if target_seq_len is not None:
        length = max(1, int(target_seq_len))
        if hidden.shape[1] >= length:
            hidden = hidden[:, :length, :]
        else:
            repeats = (length + hidden.shape[1] - 1) // hidden.shape[1]
            hidden = np.tile(hidden, (1, repeats, 1))[:, :length, :]
    return hidden, np.mean(hidden, axis=1, dtype=np.float32)


def _build_single_encoder_prompt_inputs(
    *,
    default_input_ids: np.ndarray,
    provided_input_ids: np.ndarray | None,
    target_seq_len: int,
    encode_hidden: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    text_ids = _resolve_token_ids(
        default_input_ids,
        provided_input_ids,
        name="input_ids",
    )
    text_hidden = np.asarray(encode_hidden(text_ids), dtype=np.float32)
    prompt_embeddings, pooled_prompt = _prompt_inputs_from_single_text_hidden(
        text_hidden,
        target_seq_len=target_seq_len,
    )
    return text_ids, text_hidden, prompt_embeddings, pooled_prompt


def _transpile_demo_spec(
    root: Path,
    *,
    submodule: FluxSubmodule,
    spec: ExportSpec,
) -> tuple[Path, Path, Any, Any]:
    onnx_path = export_flux_submodule_onnx(
        submodule,
        root / f"flux_{submodule}.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
    )
    generated_dir = root / f"generated_flux_{submodule}_jax"
    manifest = transpile_onnx(
        str(onnx_path),
        "jax",
        str(generated_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )
    module_name = "_".join(
        part.replace("-", "_").replace(".", "_") for part in generated_dir.parts[-2:]
    )
    module = load_generated_module(
        generated_dir / "model_jax.py",
        module_name=module_name,
    )
    params = module.load_weights(str(manifest.weights_file))
    return onnx_path, generated_dir, module, params


def _transpile_bridge_spec(
    root: Path,
    *,
    submodule: FluxSubmodule,
    spec: ExportSpec,
) -> tuple[Path, Path, Any, Any]:
    onnx_path = export_flux_submodule_onnx(
        submodule,
        root / f"{submodule}.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
    )
    unsupported = flux_source.load_unsupported_onnx_ops(onnx_path)
    if unsupported:
        raise ValueError(f"Unsupported ONNX ops for {submodule}: {unsupported}")

    generated_dir = root / f"generated_{submodule}_jax"
    manifest = transpile_onnx(
        str(onnx_path),
        "jax",
        str(generated_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )
    module = load_generated_module(
        generated_dir / "model_jax.py",
        module_name=f"generated_flux_{submodule}_bridge_jax",
    )
    params = module.load_weights(str(manifest.weights_file))
    return onnx_path, generated_dir, module, params


def run_export_spec_pytorch_reference(spec: ExportSpec) -> dict[str, list[list[int]]]:
    import torch

    with torch.no_grad():
        output = spec.module(*spec.sample_inputs)

    tensors: list[Any] = []
    if hasattr(output, "shape"):
        tensors = [output]
    elif isinstance(output, tuple | list):
        tensors = [value for value in output if hasattr(value, "shape")]
    elif isinstance(output, dict):
        tensors = [value for value in output.values() if hasattr(value, "shape")]

    return {
        "output_shapes": [
            [int(dim) for dim in tuple(getattr(value, "shape", ()))] for value in tensors
        ],
    }


def run_flux_jax_demo(
    out_root: str | Path = "examples/flux/out",
    *,
    steps: int = 4,
    seed: int = DEMO_SEED,
    image_size: int = DEMO_IMAGE_SIZE,
    blend: float = 0.4,
    prompt_embeddings: np.ndarray | None = None,
    pooled_prompt: np.ndarray | None = None,
) -> dict[str, str | float | int]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    demo_inputs = build_demo_inputs(seed=seed)
    latents_np = demo_inputs["latents"].detach().cpu().numpy()
    prompt_np, pooled_np, prompt_source = _resolve_prompt_inputs(
        demo_inputs["prompt_embeddings"].detach().cpu().numpy(),
        demo_inputs["pooled_prompt"].detach().cpu().numpy(),
        prompt_embeddings=prompt_embeddings,
        pooled_prompt=pooled_prompt,
    )

    specs = build_demo_export_specs(seed=seed, image_size=image_size)
    transformer_spec = specs["transformer"]
    vae_spec = specs["vae_decoder"]

    (
        transformer_onnx,
        transformer_dir,
        transformer_module,
        transformer_params,
    ) = _transpile_demo_spec(
        root,
        submodule="transformer",
        spec=transformer_spec,
    )
    vae_onnx, vae_dir, vae_module, vae_params = _transpile_demo_spec(
        root,
        submodule="vae_decoder",
        spec=vae_spec,
    )

    current = np.array(latents_np, copy=True)
    for _ in range(int(steps)):
        denoised = np.asarray(
            transformer_module.forward(
                transformer_params,
                {
                    "latents": current,
                    "prompt_embeddings": prompt_np,
                    "pooled_prompt": pooled_np,
                },
            )["denoised"]
        )
        current = ((1.0 - blend) * current) + (blend * denoised)

    generated = np.asarray(vae_module.forward(vae_params, {"latents": current})["image"])

    reference_latents = demo_inputs["latents"]
    prompt = torch.from_numpy(prompt_np.copy())
    pooled = torch.from_numpy(pooled_np.copy())
    torch_current = reference_latents.clone()
    for _ in range(int(steps)):
        torch_denoised = transformer_spec.module(torch_current, prompt, pooled)
        torch_current = ((1.0 - blend) * torch_current) + (blend * torch_denoised)
    expected = vae_spec.module(torch_current).detach().cpu().numpy()

    image_path = root / "flux_jax_demo.png"
    _save_png(generated, image_path)

    diff = np.abs(generated - expected)
    rgb = _to_image_array(generated)
    return {
        "transformer_onnx": str(transformer_onnx),
        "vae_onnx": str(vae_onnx),
        "transformer_module": str(transformer_dir / "model_jax.py"),
        "vae_module": str(vae_dir / "model_jax.py"),
        "image_path": str(image_path),
        "steps": int(steps),
        "image_size": int(rgb.shape[0]),
        "prompt_source": prompt_source,
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "pixel_std": float(np.std(rgb)),
    }


def run_flux_pytorch_prompt_to_image_demo(
    out_root: str | Path = "examples/flux/out",
    *,
    steps: int = 4,
    seed: int = DEMO_SEED,
    image_size: int = DEMO_IMAGE_SIZE,
    blend: float = 0.4,
    input_ids: np.ndarray | None = None,
) -> dict[str, str | float | int]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    specs = build_demo_export_specs(seed=seed, image_size=image_size)
    text_spec = specs["text_encoder"]
    transformer_spec = specs["transformer"]
    vae_spec = specs["vae_decoder"]

    text_ids, _, prompt_embeddings, pooled_prompt = _build_single_encoder_prompt_inputs(
        default_input_ids=text_spec.sample_inputs[0].detach().cpu().numpy(),
        provided_input_ids=input_ids,
        target_seq_len=int(transformer_spec.sample_inputs[0].shape[1]),
        encode_hidden=lambda token_ids: (
            text_spec.module(torch.from_numpy(token_ids.copy())).detach().cpu().numpy()
        ),
    )

    demo_inputs = build_demo_inputs(seed=seed)
    latents = demo_inputs["latents"]
    prompt = torch.from_numpy(prompt_embeddings.copy())
    pooled = torch.from_numpy(pooled_prompt.copy())

    current = latents.clone()
    for _ in range(int(steps)):
        denoised = transformer_spec.module(current, prompt, pooled)
        current = ((1.0 - blend) * current) + (blend * denoised)
    generated = vae_spec.module(current).detach().cpu().numpy()

    image_path = root / "flux_pytorch_reference.png"
    _save_png(generated, image_path)
    rgb = _to_image_array(generated)
    return {
        "image_path": str(image_path),
        "steps": int(steps),
        "image_size": int(rgb.shape[0]),
        "prompt_source": "text_encoder",
        "token_length": int(text_ids.shape[1]),
        "pixel_std": float(np.std(rgb)),
    }


def run_flux_pytorch_dummy_flux2_demo(
    out_root: str | Path = "examples/flux/out",
    *,
    steps: int = 2,
    seed: int = DEMO_SEED,
    image_size: int = flux_source.DUMMY_FLUX2_IMAGE_SIZE,
    blend: float = 0.4,
    input_ids: np.ndarray | None = None,
) -> dict[str, str | float | int]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    specs = build_dummy_flux2_export_specs(seed=seed, image_size=image_size)
    text_spec = specs["text_encoder"]
    transformer_spec = specs["transformer"]
    vae_spec = specs["vae_decoder"]

    text_ids, _, prompt_embeddings, pooled_prompt = _build_single_encoder_prompt_inputs(
        default_input_ids=text_spec.sample_inputs[0].detach().cpu().numpy(),
        provided_input_ids=input_ids,
        target_seq_len=int(transformer_spec.sample_inputs[0].shape[1]),
        encode_hidden=lambda token_ids: (
            text_spec.module(torch.from_numpy(token_ids.copy())).detach().cpu().numpy()
        ),
    )

    current = transformer_spec.sample_inputs[0].detach().clone()
    prompt = torch.from_numpy(prompt_embeddings.copy())
    pooled = torch.from_numpy(pooled_prompt.copy())
    for _ in range(int(steps)):
        denoised = transformer_spec.module(current, prompt, pooled)
        current = ((1.0 - blend) * current) + (blend * denoised)
    generated = vae_spec.module(current).detach().cpu().numpy()

    image_path = root / "flux_pytorch_dummy_flux2_reference.png"
    _save_png(generated, image_path)
    rgb = _to_image_array(generated)
    return {
        "image_path": str(image_path),
        "steps": int(steps),
        "image_size": int(rgb.shape[0]),
        "prompt_source": "dummy_flux2_text_encoder",
        "token_length": int(text_ids.shape[1]),
        "pixel_std": float(np.std(rgb)),
    }


def run_flux_jax_dummy_flux2_demo(
    out_root: str | Path = "examples/flux/out",
    *,
    steps: int = 2,
    seed: int = DEMO_SEED,
    image_size: int = flux_source.DUMMY_FLUX2_IMAGE_SIZE,
    blend: float = 0.4,
    input_ids: np.ndarray | None = None,
) -> dict[str, str | float | int]:
    import torch

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    pytorch_reference = run_flux_pytorch_dummy_flux2_demo(
        root,
        steps=steps,
        seed=seed,
        image_size=image_size,
        blend=blend,
        input_ids=input_ids,
    )

    specs = build_dummy_flux2_export_specs(seed=seed, image_size=image_size)
    text_spec = specs["text_encoder"]
    transformer_spec = specs["transformer"]
    vae_spec = specs["vae_decoder"]

    text_onnx, text_dir, text_module, text_params = _transpile_demo_spec(
        root,
        submodule="text_encoder",
        spec=text_spec,
    )
    (
        transformer_onnx,
        transformer_dir,
        transformer_module,
        transformer_params,
    ) = _transpile_demo_spec(
        root,
        submodule="transformer",
        spec=transformer_spec,
    )
    vae_onnx, vae_dir, vae_module, vae_params = _transpile_demo_spec(
        root,
        submodule="vae_decoder",
        spec=vae_spec,
    )

    text_ids, text_hidden, prompt_embeddings, pooled_prompt = _build_single_encoder_prompt_inputs(
        default_input_ids=text_spec.sample_inputs[0].detach().cpu().numpy(),
        provided_input_ids=input_ids,
        target_seq_len=int(transformer_spec.sample_inputs[0].shape[1]),
        encode_hidden=lambda token_ids: text_module.forward(text_params, {"input_ids": token_ids})[
            "hidden"
        ],
    )

    current = np.array(transformer_spec.sample_inputs[0].detach().cpu().numpy(), copy=True)
    for _ in range(int(steps)):
        denoised = np.asarray(
            transformer_module.forward(
                transformer_params,
                {
                    "latents": current,
                    "prompt_embeddings": prompt_embeddings,
                    "pooled_prompt": pooled_prompt,
                },
            )["denoised"]
        )
        current = ((1.0 - blend) * current) + (blend * denoised)

    generated = np.asarray(vae_module.forward(vae_params, {"latents": current})["image"])

    _, expected_text_hidden, expected_prompt_embeddings, expected_pooled_prompt = (
        _build_single_encoder_prompt_inputs(
            default_input_ids=text_spec.sample_inputs[0].detach().cpu().numpy(),
            provided_input_ids=text_ids,
            target_seq_len=int(transformer_spec.sample_inputs[0].shape[1]),
            encode_hidden=lambda token_ids: (
                text_spec.module(torch.from_numpy(token_ids.copy())).detach().cpu().numpy()
            ),
        )
    )
    text_diff = np.abs(text_hidden - expected_text_hidden)
    torch_current = transformer_spec.sample_inputs[0].detach().clone()
    prompt = torch.from_numpy(expected_prompt_embeddings.copy())
    pooled = torch.from_numpy(expected_pooled_prompt.copy())
    for _ in range(int(steps)):
        torch_denoised = transformer_spec.module(torch_current, prompt, pooled)
        torch_current = ((1.0 - blend) * torch_current) + (blend * torch_denoised)
    expected = vae_spec.module(torch_current).detach().cpu().numpy()

    image_path = root / "flux_jax_dummy_flux2.png"
    _save_png(generated, image_path)

    diff = np.abs(generated - expected)
    rgb = _to_image_array(generated)
    return {
        "text_encoder_onnx": str(text_onnx),
        "transformer_onnx": str(transformer_onnx),
        "vae_onnx": str(vae_onnx),
        "text_encoder_module": str(text_dir / "model_jax.py"),
        "transformer_module": str(transformer_dir / "model_jax.py"),
        "vae_module": str(vae_dir / "model_jax.py"),
        "image_path": str(image_path),
        "pytorch_reference_image_path": str(pytorch_reference["image_path"]),
        "steps": int(steps),
        "image_size": int(rgb.shape[0]),
        "prompt_source": "dummy_flux2_text_encoder",
        "text_encoder_max_abs": float(np.max(text_diff)),
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "pixel_std": float(np.std(rgb)),
    }


def run_flux_pytorch_tiny_config_e2e_demo(
    out_root: str | Path = "examples/flux/out",
    *,
    model_id: str | None = None,
    input_ids: np.ndarray | None = None,
) -> dict[str, str | float | int]:
    import torch

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    specs = build_tiny_flux2_export_specs(model_id=model_id)
    text_spec = specs["text_encoder"]
    transformer_spec = specs["transformer"]
    vae_spec = specs["vae_decoder"]

    text_ids, _, encoder_hidden_states, pooled_prompt = _build_single_encoder_prompt_inputs(
        default_input_ids=text_spec.sample_inputs[0].detach().cpu().numpy(),
        provided_input_ids=input_ids,
        target_seq_len=int(transformer_spec.sample_inputs[1].shape[1]),
        encode_hidden=lambda token_ids: (
            text_spec.module(torch.from_numpy(token_ids.copy())).detach().float().cpu().numpy()
        ),
    )

    transformer_inputs = {
        name: tensor.detach().clone()
        for name, tensor in zip(
            transformer_spec.input_names,
            transformer_spec.sample_inputs,
            strict=True,
        )
    }
    transformer_inputs["encoder_hidden_states"] = torch.from_numpy(encoder_hidden_states.copy())
    if "pooled_projections" in transformer_inputs:
        transformer_inputs["pooled_projections"] = torch.from_numpy(pooled_prompt.copy())

    denoised = (
        transformer_spec.module(
            *(transformer_inputs[name] for name in transformer_spec.input_names)
        )
        .detach()
        .float()
    )
    latent_channels = int(vae_spec.sample_inputs[0].shape[1])
    spatial_shape = tuple(int(dim) for dim in vae_spec.sample_inputs[0].shape[2:])
    if int(denoised.shape[2]) != latent_channels:
        raise ValueError(
            "Tiny transformer output channels do not match the tiny VAE latent channels."
        )
    latents = denoised.transpose(1, 2).reshape(1, latent_channels, *spatial_shape)
    generated = vae_spec.module(latents).detach().cpu().numpy().astype(np.float32)

    image_path = root / "flux_pytorch_tiny_config_e2e.png"
    _save_png(generated, image_path)
    rgb = _to_image_array(generated)
    return {
        "image_path": str(image_path),
        "image_size": int(rgb.shape[0]),
        "prompt_source": "tiny_config_text_encoder",
        "token_length": int(text_ids.shape[1]),
        "pixel_std": float(np.std(rgb)),
    }


def run_flux_jax_tiny_config_e2e_demo(
    out_root: str | Path = "examples/flux/out",
    *,
    model_id: str | None = None,
    input_ids: np.ndarray | None = None,
) -> dict[str, str | float | int]:
    import torch

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    pytorch_reference = run_flux_pytorch_tiny_config_e2e_demo(
        root,
        model_id=model_id,
        input_ids=input_ids,
    )

    specs = build_tiny_flux2_export_specs(model_id=model_id)
    text_spec = specs["text_encoder"]
    transformer_spec = specs["transformer"]
    vae_spec = specs["vae_decoder"]

    text_onnx, text_dir, text_module, text_params = _transpile_bridge_spec(
        root / "tiny_config_text",
        submodule="text_encoder",
        spec=text_spec,
    )
    (
        transformer_onnx,
        transformer_dir,
        transformer_module,
        transformer_params,
    ) = _transpile_bridge_spec(
        root / "tiny_config_transformer",
        submodule="transformer",
        spec=transformer_spec,
    )
    vae_onnx, vae_dir, vae_module, vae_params = _transpile_bridge_spec(
        root / "tiny_config_vae",
        submodule="vae_decoder",
        spec=vae_spec,
    )

    text_ids, _, encoder_hidden_states, pooled_prompt = _build_single_encoder_prompt_inputs(
        default_input_ids=text_spec.sample_inputs[0].detach().cpu().numpy(),
        provided_input_ids=input_ids,
        target_seq_len=int(transformer_spec.sample_inputs[1].shape[1]),
        encode_hidden=lambda token_ids: text_module.forward(text_params, {"input_ids": token_ids})[
            "hidden"
        ],
    )
    transformer_inputs = {
        name: np.asarray(tensor.detach().cpu().numpy())
        for name, tensor in zip(
            transformer_spec.input_names,
            transformer_spec.sample_inputs,
            strict=True,
        )
    }
    transformer_inputs["encoder_hidden_states"] = encoder_hidden_states.astype(np.float32)
    if "pooled_projections" in transformer_inputs:
        transformer_inputs["pooled_projections"] = pooled_prompt.astype(np.float32)

    denoised = np.asarray(
        transformer_module.forward(transformer_params, transformer_inputs)["denoised"],
        dtype=np.float32,
    )
    latent_channels = int(vae_spec.sample_inputs[0].shape[1])
    spatial_shape = tuple(int(dim) for dim in vae_spec.sample_inputs[0].shape[2:])
    latents = np.transpose(denoised, (0, 2, 1)).reshape(1, latent_channels, *spatial_shape)
    generated = np.asarray(
        vae_module.forward(vae_params, {"latents": latents})["image"],
        dtype=np.float32,
    )

    _, _, expected_encoder_hidden_states, expected_pooled_prompt = (
        _build_single_encoder_prompt_inputs(
            default_input_ids=text_spec.sample_inputs[0].detach().cpu().numpy(),
            provided_input_ids=text_ids,
            target_seq_len=int(transformer_spec.sample_inputs[1].shape[1]),
            encode_hidden=lambda token_ids: (
                text_spec.module(torch.from_numpy(token_ids.copy())).detach().float().cpu().numpy()
            ),
        )
    )
    expected_inputs = {
        name: tensor.detach().clone()
        for name, tensor in zip(
            transformer_spec.input_names,
            transformer_spec.sample_inputs,
            strict=True,
        )
    }
    expected_inputs["encoder_hidden_states"] = torch.from_numpy(
        expected_encoder_hidden_states.copy()
    )
    if "pooled_projections" in expected_inputs:
        expected_inputs["pooled_projections"] = torch.from_numpy(expected_pooled_prompt.copy())
    expected_denoised = (
        transformer_spec.module(*(expected_inputs[name] for name in transformer_spec.input_names))
        .detach()
        .float()
    )
    expected_latents = expected_denoised.transpose(1, 2).reshape(
        1,
        latent_channels,
        *spatial_shape,
    )
    expected = vae_spec.module(expected_latents).detach().cpu().numpy().astype(np.float32)

    image_path = root / "flux_jax_tiny_config_e2e.png"
    _save_png(generated, image_path)
    diff = np.abs(generated - expected)
    rgb = _to_image_array(generated)
    return {
        "text_encoder_onnx": str(text_onnx),
        "transformer_onnx": str(transformer_onnx),
        "vae_onnx": str(vae_onnx),
        "text_encoder_module": str(text_dir / "model_jax.py"),
        "transformer_module": str(transformer_dir / "model_jax.py"),
        "vae_module": str(vae_dir / "model_jax.py"),
        "image_path": str(image_path),
        "pytorch_reference_image_path": str(pytorch_reference["image_path"]),
        "image_size": int(rgb.shape[0]),
        "prompt_source": "tiny_config_text_encoder",
        "token_length": int(text_ids.shape[1]),
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "pixel_std": float(np.std(rgb)),
    }


def run_flux_jax_prompt_to_image_demo(
    out_root: str | Path = "examples/flux/out",
    *,
    steps: int = 4,
    seed: int = DEMO_SEED,
    image_size: int = DEMO_IMAGE_SIZE,
    blend: float = 0.4,
    input_ids: np.ndarray | None = None,
) -> dict[str, str | float | int]:
    import torch

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    pytorch_reference = run_flux_pytorch_prompt_to_image_demo(
        root,
        steps=steps,
        seed=seed,
        image_size=image_size,
        blend=blend,
        input_ids=input_ids,
    )

    specs = build_demo_export_specs(seed=seed, image_size=image_size)
    text_spec = specs["text_encoder"]

    text_onnx, text_dir, text_module, text_params = _transpile_demo_spec(
        root,
        submodule="text_encoder",
        spec=text_spec,
    )

    text_ids, text_hidden, prompt_embeddings, pooled_prompt = _build_single_encoder_prompt_inputs(
        default_input_ids=text_spec.sample_inputs[0].detach().cpu().numpy(),
        provided_input_ids=input_ids,
        target_seq_len=int(specs["transformer"].sample_inputs[0].shape[1]),
        encode_hidden=lambda token_ids: text_module.forward(text_params, {"input_ids": token_ids})[
            "hidden"
        ],
    )

    _, expected_text_hidden, _, _ = _build_single_encoder_prompt_inputs(
        default_input_ids=text_spec.sample_inputs[0].detach().cpu().numpy(),
        provided_input_ids=text_ids,
        target_seq_len=int(specs["transformer"].sample_inputs[0].shape[1]),
        encode_hidden=lambda token_ids: (
            text_spec.module(torch.from_numpy(token_ids.copy())).detach().cpu().numpy()
        ),
    )
    text_diff = np.abs(text_hidden - expected_text_hidden)

    result = run_flux_jax_demo(
        root,
        steps=steps,
        seed=seed,
        image_size=image_size,
        blend=blend,
        prompt_embeddings=prompt_embeddings,
        pooled_prompt=pooled_prompt,
    )
    result.update(
        {
            "text_encoder_onnx": str(text_onnx),
            "text_encoder_module": str(text_dir / "model_jax.py"),
            "prompt_source": "text_encoder",
            "pytorch_reference_image_path": str(pytorch_reference["image_path"]),
            "pytorch_reference_pixel_std": float(pytorch_reference["pixel_std"]),
            "text_encoder_max_abs": float(np.max(text_diff)),
            "token_length": int(text_ids.shape[1]),
        }
    )
    return result


def run_flux_jax_reduced_checkpoint_bridge_demo(
    out_root: str | Path = "examples/flux/out",
    *,
    model_id: str | None = None,
) -> dict[str, str | float | int]:
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    snapshot = flux_source.resolve_flux_snapshot(model_id)
    for submodule in flux_source.real_submodule_export_order(snapshot=snapshot, model_id=model_id):
        if not flux_source.snapshot_has_submodule_weights(snapshot, submodule):
            raise FileNotFoundError(f"Missing {submodule} weights under snapshot: {snapshot}")

    text_spec = flux_source._real_export_spec(
        "text_encoder",
        model_id=model_id,
        load_weights=False,
    )
    text_seq_len = int(text_spec.sample_inputs[0].shape[1])
    transformer_spec = flux_source._real_export_spec(
        "transformer",
        model_id=model_id,
        load_weights=False,
        transformer_image_seq_len=4,
        transformer_text_seq_len=text_seq_len,
    )
    vae_spec = flux_source._real_export_spec("vae_decoder", model_id=model_id)

    text_onnx, text_dir, text_module, text_params = _transpile_bridge_spec(
        root / "text",
        submodule="text_encoder",
        spec=text_spec,
    )
    (
        transformer_onnx,
        transformer_dir,
        transformer_module,
        transformer_params,
    ) = _transpile_bridge_spec(
        root / "transformer",
        submodule="transformer",
        spec=transformer_spec,
    )
    vae_onnx, vae_dir, vae_module, vae_params = _transpile_bridge_spec(
        root / "vae",
        submodule="vae_decoder",
        spec=vae_spec,
    )

    # Validate the PyTorch reference path first, then compare JAX against it.
    text_hidden_pt = text_spec.module(*text_spec.sample_inputs).detach().float()
    transformer_inputs_pt = {
        name: tensor.detach().clone()
        for name, tensor in zip(
            transformer_spec.input_names,
            transformer_spec.sample_inputs,
            strict=True,
        )
    }
    transformer_inputs_pt["encoder_hidden_states"] = text_hidden_pt
    if "pooled_projections" in transformer_inputs_pt:
        transformer_inputs_pt["pooled_projections"] = text_hidden_pt.mean(dim=1)
    denoised_pt = (
        transformer_spec.module(
            *(transformer_inputs_pt[name] for name in transformer_spec.input_names)
        )
        .detach()
        .float()
    )
    latent_channels = int(vae_spec.sample_inputs[0].shape[1])
    if int(denoised_pt.shape[2]) != latent_channels:
        raise ValueError(
            "Reduced transformer output channels do not match the real VAE latent channels."
        )
    latents_pt = denoised_pt.transpose(1, 2).reshape(1, latent_channels, 2, 2)
    reference = vae_spec.module(latents_pt).detach().cpu().numpy().astype(np.float32)
    pytorch_reference_image_path = root / "flux_pytorch_reduced_checkpoint_bridge.png"
    _save_png(reference, pytorch_reference_image_path)

    input_ids = np.asarray(text_spec.sample_inputs[0].detach().cpu().numpy(), dtype=np.int64)
    text_hidden_jax = np.asarray(
        text_module.forward(text_params, {"input_ids": input_ids})["hidden"],
        dtype=np.float32,
    )
    transformer_inputs_jax = {
        name: np.asarray(tensor.detach().cpu().numpy())
        for name, tensor in zip(
            transformer_spec.input_names,
            transformer_spec.sample_inputs,
            strict=True,
        )
    }
    transformer_inputs_jax["encoder_hidden_states"] = text_hidden_jax
    if "pooled_projections" in transformer_inputs_jax:
        transformer_inputs_jax["pooled_projections"] = np.mean(
            text_hidden_jax,
            axis=1,
            dtype=np.float32,
        )
    denoised_jax = np.asarray(
        transformer_module.forward(transformer_params, transformer_inputs_jax)["denoised"],
        dtype=np.float32,
    )
    latents_jax = np.transpose(denoised_jax, (0, 2, 1)).reshape(1, latent_channels, 2, 2)
    generated = np.asarray(
        vae_module.forward(vae_params, {"latents": latents_jax})["image"],
        dtype=np.float32,
    )
    image_path = root / "flux_jax_reduced_checkpoint_bridge.png"
    _save_png(generated, image_path)

    diff = np.abs(generated - reference)
    rgb = _to_image_array(generated)
    return {
        "text_encoder_onnx": str(text_onnx),
        "transformer_onnx": str(transformer_onnx),
        "vae_onnx": str(vae_onnx),
        "text_encoder_module": str(text_dir / "model_jax.py"),
        "transformer_module": str(transformer_dir / "model_jax.py"),
        "vae_module": str(vae_dir / "model_jax.py"),
        "image_path": str(image_path),
        "pytorch_reference_image_path": str(pytorch_reference_image_path),
        "image_size": int(rgb.shape[0]),
        "prompt_source": "reduced_checkpoint_bridge",
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "pixel_std": float(np.std(rgb)),
    }


def inspect_flux_checkpoint_assets(
    out_root: str | Path = "examples/flux/out",
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    report_path = root / "flux_checkpoint_asset_report.json"
    resolved_model_id = model_id or flux_source.DEFAULT_REAL_FLUX_MODEL_ID
    try:
        snapshot = flux_source.resolve_flux_snapshot(model_id)
    except FileNotFoundError as exc:
        submodule_order = flux_source.real_submodule_export_order(model_id=resolved_model_id)
        report: dict[str, Any] = {
            "model_id": resolved_model_id,
            "snapshot": None,
            "report_path": str(report_path),
            "submodule_order": list(submodule_order),
            "ready_count": 0,
            "missing_count": len(submodule_order),
            "error": str(exc),
            "submodules": {
                submodule: {
                    "status": "missing",
                    "component_root": None,
                    "config_path": None,
                    "has_config": False,
                    "has_weights": False,
                    "metadata": {},
                }
                for submodule in submodule_order
            },
        }
        report_write_error = _write_json_report(report_path, report)
        if report_write_error is not None:
            report["report_write_error"] = report_write_error
        return report

    submodule_order = flux_source.real_submodule_export_order(snapshot=snapshot, model_id=model_id)
    submodule_reports: dict[str, dict[str, Any]] = {}
    ready_count = 0
    missing_count = 0

    for submodule in submodule_order:
        component_name = "vae" if submodule == "vae_decoder" else submodule
        component_root = snapshot / component_name
        config_path = component_root / "config.json"
        config_payload = _read_json_dict(config_path)
        has_weights = flux_source.snapshot_has_submodule_weights(snapshot, submodule)
        metadata: dict[str, Any] = {}
        if config_payload is not None:
            key_map: dict[str, tuple[str, ...]] = {
                "vae_decoder": ("_class_name", "latent_channels"),
                "transformer": (
                    "_class_name",
                    "in_channels",
                    "joint_attention_dim",
                    "pooled_projection_dim",
                    "guidance_embeds",
                    "axes_dims_rope",
                ),
                "text_encoder": (
                    "_class_name",
                    "model_type",
                    "hidden_size",
                    "max_position_embeddings",
                ),
            }
            for key in key_map[submodule]:
                if key in config_payload:
                    metadata[key] = config_payload[key]

        status = "ready" if config_payload is not None and has_weights else "missing"
        if status == "ready":
            ready_count += 1
        else:
            missing_count += 1

        submodule_reports[submodule] = {
            "status": status,
            "component_root": str(component_root),
            "config_path": str(config_path) if config_payload is not None else None,
            "has_config": config_payload is not None,
            "has_weights": has_weights,
            "metadata": metadata,
        }

    report = {
        "model_id": resolved_model_id,
        "snapshot": str(snapshot),
        "report_path": str(report_path),
        "submodule_order": list(submodule_order),
        "ready_count": ready_count,
        "missing_count": missing_count,
        "error": None,
        "submodules": submodule_reports,
    }
    report_write_error = _write_json_report(report_path, report)
    if report_write_error is not None:
        report["report_write_error"] = report_write_error
    return report


def prepare_flux_jax_checkpoint_artifacts(
    out_root: str | Path = "examples/flux/out",
    *,
    model_id: str | None = None,
    checkpoint_submodule: FluxSubmodule | None = None,
    checkpoint_graph_only: bool = False,
    checkpoint_reduced_shapes: bool = False,
) -> dict[str, Any]:
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    report_path = root / "flux_checkpoint_jax_report.json"
    resolved_model_id = model_id or flux_source.DEFAULT_REAL_FLUX_MODEL_ID
    try:
        snapshot = flux_source.resolve_flux_snapshot(model_id)
    except FileNotFoundError as exc:
        submodule_order = flux_source.real_submodule_export_order(model_id=resolved_model_id)
        if checkpoint_submodule is not None:
            if checkpoint_submodule not in submodule_order:
                raise ValueError(
                    f"Submodule {checkpoint_submodule!r} is not part of the real FLUX topology for "
                    f"{resolved_model_id!r}."
                )
            submodule_order = (checkpoint_submodule,)
        report: dict[str, Any] = {
            "model_id": resolved_model_id,
            "snapshot": None,
            "report_path": str(report_path),
            "submodule_order": list(submodule_order),
            "ready_count": 0,
            "blocked_count": 0,
            "missing_count": len(submodule_order),
            "checkpoint_graph_only": checkpoint_graph_only,
            "checkpoint_reduced_shapes": checkpoint_reduced_shapes,
            "error": str(exc),
            "submodules": {
                submodule: {
                    "status": "missing",
                    "onnx_path": None,
                    "generated_dir": None,
                    "weights_file": None,
                    "reason": str(exc),
                    "unsupported_ops": [],
                    "used_reduced_config": False,
                }
                for submodule in submodule_order
            },
        }
        report_write_error = _write_json_report(report_path, report)
        if report_write_error is not None:
            report["report_write_error"] = report_write_error
        return report

    submodule_order = flux_source.real_submodule_export_order(snapshot=snapshot, model_id=model_id)
    if checkpoint_submodule is not None:
        if checkpoint_submodule not in submodule_order:
            raise ValueError(
                f"Submodule {checkpoint_submodule!r} is not part of the real FLUX topology for "
                f"{resolved_model_id!r}."
            )
        submodule_order = (checkpoint_submodule,)
    submodule_reports: dict[str, dict[str, Any]] = {}
    ready_count = 0
    blocked_count = 0
    missing_count = 0

    for submodule in submodule_order:
        entry: dict[str, Any] = {
            "status": "missing",
            "onnx_path": None,
            "generated_dir": None,
            "weights_file": None,
            "reason": None,
            "reference_output_shapes": [],
            "unsupported_ops": [],
            "used_reduced_config": False,
        }
        if not flux_source.snapshot_has_submodule_weights(snapshot, submodule):
            entry["reason"] = f"Missing {submodule} weights under snapshot: {snapshot}"
            missing_count += 1
            submodule_reports[submodule] = entry
            continue

        try:
            real_export_kwargs: dict[str, Any] = {}
            if submodule == "transformer":
                if checkpoint_reduced_shapes:
                    real_export_kwargs.update(
                        {
                            "transformer_image_seq_len": flux_source.CHECKPOINT_SMOKE_IMAGE_SEQ_LEN,
                            "transformer_text_seq_len": flux_source.CHECKPOINT_SMOKE_TEXT_SEQ_LEN,
                        }
                    )
                if checkpoint_graph_only or checkpoint_reduced_shapes:
                    real_export_kwargs["load_weights"] = False
                    entry["used_reduced_config"] = True
            if submodule == "text_encoder" and (checkpoint_graph_only or checkpoint_reduced_shapes):
                real_export_kwargs["load_weights"] = False
                entry["used_reduced_config"] = True
            spec = flux_source._real_export_spec(
                submodule,
                model_id=model_id,
                **real_export_kwargs,
            )
        except ModuleNotFoundError as exc:
            entry["status"] = "blocked"
            entry["reason"] = str(exc)
            blocked_count += 1
            submodule_reports[submodule] = entry
            continue

        try:
            reference = run_export_spec_pytorch_reference(spec)
        except Exception as exc:
            entry["status"] = "blocked"
            entry["reason"] = f"PyTorch reference failed: {exc}"
            blocked_count += 1
            submodule_reports[submodule] = entry
            continue

        entry["reference_output_shapes"] = reference["output_shapes"]

        try:
            onnx_path = flux_source.export_flux_submodule_onnx(
                submodule,
                root / f"flux_{submodule}_checkpoint.onnx",
                module=spec.module,
                sample_inputs=spec.sample_inputs,
                input_names=spec.input_names,
                output_names=spec.output_names,
                export_params=not checkpoint_graph_only,
            )
        except Exception as exc:
            entry["status"] = "blocked"
            entry["reason"] = f"ONNX export failed: {exc}"
            blocked_count += 1
            submodule_reports[submodule] = entry
            continue

        entry["onnx_path"] = str(onnx_path)
        unsupported = list(flux_source.load_unsupported_onnx_ops(onnx_path))
        entry["unsupported_ops"] = unsupported
        if unsupported:
            entry["status"] = "blocked"
            entry["reason"] = f"Unsupported ONNX ops: {unsupported}"
            blocked_count += 1
            submodule_reports[submodule] = entry
            continue

        if checkpoint_graph_only:
            entry["status"] = "ready"
            ready_count += 1
            submodule_reports[submodule] = entry
            continue

        generated_dir = root / f"generated_flux_{submodule}_checkpoint_jax"
        try:
            manifest = transpile_onnx(
                str(onnx_path),
                "jax",
                str(generated_dir),
                config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
            )
        except Exception as exc:
            entry["status"] = "blocked"
            entry["reason"] = f"JAX transpile failed: {exc}"
            blocked_count += 1
            submodule_reports[submodule] = entry
            continue

        entry["status"] = "ready"
        entry["generated_dir"] = str(generated_dir)
        entry["weights_file"] = str(manifest.weights_file)
        ready_count += 1
        submodule_reports[submodule] = entry

    report: dict[str, Any] = {
        "model_id": resolved_model_id,
        "snapshot": str(snapshot),
        "report_path": str(report_path),
        "submodule_order": list(submodule_order),
        "ready_count": ready_count,
        "blocked_count": blocked_count,
        "missing_count": missing_count,
        "checkpoint_graph_only": checkpoint_graph_only,
        "checkpoint_reduced_shapes": checkpoint_reduced_shapes,
        "error": None,
        "submodules": submodule_reports,
    }
    report_write_error = _write_json_report(report_path, report)
    if report_write_error is not None:
        report["report_write_error"] = report_write_error
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export reduced FLUX submodules, transpile them to JAX, and render a smoke image."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        default="examples/flux/out",
        help="Directory for exported ONNX and generated JAX artifacts.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of reduced denoising iterations to run.",
    )
    parser.add_argument(
        "--prompt-npz",
        dest="prompt_npz",
        default=None,
        help=(
            "Optional .npz file containing 'prompt_embeddings' and 'pooled_prompt' arrays for the "
            "reduced JAX image path."
        ),
    )
    parser.add_argument(
        "--use-text-encoders",
        action="store_true",
        help="Run the reduced prompt-to-image path using transpiled FLUX text encoders.",
    )
    parser.add_argument(
        "--use-dummy-flux2",
        action="store_true",
        help=(
            "Run the tiny synthetic FLUX.2-style single-text-encoder proof path with random "
            "weights and tiny tensors."
        ),
    )
    parser.add_argument(
        "--use-tiny-config",
        action="store_true",
        help=(
            "Run the deterministic tiny FLUX.2 config proof path with constant weights. "
            "This validates the PyTorch reference first, then runs generated JAX."
        ),
    )
    parser.add_argument(
        "--use-tiny-config-torch",
        action="store_true",
        help=(
            "Run only the deterministic tiny FLUX.2 PyTorch reference path with constant "
            "weights. Use this as the explicit preflight before the generated JAX path."
        ),
    )
    parser.add_argument(
        "--use-reduced-checkpoint-bridge",
        action="store_true",
        help=(
            "Run the reduced real FLUX.2 bridge path: reduced text encoder + reduced "
            "transformer + real VAE."
        ),
    )
    parser.add_argument(
        "--token-ids-npz",
        dest="token_ids_npz",
        default=None,
        help=(
            "Optional .npz file containing an 'input_ids' array for the reduced "
            "prompt-to-image path."
        ),
    )
    parser.add_argument(
        "--prepare-checkpoint-artifacts",
        action="store_true",
        help="Export and transpile every available real FLUX submodule and write a JSON report.",
    )
    parser.add_argument(
        "--inspect-checkpoint-assets",
        action="store_true",
        help="Inspect real FLUX checkpoint files and config metadata without loading the model.",
    )
    parser.add_argument(
        "--download-checkpoint-assets",
        action="store_true",
        help="Download the expected FLUX checkpoint assets from Hugging Face.",
    )
    parser.add_argument(
        "--model-id",
        dest="model_id",
        default=None,
        help="Optional FLUX model id to use for snapshot resolution or download.",
    )
    parser.add_argument(
        "--checkpoint-submodule",
        dest="checkpoint_submodule",
        choices=flux_source.real_submodule_export_order(),
        default=None,
        help="Optional real FLUX submodule to prepare instead of sweeping the detected topology.",
    )
    parser.add_argument(
        "--checkpoint-graph-only",
        action="store_true",
        help=(
            "Export real checkpoint ONNX graphs without serializing weights, for faster smoke runs."
        ),
    )
    parser.add_argument(
        "--checkpoint-reduced-shapes",
        action="store_true",
        help="Use reduced sample shapes for the real transformer checkpoint lane.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint_submodule and not args.prepare_checkpoint_artifacts:
        raise ValueError("--checkpoint-submodule requires --prepare-checkpoint-artifacts.")
    if args.checkpoint_graph_only and not args.prepare_checkpoint_artifacts:
        raise ValueError("--checkpoint-graph-only requires --prepare-checkpoint-artifacts.")
    if args.checkpoint_reduced_shapes and not args.prepare_checkpoint_artifacts:
        raise ValueError("--checkpoint-reduced-shapes requires --prepare-checkpoint-artifacts.")
    if args.prepare_checkpoint_artifacts and (
        args.prompt_npz
        or args.use_text_encoders
        or args.use_dummy_flux2
        or args.use_tiny_config
        or args.use_tiny_config_torch
        or args.use_reduced_checkpoint_bridge
        or args.token_ids_npz
    ):
        raise ValueError(
            "--prepare-checkpoint-artifacts cannot be combined with prompt/image generation flags."
        )
    if args.inspect_checkpoint_assets and args.prepare_checkpoint_artifacts:
        raise ValueError(
            "--inspect-checkpoint-assets cannot be combined with --prepare-checkpoint-artifacts."
        )
    if args.inspect_checkpoint_assets and (
        args.prompt_npz
        or args.use_text_encoders
        or args.use_dummy_flux2
        or args.use_tiny_config
        or args.use_tiny_config_torch
        or args.use_reduced_checkpoint_bridge
        or args.token_ids_npz
    ):
        raise ValueError(
            "--inspect-checkpoint-assets cannot be combined with prompt/image generation flags."
        )
    if args.download_checkpoint_assets and (
        args.prompt_npz
        or args.use_text_encoders
        or args.use_dummy_flux2
        or args.use_tiny_config
        or args.use_tiny_config_torch
        or args.use_reduced_checkpoint_bridge
        or args.token_ids_npz
    ):
        raise ValueError(
            "--download-checkpoint-assets cannot be combined with prompt/image generation flags."
        )
    if args.use_dummy_flux2 and (
        args.prompt_npz
        or args.use_text_encoders
        or args.use_tiny_config
        or args.use_tiny_config_torch
        or args.use_reduced_checkpoint_bridge
        or args.token_ids_npz
    ):
        raise ValueError(
            "--use-dummy-flux2 cannot be combined with other prompt/image generation flags."
        )
    if args.use_tiny_config_torch and (
        args.prompt_npz
        or args.use_text_encoders
        or args.use_dummy_flux2
        or args.use_tiny_config
        or args.use_reduced_checkpoint_bridge
        or args.token_ids_npz
    ):
        raise ValueError(
            "--use-tiny-config-torch cannot be combined with other prompt/image generation flags."
        )
    if args.use_tiny_config and (
        args.prompt_npz
        or args.use_text_encoders
        or args.use_dummy_flux2
        or args.use_tiny_config_torch
        or args.use_reduced_checkpoint_bridge
        or args.token_ids_npz
    ):
        raise ValueError(
            "--use-tiny-config cannot be combined with other prompt/image generation flags."
        )
    if args.use_reduced_checkpoint_bridge and (
        args.prompt_npz
        or args.use_text_encoders
        or args.use_dummy_flux2
        or args.use_tiny_config
        or args.use_tiny_config_torch
        or args.token_ids_npz
    ):
        raise ValueError(
            "--use-reduced-checkpoint-bridge cannot be combined with other prompt/image "
            "generation flags."
        )
    if args.use_text_encoders and args.prompt_npz:
        raise ValueError("--prompt-npz cannot be combined with --use-text-encoders.")
    if args.token_ids_npz and not args.use_text_encoders:
        raise ValueError("--token-ids-npz requires --use-text-encoders.")

    downloaded_snapshot = None
    if args.download_checkpoint_assets:
        target_model_id = args.model_id or flux_source.DEFAULT_REAL_FLUX_MODEL_ID
        downloaded_snapshot = download_snapshot_from_hub(target_model_id)
        print("FLUX checkpoint assets downloaded.")
        print(f"Downloaded snapshot: {downloaded_snapshot}")
        if not args.prepare_checkpoint_artifacts and not args.inspect_checkpoint_assets:
            return 0

    if args.inspect_checkpoint_assets:
        result = inspect_flux_checkpoint_assets(
            args.output_dir,
            model_id=args.model_id,
        )
        print("FLUX checkpoint asset inspection completed.")
        print(f"Model id: {result['model_id']}")
        print(f"Snapshot: {result['snapshot']}")
        if downloaded_snapshot is not None:
            print(f"Downloaded snapshot: {downloaded_snapshot}")
        print(f"Report: {result['report_path']}")
        if result.get("report_write_error"):
            print(f"Report write error: {result['report_write_error']}")
        print(f"Ready submodules: {result['ready_count']}")
        print(f"Missing submodules: {result['missing_count']}")
        for submodule in result.get("submodule_order", []):
            entry = result["submodules"][submodule]
            line = (
                f"{submodule}: {entry['status']} "
                f"(config={entry['has_config']}, weights={entry['has_weights']})"
            )
            print(line)
        return 0

    if args.prepare_checkpoint_artifacts:
        result = prepare_flux_jax_checkpoint_artifacts(
            args.output_dir,
            model_id=args.model_id,
            checkpoint_submodule=args.checkpoint_submodule,
            checkpoint_graph_only=args.checkpoint_graph_only,
            checkpoint_reduced_shapes=args.checkpoint_reduced_shapes,
        )
        print("FLUX checkpoint JAX artifact preparation completed.")
        print(f"Model id: {result['model_id']}")
        print(f"Snapshot: {result['snapshot']}")
        if downloaded_snapshot is not None:
            print(f"Downloaded snapshot: {downloaded_snapshot}")
        print(f"Report: {result['report_path']}")
        print(f"Checkpoint graph only: {result['checkpoint_graph_only']}")
        print(f"Checkpoint reduced shapes: {result['checkpoint_reduced_shapes']}")
        if result.get("report_write_error"):
            print(f"Report write error: {result['report_write_error']}")
        print(f"Ready submodules: {result['ready_count']}")
        print(f"Blocked submodules: {result['blocked_count']}")
        print(f"Missing submodules: {result['missing_count']}")
        for submodule in result.get("submodule_order", []):
            entry = result["submodules"][submodule]
            line = f"{submodule}: {entry['status']}"
            if entry.get("reason"):
                line = f"{line} ({entry['reason']})"
            print(line)
        return 0

    prompt_embeddings = None
    pooled_prompt = None
    if args.use_reduced_checkpoint_bridge:
        result = run_flux_jax_reduced_checkpoint_bridge_demo(
            args.output_dir,
            model_id=args.model_id,
        )
    elif args.use_tiny_config_torch:
        result = run_flux_pytorch_tiny_config_e2e_demo(
            args.output_dir,
            model_id=args.model_id,
        )
    elif args.use_tiny_config:
        result = run_flux_jax_tiny_config_e2e_demo(
            args.output_dir,
            model_id=args.model_id,
        )
    elif args.use_dummy_flux2:
        result = run_flux_jax_dummy_flux2_demo(
            args.output_dir,
            steps=args.steps,
        )
    elif args.use_text_encoders:
        input_ids = None
        if args.token_ids_npz:
            input_ids = _load_token_ids_npz(args.token_ids_npz)

        result = run_flux_jax_prompt_to_image_demo(
            args.output_dir,
            steps=args.steps,
            input_ids=input_ids,
        )
    else:
        if args.prompt_npz:
            prompt_embeddings, pooled_prompt = _load_prompt_npz(args.prompt_npz)

        result = run_flux_jax_demo(
            args.output_dir,
            steps=args.steps,
            prompt_embeddings=prompt_embeddings,
            pooled_prompt=pooled_prompt,
        )

    if args.use_tiny_config_torch:
        print("Reduced FLUX PyTorch tiny-config demo completed.")
    else:
        print("Reduced FLUX JAX demo completed.")
    if "transformer_onnx" in result:
        print(f"Transformer ONNX: {result['transformer_onnx']}")
    if "vae_onnx" in result:
        print(f"VAE ONNX: {result['vae_onnx']}")
    if "text_encoder_onnx" in result:
        print(f"Text encoder ONNX: {result['text_encoder_onnx']}")
    print(f"Image: {result['image_path']}")
    if "pytorch_reference_image_path" in result:
        print(f"PyTorch reference image: {result['pytorch_reference_image_path']}")
    print(f"Prompt source: {result['prompt_source']}")
    if "text_encoder_max_abs" in result:
        print(f"Text encoder max abs: {float(result['text_encoder_max_abs']):.6e}")
    if "max_abs" in result:
        print(f"Max abs diff: {result['max_abs']:.6e}")
    if "mean_abs" in result:
        print(f"Mean abs diff: {result['mean_abs']:.6e}")
    print(f"Pixel std: {result['pixel_std']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
