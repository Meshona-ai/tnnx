from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .runtime_env import resolve_snapshot_root

FluxSubmodule = Literal["transformer", "vae_decoder", "text_encoder"]

DEFAULT_FLUX_MODEL_ID = "black-forest-labs/FLUX.2-klein-4b-fp8"
DEFAULT_REAL_FLUX_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
FLUX_MODEL_CANDIDATES: tuple[str, ...] = (
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-4b-fp8",
    "black-forest-labs/FLUX.2-klein-base-4B",
    "black-forest-labs/FLUX.2-klein-base-4b-nvfp4",
)
SUBMODULE_EXPORT_ORDER: tuple[FluxSubmodule, ...] = (
    "vae_decoder",
    "transformer",
    "text_encoder",
)
_SUBMODULE_WEIGHT_FILES: dict[FluxSubmodule, tuple[str, ...]] = {
    "transformer": (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
    ),
    "vae_decoder": (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
    ),
    "text_encoder": (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "model-*.safetensors",
    ),
}

DEMO_SEED = 7
DEMO_SEQUENCE = 4
DEMO_DIM = 8
DEMO_IMAGE_SIZE = 16
CHECKPOINT_SMOKE_IMAGE_SEQ_LEN = 1
CHECKPOINT_SMOKE_TEXT_SEQ_LEN = 2
DUMMY_FLUX2_SEQUENCE = 2
DUMMY_FLUX2_DIM = 4
DUMMY_FLUX2_IMAGE_SIZE = 8
DUMMY_FLUX2_TOKEN_SEQ_LEN = 4
TINY_FLUX2_IMAGE_SEQ_LEN = 4
TINY_FLUX2_TEXT_SEQ_LEN = 8
TINY_FLUX2_TOKEN_SEQ_LEN = 8
REAL_SUBMODULE_EXPORT_ORDER: tuple[FluxSubmodule, ...] = (
    "vae_decoder",
    "transformer",
    "text_encoder",
)


@dataclass(frozen=True, slots=True)
class ExportSpec:
    module: nn.Module
    sample_inputs: tuple[torch.Tensor, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


class TinyFluxTransformer(nn.Module):
    def __init__(self, dim: int = DEMO_DIM) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        latents: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        pooled_prompt: torch.Tensor,
    ) -> torch.Tensor:
        pooled = pooled_prompt.unsqueeze(1)
        hidden = self.norm(latents + prompt_embeddings + pooled)
        gated = self.in_proj(hidden) * torch.sigmoid(self.gate(pooled_prompt)).unsqueeze(1)
        return self.out_proj(F.gelu(gated))


class TinyFluxVaeDecoder(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int = DEMO_SEQUENCE,
        latent_dim: int = DEMO_DIM,
        image_size: int = DEMO_IMAGE_SIZE,
    ) -> None:
        super().__init__()
        in_features = seq_len * latent_dim
        pixel_count = 3 * image_size * image_size
        hidden = max(64, in_features * 4)
        self.image_size = image_size
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, pixel_count)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        batch = latents.shape[0]
        hidden = F.gelu(self.fc1(latents.reshape(batch, -1)))
        pixels = torch.tanh(self.fc2(hidden))
        return pixels.reshape(batch, 3, self.image_size, self.image_size)


class FluxVaeDecodeWrapper(nn.Module):
    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = cast(Any, self.vae).decode(latents)
        if hasattr(decoded, "sample"):
            return decoded.sample
        if isinstance(decoded, tuple | list):
            return decoded[0]
        return decoded


class FluxTextEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input_ids=input_ids)
        if hasattr(encoded, "last_hidden_state"):
            return encoded.last_hidden_state
        if isinstance(encoded, tuple | list):
            return encoded[0]
        return encoded


class FluxTransformerForwardWrapper(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        *,
        include_pooled_projections: bool,
        include_guidance: bool,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.include_pooled_projections = include_pooled_projections
        self.include_guidance = include_guidance

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "return_dict": True,
        }
        if self.include_pooled_projections:
            kwargs["pooled_projections"] = pooled_projections
        if self.include_guidance:
            kwargs["guidance"] = guidance
        output = self.transformer(**kwargs)
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, tuple | list):
            return output[0]
        return output


def resolve_flux_snapshot(model_id: str | None = None) -> Path:
    return resolve_snapshot_root(model_id or DEFAULT_REAL_FLUX_MODEL_ID)


def _resolved_model_id(model_id: str | None = None) -> str:
    return model_id or DEFAULT_REAL_FLUX_MODEL_ID


def real_submodule_export_order(
    *,
    snapshot: Path | None = None,
    model_id: str | None = None,
) -> tuple[FluxSubmodule, ...]:
    _ = snapshot, model_id
    return REAL_SUBMODULE_EXPORT_ORDER


def snapshot_has_submodule_weights(snapshot: Path, submodule: FluxSubmodule) -> bool:
    if submodule not in real_submodule_export_order(snapshot=snapshot):
        return False

    subdir = "vae" if submodule == "vae_decoder" else submodule
    root = snapshot / subdir
    if not root.exists():
        return False
    for pattern in _SUBMODULE_WEIGHT_FILES[submodule]:
        if any(char in pattern for char in "*?[]"):
            if any(root.glob(pattern)):
                return True
            continue
        if (root / pattern).exists():
            return True
    return False


def _config_value(config: Any, name: str) -> Any:
    if hasattr(config, name):
        return getattr(config, name)

    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(name)

    if isinstance(config, dict):
        return config.get(name)

    return None


def _required_config_int(
    config: Any,
    *names: str,
) -> int:
    for name in names:
        value = _config_value(config, name)
        if value is not None:
            return int(value)

    joined = ", ".join(names)
    raise ValueError(f"Transformer config is missing required integer value(s): {joined}.")


def _reduced_transformer_config(config: dict[str, Any]) -> dict[str, Any]:
    reduced = dict(config)
    axes_dims = tuple(int(value) for value in config.get("axes_dims_rope", (4, 4, 4, 4)))
    axis_count = max(1, len(axes_dims))
    attention_head_dim = min(16, int(config.get("attention_head_dim", 16)))
    rope_axis_dim = max(1, attention_head_dim // axis_count)

    reduced["patch_size"] = 1
    reduced["in_channels"] = min(32, int(config.get("in_channels", 32)))
    reduced["out_channels"] = min(
        reduced["in_channels"],
        int(config.get("out_channels") or reduced["in_channels"]),
    )
    reduced["num_layers"] = 1
    reduced["num_single_layers"] = 1
    reduced["attention_head_dim"] = attention_head_dim
    reduced["num_attention_heads"] = min(4, int(config.get("num_attention_heads", 4)))
    reduced["joint_attention_dim"] = min(64, int(config.get("joint_attention_dim", 64)))
    reduced["timestep_guidance_channels"] = min(
        16,
        int(config.get("timestep_guidance_channels", 16)),
    )
    for key in ("pooled_projection_dim", "pooled_projection_embed_dim"):
        value = config.get(key)
        if value is not None:
            reduced[key] = min(64, int(value))
    reduced["axes_dims_rope"] = tuple(rope_axis_dim for _ in range(axis_count))
    return reduced


def _reduced_text_encoder_config(config: dict[str, Any]) -> dict[str, Any]:
    reduced = dict(config)
    num_attention_heads = max(1, min(4, int(config.get("num_attention_heads", 4))))
    original_head_dim = int(
        config.get(
            "head_dim",
            max(
                1,
                int(config.get("hidden_size", 64))
                // max(1, int(config.get("num_attention_heads", 1))),
            ),
        )
    )
    head_dim = max(1, min(16, original_head_dim))
    hidden_size = num_attention_heads * head_dim
    num_key_value_heads = max(
        1,
        min(
            num_attention_heads,
            int(config.get("num_key_value_heads", num_attention_heads)),
        ),
    )
    max_positions = max(16, min(64, int(config.get("max_position_embeddings", 64))))
    vocab_size = max(64, min(512, int(config.get("vocab_size", 512))))

    reduced["hidden_size"] = hidden_size
    reduced["intermediate_size"] = max(
        hidden_size * 2,
        min(256, int(config.get("intermediate_size", hidden_size * 4))),
    )
    reduced["num_hidden_layers"] = 1
    reduced["num_attention_heads"] = num_attention_heads
    reduced["num_key_value_heads"] = num_key_value_heads
    reduced["head_dim"] = head_dim
    reduced["max_position_embeddings"] = max_positions
    reduced["vocab_size"] = vocab_size

    if "bos_token_id" in reduced:
        reduced["bos_token_id"] = min(int(reduced["bos_token_id"]), vocab_size - 1)
    if "eos_token_id" in reduced:
        reduced["eos_token_id"] = min(int(reduced["eos_token_id"]), vocab_size - 1)
    if "pad_token_id" in reduced and reduced["pad_token_id"] is not None:
        reduced["pad_token_id"] = min(int(reduced["pad_token_id"]), vocab_size - 1)

    layer_types = reduced.get("layer_types")
    if isinstance(layer_types, list):
        reduced["layer_types"] = layer_types[:1]
    if "max_window_layers" in reduced:
        reduced["max_window_layers"] = min(
            int(reduced["max_window_layers"]),
            reduced["num_hidden_layers"],
        )
    if "sliding_window" in reduced and reduced["sliding_window"] is not None:
        reduced["sliding_window"] = min(int(reduced["sliding_window"]), max_positions)

    rope_scaling = reduced.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        rope_scaling = dict(rope_scaling)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling["original_max_position_embeddings"] = max_positions
        reduced["rope_scaling"] = rope_scaling

    return reduced


def _tiny_transformer_config(config: dict[str, Any]) -> dict[str, Any]:
    reduced = dict(config)
    axes_dims = tuple(int(value) for value in config.get("axes_dims_rope", (4, 4, 4, 4)))
    axis_count = max(1, len(axes_dims))
    attention_head_dim = min(8, int(config.get("attention_head_dim", 8)))
    rope_axis_dim = max(1, attention_head_dim // axis_count)

    reduced["patch_size"] = 1
    reduced["in_channels"] = min(8, int(config.get("in_channels", 8)))
    reduced["out_channels"] = min(
        reduced["in_channels"],
        int(config.get("out_channels") or reduced["in_channels"]),
    )
    reduced["num_layers"] = min(2, int(config.get("num_layers", 2)))
    reduced["num_single_layers"] = min(2, int(config.get("num_single_layers", 2)))
    reduced["attention_head_dim"] = attention_head_dim
    reduced["num_attention_heads"] = min(2, int(config.get("num_attention_heads", 2)))
    reduced["joint_attention_dim"] = min(32, int(config.get("joint_attention_dim", 32)))
    reduced["timestep_guidance_channels"] = min(
        8,
        int(config.get("timestep_guidance_channels", 8)),
    )
    for key in ("pooled_projection_dim", "pooled_projection_embed_dim"):
        value = config.get(key)
        if value is not None:
            reduced[key] = min(32, int(value))
    reduced["axes_dims_rope"] = tuple(rope_axis_dim for _ in range(axis_count))
    return reduced


def _tiny_text_encoder_config(
    config: dict[str, Any],
    *,
    target_hidden_size: int | None = None,
) -> dict[str, Any]:
    reduced = dict(config)
    num_attention_heads = max(1, min(2, int(config.get("num_attention_heads", 2))))
    if target_hidden_size is None:
        target_hidden_size = min(
            32,
            int(config.get("hidden_size", config.get("head_dim", 8) * num_attention_heads)),
        )
    target_hidden_size = max(num_attention_heads, int(target_hidden_size))
    target_hidden_size = max(
        num_attention_heads,
        (target_hidden_size // num_attention_heads) * num_attention_heads,
    )
    head_dim = max(1, target_hidden_size // num_attention_heads)
    hidden_size = num_attention_heads * head_dim
    num_key_value_heads = max(
        1,
        min(
            num_attention_heads,
            int(config.get("num_key_value_heads", num_attention_heads)),
        ),
    )
    max_positions = max(8, min(32, int(config.get("max_position_embeddings", 32))))
    vocab_size = max(32, min(128, int(config.get("vocab_size", 128))))

    reduced["hidden_size"] = hidden_size
    reduced["intermediate_size"] = max(
        hidden_size * 2,
        min(64, int(config.get("intermediate_size", hidden_size * 4))),
    )
    reduced["num_hidden_layers"] = min(2, int(config.get("num_hidden_layers", 2)))
    reduced["num_attention_heads"] = num_attention_heads
    reduced["num_key_value_heads"] = num_key_value_heads
    reduced["head_dim"] = head_dim
    reduced["max_position_embeddings"] = max_positions
    reduced["vocab_size"] = vocab_size

    if "bos_token_id" in reduced:
        reduced["bos_token_id"] = min(int(reduced["bos_token_id"]), vocab_size - 1)
    if "eos_token_id" in reduced:
        reduced["eos_token_id"] = min(int(reduced["eos_token_id"]), vocab_size - 1)
    if "pad_token_id" in reduced and reduced["pad_token_id"] is not None:
        reduced["pad_token_id"] = min(int(reduced["pad_token_id"]), vocab_size - 1)

    layer_types = reduced.get("layer_types")
    if isinstance(layer_types, list):
        reduced["layer_types"] = layer_types[: max(1, reduced["num_hidden_layers"])]
    if "max_window_layers" in reduced:
        reduced["max_window_layers"] = min(
            int(reduced["max_window_layers"]),
            reduced["num_hidden_layers"],
        )
    if "sliding_window" in reduced and reduced["sliding_window"] is not None:
        reduced["sliding_window"] = min(int(reduced["sliding_window"]), max_positions)

    rope_scaling = reduced.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        rope_scaling = dict(rope_scaling)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling["original_max_position_embeddings"] = max_positions
        reduced["rope_scaling"] = rope_scaling

    return reduced


def _tiny_vae_config(config: dict[str, Any]) -> dict[str, Any]:
    reduced = dict(config)
    block_types = list(config.get("down_block_types", []))
    block_count = max(1, len(block_types)) if block_types else 4
    block_out_channels = []
    for idx in range(block_count):
        block_out_channels.append(8 if idx < 2 else 16)
    reduced["block_out_channels"] = tuple(block_out_channels)
    reduced["layers_per_block"] = 1
    reduced["latent_channels"] = min(8, int(config.get("latent_channels", 8)))
    reduced["sample_size"] = min(16, int(config.get("sample_size", 16)))
    patch_size = config.get("patch_size", [1, 1])
    if isinstance(patch_size, list):
        reduced["patch_size"] = [1 for _ in patch_size] or [1, 1]
    elif isinstance(patch_size, tuple):
        reduced["patch_size"] = tuple(1 for _ in patch_size) or (1, 1)
    else:
        reduced["patch_size"] = [1, 1]
    reduced["norm_num_groups"] = min(
        int(config.get("norm_num_groups", 8)),
        max(1, min(block_out_channels)),
    )
    return reduced


def _fill_module_with_constants(
    module: nn.Module,
    *,
    weight_value: float = 0.01,
    bias_value: float = 0.0,
) -> nn.Module:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not parameter.dtype.is_floating_point:
                continue
            fill = bias_value if name.endswith("bias") else weight_value
            parameter.fill_(fill)
        for _name, buffer in module.named_buffers():
            if buffer.dtype.is_floating_point:
                buffer.zero_()
    return module


def build_flux_transformer_sample_inputs(
    transformer: nn.Module,
    *,
    image_seq_len: int = 4,
    text_seq_len: int = 8,
    seed: int = DEMO_SEED,
    fill_value: float | None = None,
) -> dict[str, torch.Tensor]:
    config = getattr(transformer, "config", None)
    if config is None:
        raise ValueError("Transformer is missing a config object.")

    forward_params = inspect.signature(transformer.forward).parameters
    include_pooled_projections = "pooled_projections" in forward_params
    include_guidance = "guidance" in forward_params and (
        not include_pooled_projections or bool(_config_value(config, "guidance_embeds"))
    )

    in_channels = _required_config_int(config, "in_channels")
    joint_attention_dim = _required_config_int(config, "joint_attention_dim")

    if fill_value is None:
        generator = torch.Generator().manual_seed(seed)
        hidden_states = torch.randn(
            (1, image_seq_len, in_channels),
            generator=generator,
            dtype=torch.float32,
        )
        encoder_hidden_states = torch.randn(
            (1, text_seq_len, joint_attention_dim),
            generator=generator,
            dtype=torch.float32,
        )
    else:
        hidden_states = torch.full(
            (1, image_seq_len, in_channels),
            float(fill_value),
            dtype=torch.float32,
        )
        encoder_hidden_states = torch.full(
            (1, text_seq_len, joint_attention_dim),
            float(fill_value),
            dtype=torch.float32,
        )
    timestep = torch.ones((1,), dtype=torch.int64)
    axes_dims_rope = _config_value(config, "axes_dims_rope")
    id_width = len(axes_dims_rope) if isinstance(axes_dims_rope, tuple | list) else 3

    txt_ids = torch.zeros((text_seq_len, id_width), dtype=torch.int64)
    if text_seq_len > 0:
        txt_ids[:, 0] = torch.arange(text_seq_len, dtype=torch.int64)

    side = max(1, int(image_seq_len**0.5))
    img_ids = torch.zeros((image_seq_len, id_width), dtype=torch.int64)
    if image_seq_len > 0:
        index = torch.arange(image_seq_len, dtype=torch.int64)
        img_ids[:, 0] = index
        if id_width > 1:
            img_ids[:, 1] = index // side
        if id_width > 2:
            img_ids[:, 2] = index % side

    inputs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
    }
    if include_pooled_projections:
        pooled_projection_dim = _required_config_int(
            config,
            "pooled_projection_dim",
            "pooled_projection_embed_dim",
        )
        if fill_value is None:
            generator = torch.Generator().manual_seed(seed)
            inputs["pooled_projections"] = torch.randn(
                (1, pooled_projection_dim),
                generator=generator,
                dtype=torch.float32,
            )
        else:
            inputs["pooled_projections"] = torch.full(
                (1, pooled_projection_dim),
                float(fill_value),
                dtype=torch.float32,
            )
    if include_guidance:
        inputs["guidance"] = torch.ones((1,), dtype=torch.float32)
    return inputs


def build_demo_inputs(
    *,
    seed: int = DEMO_SEED,
    seq_len: int = DEMO_SEQUENCE,
    dim: int = DEMO_DIM,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "latents": torch.randn((1, seq_len, dim), generator=generator, dtype=torch.float32),
        "prompt_embeddings": torch.randn(
            (1, seq_len, dim),
            generator=generator,
            dtype=torch.float32,
        ),
        "pooled_prompt": torch.randn((1, dim), generator=generator, dtype=torch.float32),
    }


def build_demo_export_specs(
    *,
    seed: int = DEMO_SEED,
    seq_len: int = DEMO_SEQUENCE,
    dim: int = DEMO_DIM,
    image_size: int = DEMO_IMAGE_SIZE,
) -> dict[FluxSubmodule, ExportSpec]:
    torch.manual_seed(seed)
    demo_inputs = build_demo_inputs(seed=seed, seq_len=seq_len, dim=dim)
    return {
        "transformer": ExportSpec(
            module=TinyFluxTransformer(dim=dim).eval(),
            sample_inputs=(
                demo_inputs["latents"],
                demo_inputs["prompt_embeddings"],
                demo_inputs["pooled_prompt"],
            ),
            input_names=("latents", "prompt_embeddings", "pooled_prompt"),
            output_names=("denoised",),
        ),
        "vae_decoder": ExportSpec(
            module=TinyFluxVaeDecoder(
                seq_len=seq_len,
                latent_dim=dim,
                image_size=image_size,
            ).eval(),
            sample_inputs=(demo_inputs["latents"],),
            input_names=("latents",),
            output_names=("image",),
        ),
        "text_encoder": ExportSpec(
            module=nn.Embedding(64, dim).eval(),
            sample_inputs=(torch.arange(seq_len, dtype=torch.long).unsqueeze(0),),
            input_names=("input_ids",),
            output_names=("hidden",),
        ),
    }


def build_dummy_flux2_export_specs(
    *,
    seed: int = DEMO_SEED,
    seq_len: int = DUMMY_FLUX2_SEQUENCE,
    dim: int = DUMMY_FLUX2_DIM,
    image_size: int = DUMMY_FLUX2_IMAGE_SIZE,
    token_seq_len: int = DUMMY_FLUX2_TOKEN_SEQ_LEN,
) -> dict[FluxSubmodule, ExportSpec]:
    torch.manual_seed(seed)
    demo_inputs = build_demo_inputs(seed=seed, seq_len=seq_len, dim=dim)
    token_length = max(1, int(token_seq_len))
    vocab_size = max(16, token_length * 4)
    token_ids = torch.arange(token_length, dtype=torch.long).unsqueeze(0) % vocab_size
    return {
        "transformer": ExportSpec(
            module=TinyFluxTransformer(dim=dim).eval(),
            sample_inputs=(
                demo_inputs["latents"],
                demo_inputs["prompt_embeddings"],
                demo_inputs["pooled_prompt"],
            ),
            input_names=("latents", "prompt_embeddings", "pooled_prompt"),
            output_names=("denoised",),
        ),
        "vae_decoder": ExportSpec(
            module=TinyFluxVaeDecoder(
                seq_len=seq_len,
                latent_dim=dim,
                image_size=image_size,
            ).eval(),
            sample_inputs=(demo_inputs["latents"],),
            input_names=("latents",),
            output_names=("image",),
        ),
        "text_encoder": ExportSpec(
            module=nn.Embedding(vocab_size, dim).eval(),
            sample_inputs=(token_ids,),
            input_names=("input_ids",),
            output_names=("hidden",),
        ),
    }


def build_tiny_flux2_export_specs(
    *,
    model_id: str | None = None,
    image_seq_len: int = TINY_FLUX2_IMAGE_SEQ_LEN,
    text_seq_len: int = TINY_FLUX2_TEXT_SEQ_LEN,
    token_seq_len: int = TINY_FLUX2_TOKEN_SEQ_LEN,
    fill_value: float = 0.01,
) -> dict[FluxSubmodule, ExportSpec]:
    snapshot = resolve_flux_snapshot(model_id)
    AutoencoderKLFlux2, Flux2Transformer2DModel = _require_diffusers_flux2_components()
    transformers = _require_transformers()

    vae_config = AutoencoderKLFlux2.load_config(
        str(snapshot / "vae"),
        local_files_only=True,
    )
    transformer_config = Flux2Transformer2DModel.load_config(
        str(snapshot / "transformer"),
        local_files_only=True,
    )
    text_config = transformers.AutoConfig.from_pretrained(
        str(snapshot / "text_encoder"),
        local_files_only=True,
    )

    vae = _fill_module_with_constants(
        AutoencoderKLFlux2.from_config(_tiny_vae_config(dict(vae_config))).eval(),
        weight_value=fill_value,
    )
    tiny_transformer_config = _tiny_transformer_config(dict(transformer_config))
    transformer = _fill_module_with_constants(
        Flux2Transformer2DModel.from_config(tiny_transformer_config).eval(),
        weight_value=fill_value,
    )
    tiny_text_config = text_config.__class__(
        **_tiny_text_encoder_config(
            text_config.to_dict(),
            target_hidden_size=int(tiny_transformer_config["joint_attention_dim"]),
        )
    )
    text_encoder = _fill_module_with_constants(
        transformers.AutoModel.from_config(tiny_text_config).eval(),
        weight_value=fill_value,
    )

    transformer_samples = build_flux_transformer_sample_inputs(
        transformer,
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        fill_value=0.0,
    )
    include_pooled_projections = "pooled_projections" in transformer_samples
    include_guidance = "guidance" in transformer_samples
    transformer_input_names: tuple[str, ...] = (
        "hidden_states",
        "encoder_hidden_states",
        "timestep",
        "img_ids",
        "txt_ids",
    )
    transformer_sample_inputs: tuple[torch.Tensor, ...] = tuple(
        transformer_samples[name] for name in transformer_input_names
    )
    if include_guidance or include_pooled_projections:
        transformer_input_names = transformer_input_names + ("guidance",)
        transformer_sample_inputs = transformer_sample_inputs + (
            transformer_samples.get("guidance", torch.ones((1,), dtype=torch.float32)),
        )
    if include_pooled_projections:
        transformer_input_names = transformer_input_names + ("pooled_projections",)
        transformer_sample_inputs = transformer_sample_inputs + (
            transformer_samples["pooled_projections"],
        )

    token_ids = torch.zeros((1, max(1, int(token_seq_len))), dtype=torch.long)
    latent_channels = int(getattr(vae.config, "latent_channels"))
    latents = torch.zeros((1, latent_channels, 2, 2), dtype=torch.float32)

    return {
        "text_encoder": ExportSpec(
            module=FluxTextEncoderWrapper(text_encoder).eval(),
            sample_inputs=(token_ids,),
            input_names=("input_ids",),
            output_names=("hidden",),
        ),
        "transformer": ExportSpec(
            module=FluxTransformerForwardWrapper(
                transformer,
                include_pooled_projections=include_pooled_projections,
                include_guidance=include_guidance,
            ).eval(),
            sample_inputs=transformer_sample_inputs,
            input_names=transformer_input_names,
            output_names=("denoised",),
        ),
        "vae_decoder": ExportSpec(
            module=FluxVaeDecodeWrapper(vae).eval(),
            sample_inputs=(latents,),
            input_names=("latents",),
            output_names=("image",),
        ),
    }


def _require_diffusers_flux2_components() -> tuple[Any, Any]:
    try:
        from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "This FLUX.2 path requires a diffusers build with AutoencoderKLFlux2 and "
            "Flux2Transformer2DModel. Run: uv sync --dev --group examples"
        ) from exc
    return AutoencoderKLFlux2, Flux2Transformer2DModel


def _require_transformers() -> Any:
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This FLUX path requires 'transformers'. Run: uv sync --dev --group examples"
        ) from exc
    return transformers


def _load_real_component(
    submodule: FluxSubmodule,
    model_id: str | None = None,
    *,
    load_weights: bool = True,
) -> Any:
    snapshot = resolve_flux_snapshot(model_id)
    AutoencoderKLFlux2, Flux2Transformer2DModel = _require_diffusers_flux2_components()
    transformers = _require_transformers()
    if submodule == "vae_decoder":
        return AutoencoderKLFlux2.from_pretrained(
            str(snapshot / "vae"),
            local_files_only=True,
        )
    if submodule == "transformer":
        if not load_weights:
            config = Flux2Transformer2DModel.load_config(
                str(snapshot / "transformer"),
                local_files_only=True,
            )
            return Flux2Transformer2DModel.from_config(_reduced_transformer_config(dict(config)))
        return Flux2Transformer2DModel.from_pretrained(
            str(snapshot / "transformer"),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    if submodule == "text_encoder":
        if not load_weights:
            config = transformers.AutoConfig.from_pretrained(
                str(snapshot / "text_encoder"),
                local_files_only=True,
            )
            reduced_config = config.__class__(**_reduced_text_encoder_config(config.to_dict()))
            return transformers.AutoModel.from_config(reduced_config)
        return transformers.AutoModel.from_pretrained(
            str(snapshot / "text_encoder"),
            local_files_only=True,
        )
    raise ValueError(f"Submodule {submodule!r} is not part of the real FLUX.2 topology.")


def _real_export_spec(
    submodule: FluxSubmodule,
    *,
    model_id: str | None = None,
    transformer_image_seq_len: int | None = None,
    transformer_text_seq_len: int | None = None,
    load_weights: bool = True,
) -> ExportSpec:
    snapshot = resolve_flux_snapshot(model_id)
    if submodule not in real_submodule_export_order(snapshot=snapshot, model_id=model_id):
        raise ValueError(
            f"Submodule {submodule!r} is not part of the real FLUX topology for "
            f"{_resolved_model_id(model_id)!r}."
        )
    if submodule == "vae_decoder":
        vae = _load_real_component("vae_decoder", model_id)
        vae_config = getattr(vae, "config", None)
        if vae_config is None or not hasattr(vae_config, "latent_channels"):
            raise ValueError("Real FLUX VAE is missing config.latent_channels.")
        latent_channels = int(getattr(vae_config, "latent_channels"))
        latents = torch.randn((1, latent_channels, 2, 2), dtype=torch.float32)
        return ExportSpec(
            module=FluxVaeDecodeWrapper(vae).eval(),
            sample_inputs=(latents,),
            input_names=("latents",),
            output_names=("image",),
        )

    if submodule == "text_encoder":
        encoder = _load_real_component("text_encoder", model_id, load_weights=load_weights)
        max_positions = int(getattr(encoder.config, "max_position_embeddings", 16))
        length = min(16, max_positions)
        sample = torch.zeros((1, length), dtype=torch.long)
        return ExportSpec(
            module=FluxTextEncoderWrapper(encoder).eval(),
            sample_inputs=(sample,),
            input_names=("input_ids",),
            output_names=("hidden",),
        )

    if submodule == "transformer":
        transformer = _load_real_component("transformer", model_id, load_weights=load_weights)
        sample_map = build_flux_transformer_sample_inputs(
            transformer,
            image_seq_len=4 if transformer_image_seq_len is None else transformer_image_seq_len,
            text_seq_len=8 if transformer_text_seq_len is None else transformer_text_seq_len,
        )
        include_pooled_projections = "pooled_projections" in sample_map
        include_guidance = "guidance" in sample_map
        input_names = (
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "img_ids",
            "txt_ids",
        )
        sample_inputs = tuple(sample_map[name] for name in input_names)
        if include_guidance or include_pooled_projections:
            input_names = input_names + ("guidance",)
            sample_inputs = sample_inputs + (
                sample_map.get("guidance", torch.zeros((1,), dtype=torch.float32)),
            )
        if include_pooled_projections:
            input_names = input_names + ("pooled_projections",)
            sample_inputs = sample_inputs + (sample_map["pooled_projections"],)
        return ExportSpec(
            module=FluxTransformerForwardWrapper(
                transformer,
                include_pooled_projections=include_pooled_projections,
                include_guidance=include_guidance,
            ).eval(),
            sample_inputs=sample_inputs,
            input_names=input_names,
            output_names=("denoised",),
        )

    raise ValueError(f"Unsupported FLUX submodule for real export: {submodule}")


def export_flux_submodule_onnx(
    submodule: FluxSubmodule,
    out_path: str | Path,
    *,
    module: nn.Module | None = None,
    sample_inputs: Sequence[torch.Tensor] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    model_id: str | None = None,
    export_params: bool = True,
) -> Path:
    if submodule not in SUBMODULE_EXPORT_ORDER:
        raise ValueError(f"Unsupported FLUX submodule: {submodule}")

    if module is None:
        spec = _real_export_spec(submodule, model_id=model_id)
    else:
        if sample_inputs is None:
            raise ValueError("sample_inputs are required when exporting a custom module.")
        spec = ExportSpec(
            module=module.eval(),
            sample_inputs=tuple(sample_inputs),
            input_names=tuple(input_names or [f"input_{idx}" for idx in range(len(sample_inputs))]),
            output_names=tuple(output_names or ("output",)),
        )

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    args = tuple(spec.sample_inputs)
    use_dynamo_export = submodule in {"transformer", "text_encoder"}
    try:
        torch.onnx.export(
            spec.module,
            args,
            target,
            input_names=list(spec.input_names),
            output_names=list(spec.output_names),
            opset_version=18,
            dynamo=use_dynamo_export,
            export_params=export_params,
            external_data=export_params,
        )
    except ModuleNotFoundError as exc:
        if use_dynamo_export and exc.name == "onnxscript":
            raise ModuleNotFoundError(
                "FLUX ONNX export with dynamo=True requires 'onnxscript'. "
                "Run: uv sync --dev --group examples"
            ) from exc
        raise
    return target


def load_onnx_op_inventory(onnx_path: str | Path) -> tuple[str, ...]:
    import onnx

    exported = onnx.load(Path(onnx_path))
    return tuple(sorted({node.op_type for node in exported.graph.node}))


def load_unsupported_onnx_ops(
    onnx_path: str | Path,
    *,
    extra_supported: Sequence[str] = ("Constant",),
) -> tuple[str, ...]:
    from tnnx.ingest.op_map import ONNX_TO_SEMANTIC

    supported = set(ONNX_TO_SEMANTIC) | set(extra_supported)
    return tuple(sorted(set(load_onnx_op_inventory(onnx_path)) - supported))
