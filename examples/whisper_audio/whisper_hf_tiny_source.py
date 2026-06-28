from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Any

import numpy as np

from .runtime_env import ensure_runtime_paths, resolve_model_snapshot

ensure_runtime_paths(require_mlx=False)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torch import Tensor  # noqa: E402

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH
DEFAULT_DECODE_TOKENS = 64
_ASSET_MEL_FILTERS = Path(__file__).with_name("assets") / "mel_filters.npz"


@dataclass(frozen=True, slots=True)
class WhisperTinyConfig:
    d_model: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    encoder_layers: int
    decoder_attention_heads: int
    decoder_ffn_dim: int
    decoder_layers: int
    num_mel_bins: int
    max_source_positions: int
    max_target_positions: int
    vocab_size: int
    decoder_start_token_id: int
    eos_token_id: int
    pad_token_id: int
    forced_decoder_ids: list[list[int | None]]
    suppress_tokens: list[int]
    begin_suppress_tokens: list[int]
    no_timestamps_token_id: int
    transcribe_token_id: int
    english_token_id: int

    @classmethod
    def from_snapshot(cls, snapshot: Path) -> WhisperTinyConfig:
        config_payload = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        generation_payload = json.loads(
            (snapshot / "generation_config.json").read_text(encoding="utf-8")
        )
        lang_to_id = generation_payload["lang_to_id"]
        task_to_id = generation_payload["task_to_id"]
        return cls(
            d_model=int(config_payload["d_model"]),
            encoder_attention_heads=int(config_payload["encoder_attention_heads"]),
            encoder_ffn_dim=int(config_payload["encoder_ffn_dim"]),
            encoder_layers=int(config_payload["encoder_layers"]),
            decoder_attention_heads=int(config_payload["decoder_attention_heads"]),
            decoder_ffn_dim=int(config_payload["decoder_ffn_dim"]),
            decoder_layers=int(config_payload["decoder_layers"]),
            num_mel_bins=int(config_payload["num_mel_bins"]),
            max_source_positions=int(config_payload["max_source_positions"]),
            max_target_positions=int(config_payload["max_target_positions"]),
            vocab_size=int(config_payload["vocab_size"]),
            decoder_start_token_id=int(config_payload["decoder_start_token_id"]),
            eos_token_id=int(config_payload["eos_token_id"]),
            pad_token_id=int(config_payload["pad_token_id"]),
            forced_decoder_ids=[list(item) for item in config_payload["forced_decoder_ids"]],
            suppress_tokens=[int(v) for v in generation_payload["suppress_tokens"]],
            begin_suppress_tokens=[int(v) for v in generation_payload["begin_suppress_tokens"]],
            no_timestamps_token_id=int(generation_payload["no_timestamps_token_id"]),
            transcribe_token_id=int(task_to_id["transcribe"]),
            english_token_id=int(lang_to_id["<|en|>"]),
        )


class WhisperAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, *, key_bias: bool) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.25
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=key_bias)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: Tensor,
        *,
        kv: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        source = x if kv is None else kv
        batch = x.shape[0]
        q_len = x.shape[1]
        kv_len = source.shape[1]

        q = self.q_proj(x).reshape(batch, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = (
            self.k_proj(source)
            .reshape(batch, kv_len, self.num_heads, self.head_dim)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.v_proj(source)
            .reshape(batch, kv_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        scores = (q * self.scale) @ (k * self.scale)
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).permute(0, 2, 1, 3).reshape(batch, q_len, -1)
        return self.out_proj(out)


class WhisperEncoderLayer(nn.Module):
    def __init__(self, cfg: WhisperTinyConfig) -> None:
        super().__init__()
        self.self_attn = WhisperAttention(
            cfg.d_model,
            cfg.encoder_attention_heads,
            key_bias=False,
        )
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        x = x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        return x


class WhisperDecoderLayer(nn.Module):
    def __init__(self, cfg: WhisperTinyConfig) -> None:
        super().__init__()
        self.self_attn = WhisperAttention(
            cfg.d_model,
            cfg.decoder_attention_heads,
            key_bias=False,
        )
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.encoder_attn = WhisperAttention(
            cfg.d_model,
            cfg.decoder_attention_heads,
            key_bias=False,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.decoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.decoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor, encoder_out: Tensor, *, causal_mask: Tensor) -> Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask=causal_mask)
        x = x + self.encoder_attn(self.encoder_attn_layer_norm(x), kv=encoder_out)
        x = x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        return x


class WhisperEncoder(nn.Module):
    def __init__(self, cfg: WhisperTinyConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, cfg.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(cfg.max_source_positions, cfg.d_model)
        self.layers = nn.ModuleList([WhisperEncoderLayer(cfg) for _ in range(cfg.encoder_layers)])
        self.layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, mel: Tensor) -> Tensor:
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        pos = self.embed_positions.weight[: x.shape[1]].unsqueeze(0)
        x = x + pos
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class WhisperDecoder(nn.Module):
    def __init__(self, cfg: WhisperTinyConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_positions = nn.Embedding(cfg.max_target_positions, cfg.d_model)
        self.layers = nn.ModuleList([WhisperDecoderLayer(cfg) for _ in range(cfg.decoder_layers)])
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        mask = torch.full(
            (cfg.max_target_positions, cfg.max_target_positions),
            -1e4,
            dtype=torch.float32,
        )
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, tokens: Tensor, encoder_out: Tensor) -> Tensor:
        pos = self.embed_positions.weight[: tokens.shape[1]].unsqueeze(0)
        x = self.embed_tokens(tokens) + pos
        mask = self.causal_mask[: tokens.shape[1], : tokens.shape[1]].unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, encoder_out, causal_mask=mask)
        x = self.layer_norm(x)
        return torch.matmul(x, self.embed_tokens.weight.transpose(0, 1))


class WhisperTinyForTranspile(nn.Module):
    def __init__(self, cfg: WhisperTinyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = nn.Module()
        self.model.encoder = WhisperEncoder(cfg)
        self.model.decoder = WhisperDecoder(cfg)

    def forward(self, mel: Tensor, tokens: Tensor) -> Tensor:
        encoder_out = self.model.encoder(mel)
        return self.model.decoder(tokens, encoder_out)

    @classmethod
    def from_hf_snapshot(
        cls,
        snapshot: Path | None = None,
    ) -> tuple[WhisperTinyForTranspile, WhisperTinyConfig]:
        resolved_snapshot = snapshot or resolve_model_snapshot()
        cfg = WhisperTinyConfig.from_snapshot(resolved_snapshot)
        model = cls(cfg)
        state = load_file(str(resolved_snapshot / "model.safetensors"), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected HF weights: {unexpected}")
        if missing:
            raise ValueError(f"Missing HF weights: {missing}")
        model.eval()
        return model, cfg


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray[Any, Any]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        output = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        raise RuntimeError(f"Failed to load audio via ffmpeg: {exc.stderr.decode()}") from exc
    return np.frombuffer(output, np.int16).astype(np.float32) / 32768.0


def pad_or_trim(audio: np.ndarray[Any, Any], *, length: int = N_SAMPLES) -> np.ndarray[Any, Any]:
    if audio.shape[0] > length:
        return audio[:length]
    if audio.shape[0] < length:
        return np.pad(audio, (0, length - audio.shape[0]))
    return audio


def log_mel_spectrogram(audio_path: str | Path) -> torch.Tensor:
    audio = pad_or_trim(load_audio(audio_path))
    audio_t = torch.from_numpy(audio)
    window = torch.hann_window(N_FFT)
    stft = torch.stft(
        audio_t,
        N_FFT,
        HOP_LENGTH,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs() ** 2
    with np.load(_ASSET_MEL_FILTERS, allow_pickle=False) as payload:
        filters = torch.from_numpy(payload["mel_80"])
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.unsqueeze(0)


def build_initial_tokens(cfg: WhisperTinyConfig) -> list[int]:
    return [
        cfg.decoder_start_token_id,
        cfg.english_token_id,
        cfg.transcribe_token_id,
        cfg.no_timestamps_token_id,
    ]


def build_padded_tokens(
    cfg: WhisperTinyConfig,
    tokens: list[int],
    *,
    decode_tokens: int,
) -> torch.Tensor:
    if len(tokens) > decode_tokens:
        raise ValueError(f"Token prefix length {len(tokens)} exceeds decode window {decode_tokens}")
    padded = [cfg.pad_token_id] * decode_tokens
    padded[: len(tokens)] = tokens
    return torch.tensor([padded], dtype=torch.long)


def export_onnx(
    path: str | Path,
    *,
    model: WhisperTinyForTranspile,
    mel: torch.Tensor,
    tokens: torch.Tensor,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (mel, tokens),
        out_path,
        input_names=["mel", "tokens"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )
    return out_path
