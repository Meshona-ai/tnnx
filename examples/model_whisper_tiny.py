from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.common import (
    add_output_dir_argument,
    add_target_argument,
    export_and_transpile,
    print_artifact_summary,
)


@dataclass(frozen=True, slots=True)
class WhisperTinyConfig:
    n_mels: int = 80
    audio_ctx: int = 32
    text_ctx: int = 12
    d_model: int = 64
    n_heads: int = 4
    n_audio_layers: int = 2
    n_text_layers: int = 2
    mlp_hidden_dim: int = 256
    vocab: str = " _abcdefghijklmnopqrstuvwxyz'.,?"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def audio_tokens(self) -> int:
        if self.audio_ctx % 2 != 0:
            raise ValueError("audio_ctx must be divisible by 2 after stride-2 conv front-end.")
        return self.audio_ctx // 2


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, *, kv: torch.Tensor | None = None) -> torch.Tensor:
        source = x if kv is None else kv
        batch = x.shape[0]
        q_len = x.shape[1]
        kv_len = source.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(source)
        v = self.v_proj(source)

        q = q.reshape(batch, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).reshape(batch, q_len, self.n_heads * self.head_dim)
        return self.out_proj(y)


class WhisperEncoderBlock(nn.Module):
    def __init__(self, cfg: WhisperTinyConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp_up = nn.Linear(cfg.d_model, cfg.mlp_hidden_dim)
        self.mlp_down = nn.Linear(cfg.mlp_hidden_dim, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.ln2(x))))
        return x


class WhisperDecoderBlock(nn.Module):
    def __init__(self, cfg: WhisperTinyConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.mlp_up = nn.Linear(cfg.d_model, cfg.mlp_hidden_dim)
        self.mlp_down = nn.Linear(cfg.mlp_hidden_dim, cfg.d_model)

    def forward(self, x: torch.Tensor, xa: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), kv=xa)
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.ln3(x))))
        return x


class TinyWhisper(nn.Module):
    """Whisper-style tiny model (Conv front-end + encoder/decoder transformer)."""

    def __init__(self, cfg: WhisperTinyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv1d(cfg.n_mels, cfg.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1)
        self.audio_pos = nn.Parameter(torch.zeros(1, cfg.audio_tokens, cfg.d_model))
        self.encoder = nn.ModuleList([WhisperEncoderBlock(cfg) for _ in range(cfg.n_audio_layers)])
        self.encoder_ln = nn.LayerNorm(cfg.d_model)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.text_pos = nn.Parameter(torch.zeros(1, cfg.text_ctx, cfg.d_model))
        self.decoder = nn.ModuleList([WhisperDecoderBlock(cfg) for _ in range(cfg.n_text_layers)])
        self.decoder_ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        x = x + self.audio_pos
        for block in self.encoder:
            x = block(x)
        return self.encoder_ln(x)

    def decode(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(tokens) + self.text_pos
        for block in self.decoder:
            x = block(x, audio_features)
        x = self.decoder_ln(x)
        return self.lm_head(x)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        audio_features = self.encode(mel)
        return self.decode(tokens, audio_features)


def build_demo_mel(cfg: WhisperTinyConfig, *, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        (1, cfg.n_mels, cfg.audio_ctx),
        generator=generator,
        dtype=torch.float32,
    )


def build_demo_tokens(cfg: WhisperTinyConfig, *, seed: int = 11) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        cfg.vocab_size,
        (1, cfg.text_ctx),
        dtype=torch.long,
        generator=generator,
    )


def decode_token_ids(ids: torch.Tensor, cfg: WhisperTinyConfig) -> str:
    return "".join(cfg.vocab[int(idx)] for idx in ids.tolist())


def export_onnx(
    path: str | Path,
    *,
    model: TinyWhisper | None = None,
    sample_mel: torch.Tensor | None = None,
    sample_tokens: torch.Tensor | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = WhisperTinyConfig()
    resolved_model = model or TinyWhisper(cfg)
    resolved_model = resolved_model.eval()
    mel = sample_mel if sample_mel is not None else build_demo_mel(resolved_model.cfg)
    tokens = sample_tokens if sample_tokens is not None else build_demo_tokens(resolved_model.cfg)

    torch.onnx.export(
        resolved_model,
        (mel, tokens),
        out_path,
        input_names=["mel", "tokens"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the tiny Whisper example and transpile it."
    )
    add_output_dir_argument(parser)
    add_target_argument(parser, default="jax")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = export_and_transpile(
        output_dir=args.output_dir,
        onnx_name="whisper_tiny.onnx",
        export_fn=export_onnx,
        target=args.target,
    )
    print_artifact_summary("Whisper Tiny", artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
