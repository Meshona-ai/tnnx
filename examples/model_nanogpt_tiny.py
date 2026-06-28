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
class NanoGPTTinyConfig:
    vocab_size: int = 64
    seq_len: int = 8
    embed_dim: int = 16
    n_heads: int = 4
    mlp_hidden_dim: int = 64


class TinyNanoGPT(nn.Module):
    """Single-block NanoGPT-style toy model for conversion tests."""

    def __init__(self, cfg: NanoGPTTinyConfig) -> None:
        super().__init__()
        if cfg.embed_dim % cfg.n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.cfg = cfg
        self.head_dim = cfg.embed_dim // cfg.n_heads
        self.scale = self.head_dim**-0.5

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.embed_dim))

        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.q_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.k_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.v_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.attn_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp_up = nn.Linear(cfg.embed_dim, cfg.mlp_hidden_dim)
        self.mlp_down = nn.Linear(cfg.mlp_hidden_dim, cfg.embed_dim)

        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch, self.cfg.seq_len, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, self.cfg.seq_len, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, self.cfg.seq_len, self.cfg.n_heads, self.head_dim).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v)
        return y.transpose(1, 2).reshape(batch, self.cfg.seq_len, self.cfg.embed_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx) + self.pos_emb
        x = x + self.attn_proj(self._self_attention(self.ln1(x)))
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.ln2(x))))
        x = self.ln_f(x)
        return self.lm_head(x)


def build_demo_tokens(
    cfg: NanoGPTTinyConfig,
    *,
    seed: int = 7,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        cfg.vocab_size,
        (1, cfg.seq_len),
        dtype=torch.long,
        generator=generator,
    )


def export_onnx(
    path: str | Path,
    *,
    model: TinyNanoGPT | None = None,
    sample_tokens: torch.Tensor | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_model = model or TinyNanoGPT(NanoGPTTinyConfig())
    resolved_model = resolved_model.eval()
    tokens = sample_tokens if sample_tokens is not None else build_demo_tokens(resolved_model.cfg)
    torch.onnx.export(
        resolved_model,
        tokens,
        out_path,
        input_names=["idx"],
        output_names=["y"],
        opset_version=18,
        dynamo=False,
    )
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the tiny NanoGPT example and transpile it."
    )
    add_output_dir_argument(parser)
    add_target_argument(parser, default="jax")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = export_and_transpile(
        output_dir=args.output_dir,
        onnx_name="nanogpt_tiny.onnx",
        export_fn=export_onnx,
        target=args.target,
    )
    print_artifact_summary("NanoGPT Tiny", artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
