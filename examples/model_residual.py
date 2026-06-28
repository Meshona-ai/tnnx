from __future__ import annotations

import argparse
from collections.abc import Sequence
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


class ResidualMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 12)
        self.ff1 = nn.Linear(12, 24)
        self.ff2 = nn.Linear(24, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.proj(x)
        y = self.ff2(F.gelu(self.ff1(skip)))
        return y + skip


def export_onnx(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = ResidualMLP().eval()
    sample = torch.randn(1, 12)
    torch.onnx.export(
        model,
        sample,
        out_path,
        input_names=["x"],
        output_names=["y"],
        opset_version=18,
        dynamo=False,
    )
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the residual MLP example and transpile it."
    )
    add_output_dir_argument(parser)
    add_target_argument(parser, default="jax")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = export_and_transpile(
        output_dir=args.output_dir,
        onnx_name="residual_mlp.onnx",
        export_fn=export_onnx,
        target=args.target,
    )
    print_artifact_summary("Residual MLP", artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
