from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from examples.common import (
    add_output_dir_argument,
    add_target_argument,
    export_and_transpile,
    print_artifact_summary,
)


class TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv(x))
        y = y.reshape(y.shape[0], -1)
        return self.fc(y)


def export_onnx(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = TinyConv().eval()
    sample = torch.randn(1, 3, 16, 16)
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
        description="Export the tiny convolution example and transpile it."
    )
    add_output_dir_argument(parser)
    add_target_argument(parser, default="jax")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = export_and_transpile(
        output_dir=args.output_dir,
        onnx_name="tiny_conv.onnx",
        export_fn=export_onnx,
        target=args.target,
    )
    print_artifact_summary("Tiny Conv", artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
