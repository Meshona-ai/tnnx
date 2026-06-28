from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from examples.common import (
    add_output_dir_argument,
    export_and_transpile,
    load_generated_module,
)
from tnnx.config import CompileConfig


def run_demo(out_root: str | Path = "examples/out") -> dict[str, str | float]:
    return run_demo_with_model(out_root)


def run_demo_with_model(
    out_root: str | Path = "examples/out",
    *,
    model: Any | None = None,
    sample_tokens: Any | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, str | float]:
    import torch

    from .model_nanogpt_tiny import NanoGPTTinyConfig, TinyNanoGPT, build_demo_tokens, export_onnx

    torch.manual_seed(7)
    np.random.seed(7)
    cfg = NanoGPTTinyConfig()
    resolved_model = model if model is not None else TinyNanoGPT(cfg)
    resolved_model = resolved_model.eval()
    tokens = sample_tokens if sample_tokens is not None else build_demo_tokens(cfg, seed=7)

    artifacts = export_and_transpile(
        output_dir=out_root,
        onnx_name="nanogpt_tiny.onnx",
        export_fn=lambda path: export_onnx(path, model=resolved_model, sample_tokens=tokens),
        target="jax",
        generated_dir_name="generated_nanogpt_tiny_jax",
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )

    assert artifacts.generated_entrypoint is not None
    assert artifacts.weights_path is not None
    module = load_generated_module(
        artifacts.generated_entrypoint,
        module_name="generated_nanogpt_tiny_jax",
    )
    params = module.load_weights(str(artifacts.weights_path))
    expected = resolved_model(tokens).detach().cpu().numpy()
    actual = np.asarray(module.forward(params, {"idx": tokens.detach().cpu().numpy()})["y"])

    max_abs = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"NanoGPT tiny JAX parity failed: max_abs={max_abs}, rtol={rtol}, atol={atol}"
        )

    return {
        "onnx_path": str(artifacts.onnx_path),
        "graph_path": str(artifacts.graph_path),
        "generated_module": str(artifacts.generated_entrypoint),
        "weights_path": str(artifacts.weights_path),
        "max_abs": max_abs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export NanoGPT tiny, transpile it to JAX, and run a parity check."
    )
    add_output_dir_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import jax  # noqa: F401
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance only
        raise SystemExit("This demo requires optional deps: torch and jax.") from exc

    result = run_demo(args.output_dir)
    print("NanoGPT tiny conversion + parity succeeded.")
    print(f"ONNX: {result['onnx_path']}")
    print(f"Graph IR: {result['graph_path']}")
    print(f"Generated: {result['generated_module']}")
    print(f"Weights: {result['weights_path']}")
    print(f"Max abs diff: {result['max_abs']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
