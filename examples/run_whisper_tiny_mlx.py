from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from examples.common import (
    add_output_dir_argument,
    export_and_transpile,
    load_generated_module,
)
from tnnx.config import CompileConfig


def run_demo(
    out_root: str | Path = "examples/out",
    *,
    rtol: float = 2e-4,
    atol: float = 2e-5,
) -> dict[str, str | float]:
    import torch

    from .model_whisper_tiny import (
        TinyWhisper,
        WhisperTinyConfig,
        build_demo_mel,
        build_demo_tokens,
        decode_token_ids,
        export_onnx,
    )

    torch.manual_seed(7)
    np.random.seed(7)

    cfg = WhisperTinyConfig()
    model = TinyWhisper(cfg).eval()
    mel = build_demo_mel(cfg, seed=7)
    tokens = build_demo_tokens(cfg, seed=11)

    artifacts = export_and_transpile(
        output_dir=out_root,
        onnx_name="whisper_tiny.onnx",
        export_fn=lambda path: export_onnx(
            path,
            model=model,
            sample_mel=mel,
            sample_tokens=tokens,
        ),
        target="mlx",
        generated_dir_name="generated_whisper_tiny_mlx",
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )

    assert artifacts.generated_entrypoint is not None
    assert artifacts.weights_path is not None
    module = load_generated_module(
        artifacts.generated_entrypoint,
        module_name="generated_whisper_tiny_mlx",
    )
    params = module.load_weights(str(artifacts.weights_path))

    expected = model(mel, tokens).detach().cpu().numpy()
    actual = np.asarray(
        module.forward(
            params,
            {
                "mel": mel.detach().cpu().numpy(),
                "tokens": tokens.detach().cpu().numpy(),
            },
        )["logits"]
    )

    max_abs = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"Whisper tiny MLX parity failed: max_abs={max_abs}, rtol={rtol}, atol={atol}"
        )

    predicted_ids = np.argmax(actual[0], axis=-1)
    predicted_text = decode_token_ids(torch.from_numpy(predicted_ids), cfg)
    prompt_text = decode_token_ids(tokens[0].detach().cpu(), cfg)

    return {
        "onnx_path": str(artifacts.onnx_path),
        "graph_path": str(artifacts.graph_path),
        "generated_module": str(artifacts.generated_entrypoint),
        "weights_path": str(artifacts.weights_path),
        "max_abs": max_abs,
        "prompt_text": prompt_text,
        "predicted_text": predicted_text,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Whisper tiny, transpile it to MLX, and run a parity check."
    )
    add_output_dir_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import mlx.core  # noqa: F401
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance only
        raise SystemExit(
            "This demo requires optional deps: torch and mlx.\n"
            "Run: uv sync --dev\n"
            "Then: uv run python -m examples.run_whisper_tiny_mlx"
        ) from exc

    result = run_demo(args.output_dir)
    print("Whisper tiny MLX conversion + parity succeeded.")
    print(f"ONNX: {result['onnx_path']}")
    print(f"Graph IR: {result['graph_path']}")
    print(f"Generated: {result['generated_module']}")
    print(f"Weights: {result['weights_path']}")
    print(f"Max abs diff: {result['max_abs']:.6e}")
    print(f"Prompt text: {result['prompt_text']!r}")
    print(f"Predicted text: {result['predicted_text']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
