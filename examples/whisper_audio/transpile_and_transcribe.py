from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from examples.common import add_output_dir_argument, load_generated_module  # noqa: E402

from .runtime_env import ensure_runtime_paths, resolve_model_snapshot  # noqa: E402
from .whisper_hf_tiny_source import (  # noqa: E402
    DEFAULT_DECODE_TOKENS,
    WhisperTinyForTranspile,
    build_initial_tokens,
    build_padded_tokens,
    export_onnx,
    log_mel_spectrogram,
)


def _unsupported_ops(onnx_path: Path) -> list[str]:
    from tnnx.ingest.op_map import ONNX_TO_SEMANTIC

    exported = onnx.load(onnx_path)
    exported_ops = {node.op_type for node in exported.graph.node}
    return sorted(exported_ops - (set(ONNX_TO_SEMANTIC.keys()) | {"Constant"}))


def _decode_text(tokenizer: Any, token_ids: list[int]) -> str:
    filtered = [int(v) for v in token_ids if int(v) < 50364]
    return tokenizer.decode(filtered, skip_special_tokens=True).strip()


def run_demo(
    audio_path: str | Path,
    *,
    out_dir: str | Path = "examples/whisper_audio/out",
    decode_tokens: int = DEFAULT_DECODE_TOKENS,
    max_new_tokens: int = 48,
) -> dict[str, str | int]:
    from tnnx.api import transpile_onnx
    from tnnx.config import CompileConfig

    ensure_runtime_paths(require_mlx=True)
    import mlx.core as mx
    from tokenizers import Tokenizer

    audio = Path(audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    if decode_tokens > 448:
        raise ValueError("decode_tokens must be <= 448 for openai/whisper-tiny")

    model_snapshot = resolve_model_snapshot()
    model, cfg = WhisperTinyForTranspile.from_hf_snapshot(model_snapshot)
    mel = log_mel_spectrogram(audio)
    prompt_tokens = build_initial_tokens(cfg)
    sample_tokens = build_padded_tokens(cfg, prompt_tokens, decode_tokens=decode_tokens)

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_onnx(
        target_dir / f"openai_whisper_tiny_{decode_tokens}.onnx",
        model=model,
        mel=mel,
        tokens=sample_tokens,
    )
    unsupported = _unsupported_ops(onnx_path)
    if unsupported:
        raise ValueError(f"Exported Whisper graph has unsupported ops: {unsupported}")

    generated_dir = target_dir / f"generated_mlx_{decode_tokens}"
    manifest = transpile_onnx(
        str(onnx_path),
        "mlx",
        str(generated_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )
    module = load_generated_module(
        generated_dir / "model_mlx.py",
        module_name="generated_whisper_hf_tiny_mlx",
    )
    params = module.load_weights(str(manifest.weights_file))

    mel_np = mel.detach().cpu().numpy().astype(np.float32)
    running_tokens = list(prompt_tokens)
    max_steps = min(max_new_tokens, decode_tokens - len(prompt_tokens))

    suppress_tokens = set(int(v) for v in cfg.suppress_tokens)
    begin_suppress_tokens = set(int(v) for v in cfg.begin_suppress_tokens)
    timestamp_begin = cfg.no_timestamps_token_id + 1

    for step in range(max_steps):
        padded = build_padded_tokens(cfg, running_tokens, decode_tokens=decode_tokens)
        logits = module.forward(
            params,
            {
                "mel": mel_np,
                "tokens": padded.detach().cpu().numpy(),
            },
        )["logits"]
        next_logits = np.asarray(logits)[0, len(running_tokens) - 1].astype(np.float32, copy=True)
        next_logits[list(suppress_tokens)] = -1e9
        next_logits[timestamp_begin:] = -1e9
        if step == 0:
            next_logits[list(begin_suppress_tokens)] = -1e9
        next_token = int(np.argmax(next_logits))
        running_tokens.append(next_token)
        if next_token == cfg.eos_token_id:
            break

    tokenizer = Tokenizer.from_file(str(model_snapshot / "tokenizer.json"))
    generated_tokens = running_tokens[len(prompt_tokens) :]
    if generated_tokens and generated_tokens[-1] == cfg.eos_token_id:
        generated_tokens = generated_tokens[:-1]
    text = _decode_text(tokenizer, generated_tokens)

    text_path = target_dir / f"{audio.stem}.transcript.txt"
    json_path = target_dir / f"{audio.stem}.transcript.json"
    text_path.write_text(f"{text}\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "audio": str(audio),
                "onnx": str(onnx_path),
                "graph_ir": str(generated_dir / "graph_ir.json"),
                "generated_module": str(generated_dir / "model_mlx.py"),
                "weights": str(manifest.weights_file),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "text": text,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    mx.eval(params)
    return {
        "audio": str(audio),
        "onnx": str(onnx_path),
        "graph_ir": str(generated_dir / "graph_ir.json"),
        "generated_module": str(generated_dir / "model_mlx.py"),
        "weights": str(manifest.weights_file),
        "text_file": str(text_path),
        "json_file": str(json_path),
        "text": text,
        "generated_token_count": len(generated_tokens),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export openai/whisper-tiny from local HF weights, transpile to MLX, "
            "and run generated MLX transcription."
        )
    )
    parser.add_argument("--audio", default="examples/terminator.mp3", help="Input audio file.")
    add_output_dir_argument(parser, default="examples/whisper_audio/out")
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=DEFAULT_DECODE_TOKENS,
        help="Fixed decoder context exported into the MLX graph (<=448).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Maximum greedy decode steps after the Whisper prompt tokens.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = run_demo(
        args.audio,
        out_dir=args.output_dir,
        decode_tokens=int(args.decode_tokens),
        max_new_tokens=int(args.max_new_tokens),
    )
    print("=== Whisper Tiny Via Transpiled MLX ===")
    print(f"Audio: {result['audio']}")
    print(f"ONNX: {result['onnx']}")
    print(f"Graph IR: {result['graph_ir']}")
    print(f"Generated MLX: {result['generated_module']}")
    print(f"Weights: {result['weights']}")
    print(f"Generated tokens: {result['generated_token_count']}")
    print(f"Transcript: {result['text']}")
    print(f"Text file: {result['text_file']}")
    print(f"JSON file: {result['json_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
