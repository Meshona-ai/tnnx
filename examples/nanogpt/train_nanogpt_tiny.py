from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from examples.common import add_output_dir_argument, load_generated_module
from examples.model_nanogpt_tiny import NanoGPTTinyConfig
from examples.nanogpt.utils import build_window
from examples.run_nanogpt_tiny_jax import run_demo_with_model

_VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."
_DEFAULT_CORPUS = (
    "hello jax this tiny model speaks in short sentences. "
    "the transpiler converts pytorch onnx and jax correctly. "
    "machine learning examples should produce meaningful text. "
) * 40
_PARITY_RTOL = 5e-5
_PARITY_ATOL = 2e-5
_CHECKPOINT_NAME = "nanogpt_tiny_train_checkpoint.pt"


def _decode_ids(ids: np.ndarray, itos: dict[int, str]) -> str:
    return "".join(itos.get(int(token), "?") for token in ids.tolist())


def _encode_text(prompt: str, stoi: dict[str, int]) -> list[int]:
    return [stoi.get(ch, stoi[" "]) for ch in prompt]


def _window_input(ids: list[int], seq_len: int, pad_id: int) -> np.ndarray:
    return np.asarray([build_window(ids, seq_len, pad_id)], dtype=np.int64)


def _corpus_ids(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi.get(ch, stoi[" "]) for ch in text]


def _sample_batch(
    ids: list[int],
    *,
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    max_start = len(ids) - seq_len - 1
    starts = rng.integers(0, max_start + 1, size=batch_size)
    x = np.stack([np.asarray(ids[s : s + seq_len], dtype=np.int64) for s in starts], axis=0)
    y = np.stack([np.asarray(ids[s + 1 : s + seq_len + 1], dtype=np.int64) for s in starts], axis=0)
    return x, y


def _read_corpus(corpus_file: str | None) -> str:
    if corpus_file is None:
        return _DEFAULT_CORPUS
    text = Path(corpus_file).read_text(encoding="utf-8")
    return text if text else _DEFAULT_CORPUS


def _train_tiny_model(
    cfg: NanoGPTTinyConfig,
    *,
    stoi: dict[str, int],
    corpus_text: str,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[Any, float]:
    import torch
    import torch.nn.functional as F

    from examples.model_nanogpt_tiny import TinyNanoGPT

    data_ids = _corpus_ids(corpus_text, stoi)
    if len(data_ids) <= cfg.seq_len + 1:
        raise ValueError("Corpus is too short for training. Provide a longer corpus text.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = TinyNanoGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    loss_value = 0.0
    for _ in range(max(1, steps)):
        x_np, y_np = _sample_batch(data_ids, seq_len=cfg.seq_len, batch_size=batch_size, rng=rng)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
    model.eval()
    return model, loss_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the toy NanoGPT tiny checkpoint, then transpile and run it on JAX."
    )
    parser.add_argument("--prompt", default="hello jax", help="Prompt text for demo inference.")
    add_output_dir_argument(parser, default="examples/nanogpt/out")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Maximum number of autoregressive tokens to generate.",
    )
    parser.add_argument(
        "--eos-char",
        default=".",
        help="Stop generation when this character is produced.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1200,
        help="Training steps for the tiny model when no checkpoint is present.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for tiny training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Learning rate for tiny training.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if a checkpoint exists.",
    )
    parser.add_argument(
        "--corpus-file",
        default=None,
        help="Optional text file for tiny training corpus.",
    )
    return parser


def main() -> int:
    cfg = NanoGPTTinyConfig()
    if len(_VOCAB) != cfg.vocab_size:
        raise ValueError(
            f"Demo vocab size {len(_VOCAB)} must equal model vocab_size {cfg.vocab_size}."
        )

    parser = build_parser()
    args = parser.parse_args()
    prompt = str(args.prompt)
    out_dir = Path(args.output_dir)
    max_new_tokens = int(args.max_new_tokens)
    eos_char = str(args.eos_char)
    train_steps = int(args.train_steps)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    retrain = bool(args.retrain)
    corpus_file = str(args.corpus_file) if args.corpus_file is not None else None

    try:
        import jax  # noqa: F401
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing optional deps. Run:\n"
            "uv sync --dev\n"
            "uv run python -m examples.nanogpt.train_nanogpt_tiny"
        ) from exc

    stoi = {ch: i for i, ch in enumerate(_VOCAB)}
    itos = {i: ch for i, ch in enumerate(_VOCAB)}
    if eos_char not in stoi:
        raise ValueError(f"eos char {eos_char!r} is not in demo vocab.")
    eos_id = stoi[eos_char]

    checkpoint_path = out_dir / _CHECKPOINT_NAME
    if checkpoint_path.exists() and not retrain:
        from examples.model_nanogpt_tiny import TinyNanoGPT

        model = TinyNanoGPT(cfg)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        train_info = f"loaded checkpoint: {checkpoint_path}"
        final_loss = None
    else:
        corpus = _read_corpus(corpus_file)
        model, final_loss = _train_tiny_model(
            cfg,
            stoi=stoi,
            corpus_text=corpus,
            steps=train_steps,
            batch_size=batch_size,
            lr=lr,
            seed=7,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        train_info = f"trained and saved checkpoint: {checkpoint_path}"

    prompt_ids = _encode_text(prompt, stoi)
    if not prompt_ids:
        prompt_ids = [stoi[" "]]
    sample_tokens = torch.from_numpy(_window_input(prompt_ids, cfg.seq_len, stoi[" "]))

    parity = run_demo_with_model(
        out_dir,
        model=model,
        sample_tokens=sample_tokens,
        rtol=_PARITY_RTOL,
        atol=_PARITY_ATOL,
    )
    generated_module = Path(str(parity["generated_module"]))
    weights_path = Path(str(parity["weights_path"]))
    module = load_generated_module(generated_module, module_name="generated_nanogpt_tiny_jax")
    params = module.load_weights(str(weights_path))
    running_ids = list(prompt_ids)
    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        x = _window_input(running_ids, cfg.seq_len, stoi[" "])
        logits = np.asarray(module.forward(params, {"idx": x})["y"])
        next_id = int(logits[0, -1].argmax())
        running_ids.append(next_id)
        generated_ids.append(next_id)
        if next_id == eos_id:
            stop_reason = "eos"
            break

    print("=== Train NanoGPT Tiny ===")
    print(f"ONNX path: {parity['onnx_path']}")
    print(f"Graph IR: {parity['graph_path']}")
    print(f"Generated module: {generated_module}")
    print(f"Weights: {weights_path}")
    print(f"Parity max abs diff: {float(parity['max_abs']):.6e}")
    print(f"Parity tolerance: rtol={_PARITY_RTOL}, atol={_PARITY_ATOL}")
    print(f"Checkpoint: {train_info}")
    if final_loss is not None:
        print(f"Final train loss: {final_loss:.6f}")
    print("")
    print(f"Prompt: {prompt!r}")
    print(f"Prompt token ids: {prompt_ids}")
    print(f"Generated token ids: {generated_ids}")
    print(f"Generated text: {_decode_ids(np.asarray(generated_ids), itos)!r}")
    print(f"Full text: {_decode_ids(np.asarray(running_ids), itos)!r}")
    print(f"Stop reason: {stop_reason}")
    if generated_ids:
        next_token_id = int(generated_ids[-1])
        print(f"Last generated token: id={next_token_id}, char={itos.get(next_token_id, '?')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
