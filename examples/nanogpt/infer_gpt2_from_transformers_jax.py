from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from examples.common import add_output_dir_argument, export_and_transpile, load_generated_module
from tnnx.config import CompileConfig

_DEFAULT_MODEL_ID = "sshleifer/tiny-gpt2"
_DEFAULT_PROMPT = "Hello from tnnx"
_DEFAULT_CONTEXT_WINDOW = 128
_DEFAULT_MAX_NEW_TOKENS = 16
_DEFAULT_TEMPERATURE = 0.8
_DEFAULT_TOP_K = 50
_DEFAULT_REPETITION_PENALTY = 1.15
_DEFAULT_SEED = 0
_PARITY_RTOL = 2e-4
_PARITY_ATOL = 2e-5


def _slug(value: str) -> str:
    out: list[str] = []
    last_was_sep = False
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
            last_was_sep = False
            continue
        if not last_was_sep:
            out.append("_")
            last_was_sep = True
    return "".join(out).strip("_")


def _prepare_tokenizer(tokenizer: Any) -> Any:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer must expose eos_token_id.")
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("Tokenizer must expose eos_token so pad_token can be derived.")
        setattr(tokenizer, "pad_token", eos_token)
        if getattr(tokenizer, "pad_token_id", None) is None:
            setattr(tokenizer, "pad_token_id", int(eos_token_id))
    return tokenizer


def _load_pretrained_model_and_tokenizer(
    model_id: str,
    *,
    local_files_only: bool,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer, GPT2LMHeadModel
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance only
        raise SystemExit(
            "This example requires the optional dev dependencies.\nRun: uv sync --dev"
        ) from exc

    try:
        model = GPT2LMHeadModel.from_pretrained(model_id, local_files_only=local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    except OSError as exc:  # pragma: no cover - depends on local cache/network state
        raise SystemExit(
            f"Could not load {model_id!r} from transformers. "
            "Make sure the checkpoint is cached locally or available from Hugging Face."
        ) from exc

    tokenizer = _prepare_tokenizer(tokenizer)
    model = model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.pad_token_id = int(tokenizer.pad_token_id)
    return model, tokenizer


def _build_export_model(model: Any) -> Any:
    import torch.nn as nn

    class GPT2ExportWrapper(nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            return outputs.logits

    return GPT2ExportWrapper(model).eval()


def _build_inputs(
    ids: list[int],
    *,
    seq_len: int,
    pad_id: int,
) -> tuple[dict[str, np.ndarray], int]:
    window = [int(token) for token in ids[-seq_len:]]
    valid_len = len(window)
    if valid_len < seq_len:
        window = window + ([int(pad_id)] * (seq_len - valid_len))
    attention_mask = ([1] * valid_len) + ([0] * (seq_len - valid_len))
    return (
        {
            "input_ids": np.asarray([window], dtype=np.int64),
            "attention_mask": np.asarray([attention_mask], dtype=np.int64),
        },
        valid_len,
    )


def _export_onnx(path: str | Path, *, model: Any, sample_inputs: dict[str, np.ndarray]) -> Path:
    import torch

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = _build_export_model(model)
    torch.onnx.export(
        export_model,
        (
            torch.from_numpy(sample_inputs["input_ids"]),
            torch.from_numpy(sample_inputs["attention_mask"]),
        ),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )
    return out_path


def _validate_onnx(onnx_path: Path) -> None:
    from tnnx.ingest.op_map import ONNX_TO_SEMANTIC

    exported = onnx.load(onnx_path)
    exported_ops = {node.op_type for node in exported.graph.node}
    unsupported = sorted(exported_ops - (set(ONNX_TO_SEMANTIC.keys()) | {"Constant"}))
    if unsupported:
        raise ValueError(f"Exported GPT-2 graph has unsupported ops: {unsupported}")


def _sample_next_token(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    recent_ids: list[int],
    repetition_penalty: float,
    rng: np.random.Generator,
) -> int:
    scores = np.asarray(logits, dtype=np.float64).copy()
    if repetition_penalty > 1:
        for token in set(recent_ids):
            token_id = int(token)
            value = float(scores[token_id])
            if value < 0:
                scores[token_id] = value * repetition_penalty
            else:
                scores[token_id] = value / repetition_penalty
    if temperature <= 0:
        return int(np.argmax(scores))

    scores /= float(temperature)
    if top_k > 0 and top_k < scores.shape[0]:
        keep = np.argpartition(scores, -top_k)[-top_k:]
        filtered = np.full_like(scores, -np.inf)
        filtered[keep] = scores[keep]
        scores = filtered
    finite = np.isfinite(scores)
    if not np.any(finite):
        return int(np.argmax(logits))
    max_score = float(np.max(scores[finite]))
    probs = np.zeros_like(scores, dtype=np.float64)
    probs[finite] = np.exp(scores[finite] - max_score)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(logits))
    probs /= total
    return int(rng.choice(scores.shape[0], p=probs))


def run_demo(
    out_dir: str | Path = "examples/nanogpt/out/gpt2",
    *,
    prompt: str = _DEFAULT_PROMPT,
    model_id: str = _DEFAULT_MODEL_ID,
    context_window: int = _DEFAULT_CONTEXT_WINDOW,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
    top_k: int = _DEFAULT_TOP_K,
    repetition_penalty: float = _DEFAULT_REPETITION_PENALTY,
    seed: int = _DEFAULT_SEED,
    local_files_only: bool = False,
    model: Any | None = None,
    tokenizer: Any | None = None,
    rtol: float = _PARITY_RTOL,
    atol: float = _PARITY_ATOL,
) -> dict[str, str | float | int]:
    import torch

    if (model is None) != (tokenizer is None):
        raise ValueError("Provide both model and tokenizer together, or provide neither.")

    if model is None:
        model, tokenizer = _load_pretrained_model_and_tokenizer(
            model_id,
            local_files_only=local_files_only,
        )
    else:
        tokenizer = _prepare_tokenizer(tokenizer)
        if hasattr(model, "config"):
            model.config.use_cache = False
            model.config.pad_token_id = int(tokenizer.pad_token_id)
        model = model.eval() if hasattr(model, "eval") else model

    if context_window < 1:
        raise ValueError("context_window must be >= 1.")
    if top_k < 0:
        raise ValueError("top_k must be >= 0.")
    if repetition_penalty < 1:
        raise ValueError("repetition_penalty must be >= 1.")

    position_limit = int(getattr(getattr(model, "config", None), "n_positions", context_window))
    if context_window > position_limit:
        raise ValueError(f"context_window must be <= model.config.n_positions ({position_limit}).")

    prompt_ids = [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)
    if not prompt_ids:
        prompt_ids = [eos_id]

    sample_inputs, _ = _build_inputs(prompt_ids, seq_len=context_window, pad_id=pad_id)
    slug = _slug(model_id)
    artifacts = export_and_transpile(
        output_dir=out_dir,
        onnx_name=f"{slug}.onnx",
        export_fn=lambda path: _export_onnx(path, model=model, sample_inputs=sample_inputs),
        after_export=_validate_onnx,
        target="jax",
        generated_dir_name=f"generated_{slug}_jax",
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )

    assert artifacts.generated_entrypoint is not None
    assert artifacts.weights_path is not None
    module = load_generated_module(
        artifacts.generated_entrypoint,
        module_name=f"generated_{slug}_jax",
    )
    params = module.load_weights(str(artifacts.weights_path))

    export_model = _build_export_model(model)
    with torch.no_grad():
        expected = (
            export_model(
                torch.from_numpy(sample_inputs["input_ids"]),
                torch.from_numpy(sample_inputs["attention_mask"]),
            )
            .detach()
            .cpu()
            .numpy()
        )
    actual = np.asarray(module.forward(params, sample_inputs)["logits"])
    max_abs = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"GPT-2 JAX parity failed: max_abs={max_abs}, rtol={rtol}, atol={atol}"
        )

    running_ids = list(prompt_ids)
    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    rng = np.random.default_rng(seed)
    for _ in range(max_new_tokens):
        model_inputs, valid_len = _build_inputs(
            running_ids,
            seq_len=context_window,
            pad_id=pad_id,
        )
        logits = np.asarray(module.forward(params, model_inputs)["logits"])
        recent_ids = [int(token) for token in model_inputs["input_ids"][0, :valid_len]]
        next_id = _sample_next_token(
            logits[0, valid_len - 1],
            temperature=temperature,
            top_k=top_k,
            recent_ids=recent_ids,
            repetition_penalty=repetition_penalty,
            rng=rng,
        )
        running_ids.append(next_id)
        generated_ids.append(next_id)
        if next_id == eos_id:
            stop_reason = "eos"
            break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(running_ids, skip_special_tokens=True)
    return {
        "model_id": model_id,
        "onnx_path": str(artifacts.onnx_path),
        "graph_path": str(artifacts.graph_path),
        "generated_module": str(artifacts.generated_entrypoint),
        "weights_path": str(artifacts.weights_path),
        "max_abs": max_abs,
        "generated_token_count": len(generated_ids),
        "generated_text": generated_text,
        "full_text": full_text,
        "stop_reason": stop_reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a pretrained GPT-2 checkpoint from transformers, transpile it to JAX, "
            "and run autoregressive inference."
        )
    )
    parser.add_argument(
        "--model-id",
        default=_DEFAULT_MODEL_ID,
        help="Hugging Face model id passed to GPT2LMHeadModel.from_pretrained().",
    )
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT, help="Prompt text for generation.")
    add_output_dir_argument(parser, default="examples/nanogpt/out/gpt2")
    parser.add_argument(
        "--context-window",
        type=int,
        default=_DEFAULT_CONTEXT_WINDOW,
        help="Static token window exported into the ONNX/JAX graph.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=_DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of autoregressive tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_DEFAULT_TEMPERATURE,
        help="Sampling temperature. Use 0 for deterministic argmax decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help="Sample only from the top-k logits. Use 0 to disable filtering.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=_DEFAULT_REPETITION_PENALTY,
        help="Penalize tokens already present in the current context window. Use 1 to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Random seed used for token sampling.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the model/tokenizer from the local Hugging Face cache only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_demo(
        out_dir=args.output_dir,
        prompt=str(args.prompt),
        model_id=str(args.model_id),
        context_window=int(args.context_window),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        seed=int(args.seed),
        local_files_only=bool(args.local_files_only),
    )
    print("=== GPT-2 From Transformers ===")
    print(f"Model id: {result['model_id']}")
    print(f"ONNX: {result['onnx_path']}")
    print(f"Graph IR: {result['graph_path']}")
    print(f"Generated: {result['generated_module']}")
    print(f"Weights: {result['weights_path']}")
    print(f"Parity max abs diff: {result['max_abs']:.6e}")
    print(f"Generated tokens: {result['generated_token_count']}")
    print("Generated text:")
    print(str(result["generated_text"]))
    print("Full text:")
    print(str(result["full_text"]))
    print(f"Stop reason: {result['stop_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
