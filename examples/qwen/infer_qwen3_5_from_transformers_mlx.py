from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from examples.common import add_output_dir_argument, export_and_transpile, load_generated_module
from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig

from .infer_qwen3_5_from_transformers_jax import (
    _build_decode_step_inputs,
    _build_export_model,
    _build_inputs,
    _decoder_state_from_specs,
    _export_onnx,
    _has_meaningful_text,
    _inspect_onnx,
    _load_prebuilt_tokenizer,
    _materialize_prebuilt_onnx_assets,
    _next_decoder_state,
    _onnx_input_specs,
    _preferred_torch_device,
    _prepare_model,
    _prepare_tokenizer,
    _resolve_prebuilt_snapshot,
    _sample_next_token,
    _slug,
)
from .infer_qwen3_5_from_transformers_jax import (
    build_tiny_qwen3_5_config as _build_tiny_qwen3_5_config,
)

_DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
_DEFAULT_ONNX_MODEL_ID = "onnx-community/Qwen3.5-0.8B-ONNX"
_DEFAULT_PROMPT = "Write one short sentence about compilers."
_DEFAULT_CONTEXT_WINDOW = 128
_DEFAULT_MAX_NEW_TOKENS = 12
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_TOP_K = 0
_DEFAULT_REPETITION_PENALTY = 1.0
_DEFAULT_SEED = 0
_PARITY_RTOL = 3e-4
_PARITY_ATOL = 3e-5


def build_tiny_qwen3_5_config() -> Any:
    return _build_tiny_qwen3_5_config()


def _runtime_versions() -> dict[str, str]:
    import torch
    import transformers

    return {
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "onnx_version": str(onnx.__version__),
        "mlx_version": str(importlib.metadata.version("mlx")),
    }


def _run_prebuilt_onnx_demo(
    out_dir: str | Path,
    *,
    prompt: str,
    context_window: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
    local_files_only: bool,
    versions: dict[str, str],
) -> dict[str, Any]:
    snapshot_root = _resolve_prebuilt_snapshot(local_files_only=local_files_only)
    tokenizer = _load_prebuilt_tokenizer(snapshot_root)
    eos_id = int(tokenizer.eos_token_id)
    prompt_ids = [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
    if not prompt_ids:
        prompt_ids = [eos_id]
    if len(prompt_ids) > context_window:
        raise ValueError(
            f"Prompt token count ({len(prompt_ids)}) exceeds context_window ({context_window})."
        )

    embed_onnx_path, decoder_onnx_path = _materialize_prebuilt_onnx_assets(out_dir, snapshot_root)
    decoder_ops, decoder_unsupported = _inspect_onnx(decoder_onnx_path)
    if decoder_unsupported:
        raise ValueError(
            "Prebuilt Qwen decoder still has unsupported ops after decode-only specialization: "
            f"{list(decoder_unsupported)}"
        )

    embed_generated_dir = Path(out_dir) / "generated_qwen_embed_tokens_mlx"
    decoder_generated_dir = Path(out_dir) / "generated_qwen_decoder_merged_mlx"
    embed_manifest = transpile_onnx(
        str(embed_onnx_path),
        "mlx",
        str(embed_generated_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )
    decoder_manifest = transpile_onnx(
        str(decoder_onnx_path),
        "mlx",
        str(decoder_generated_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )

    embed_module = load_generated_module(
        embed_generated_dir / "model_mlx.py",
        module_name="generated_qwen_embed_tokens_mlx",
    )
    embed_params = embed_module.load_weights(str(embed_manifest.weights_file))
    decoder_module = load_generated_module(
        decoder_generated_dir / "model_mlx.py",
        module_name="generated_qwen_decoder_merged_mlx",
    )
    decoder_params = decoder_module.load_weights(str(decoder_manifest.weights_file))
    decoder_specs = _onnx_input_specs(decoder_onnx_path)
    state = _decoder_state_from_specs(decoder_specs)

    rng = np.random.default_rng(seed)
    current_token_id = int(prompt_ids[0])
    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    total_steps = len(prompt_ids) - 1 + max_new_tokens
    position = 0

    for step_idx in range(total_steps):
        token_inputs = {"input_ids": np.asarray([[current_token_id]], dtype=np.int64)}
        embed_outputs = embed_module.forward(embed_params, token_inputs)
        inputs_embeds = np.asarray(embed_outputs["inputs_embeds"])
        decoder_inputs = _build_decode_step_inputs(
            state=state,
            inputs_embeds=inputs_embeds,
            position=position,
        )
        outputs = decoder_module.forward(decoder_params, decoder_inputs)
        logits = np.asarray(outputs["logits"])
        state = _next_decoder_state(outputs)

        if step_idx < len(prompt_ids) - 1:
            current_token_id = int(prompt_ids[step_idx + 1])
            position += 1
            continue

        next_id = _sample_next_token(
            logits[0, 0],
            temperature=temperature,
            top_k=top_k,
            recent_ids=prompt_ids + generated_ids,
            repetition_penalty=repetition_penalty,
            rng=rng,
        )
        generated_ids.append(next_id)
        position += 1
        if position >= context_window:
            stop_reason = "context_window"
            break
        if next_id == eos_id:
            stop_reason = "eos"
            break
        current_token_id = next_id

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(prompt_ids + generated_ids, skip_special_tokens=True)
    if not generated_ids:
        raise AssertionError("Qwen3.5 MLX generation returned zero decoded tokens.")
    if not _has_meaningful_text(generated_text):
        raise AssertionError("Qwen3.5 MLX generation did not produce meaningful decoded text.")

    return {
        "model_id": _DEFAULT_ONNX_MODEL_ID,
        "prompt": prompt,
        "onnx_path": str(decoder_onnx_path),
        "embed_onnx_path": str(embed_onnx_path),
        "graph_path": str(decoder_generated_dir / "graph_ir.json"),
        "generated_module": str(decoder_generated_dir / "model_mlx.py"),
        "weights_path": str(decoder_manifest.weights_file),
        "embed_generated_module": str(embed_generated_dir / "model_mlx.py"),
        "embed_weights_path": str(embed_manifest.weights_file),
        "max_abs": 0.0,
        "generated_token_count": len(generated_ids),
        "generated_text": generated_text,
        "full_text": full_text,
        "stop_reason": stop_reason,
        "exported_ops": decoder_ops,
        "unsupported_ops": decoder_unsupported,
        **versions,
    }


def run_demo(
    out_dir: str | Path = "examples/qwen/out/qwen3_5_0_8b_mlx",
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
) -> dict[str, Any]:
    import torch

    if (model is None) != (tokenizer is None):
        raise ValueError("Provide both model and tokenizer together, or provide neither.")
    if context_window < 1:
        raise ValueError("context_window must be >= 1.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1.")
    if top_k < 0:
        raise ValueError("top_k must be >= 0.")
    if repetition_penalty < 1:
        raise ValueError("repetition_penalty must be >= 1.")

    versions = _runtime_versions()
    if model is None:
        if model_id != _DEFAULT_MODEL_ID:
            raise ValueError(
                "The real Qwen3.5 path uses the fixed ONNX-community checkpoint "
                f"{_DEFAULT_ONNX_MODEL_ID!r}. "
                "Pass a synthetic model/tokenizer pair to override the model id."
            )
        return _run_prebuilt_onnx_demo(
            out_dir,
            prompt=prompt,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            local_files_only=local_files_only,
            versions=versions,
        )

    tokenizer = _prepare_tokenizer(tokenizer)
    model = _prepare_model(model)
    if hasattr(model, "config"):
        model.config.pad_token_id = int(tokenizer.pad_token_id)

    position_limit = int(
        getattr(getattr(model, "config", None), "max_position_embeddings", context_window)
    )
    if context_window > position_limit:
        raise ValueError(
            f"context_window must be <= model.config.max_position_embeddings ({position_limit})."
        )

    prompt_ids = [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)
    if not prompt_ids:
        prompt_ids = [eos_id]

    sample_inputs, _ = _build_inputs(prompt_ids, seq_len=context_window, pad_id=pad_id)
    slug = _slug(model_id)
    export_diagnostics: dict[str, tuple[str, ...]] = {}

    def _after_export(onnx_path: Path) -> None:
        exported_ops, unsupported_ops = _inspect_onnx(onnx_path)
        export_diagnostics["exported_ops"] = exported_ops
        export_diagnostics["unsupported_ops"] = unsupported_ops
        if unsupported_ops:
            raise ValueError(f"Exported Qwen graph has unsupported ops: {list(unsupported_ops)}")

    artifacts = export_and_transpile(
        output_dir=out_dir,
        onnx_name=f"{slug}.onnx",
        export_fn=lambda path: _export_onnx(path, model=model, sample_inputs=sample_inputs),
        after_export=_after_export,
        target="mlx",
        generated_dir_name=f"generated_{slug}_mlx",
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )

    assert artifacts.generated_entrypoint is not None
    assert artifacts.weights_path is not None
    module = load_generated_module(
        artifacts.generated_entrypoint,
        module_name=f"generated_{slug}_mlx",
    )
    params = module.load_weights(str(artifacts.weights_path))

    device = _preferred_torch_device()
    export_model = _build_export_model(model).to(device)
    input_ids_t = torch.from_numpy(sample_inputs["input_ids"]).to(device)
    attention_mask_t = torch.from_numpy(sample_inputs["attention_mask"]).to(device)
    with torch.no_grad():
        expected = export_model(input_ids_t, attention_mask_t).detach().cpu().numpy()
    actual = np.asarray(module.forward(params, sample_inputs)["logits"])
    max_abs = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"Qwen3.5 MLX parity failed: max_abs={max_abs}, rtol={rtol}, atol={atol}"
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
    if not generated_ids:
        raise AssertionError("Qwen3.5 MLX generation returned zero decoded tokens.")
    if not _has_meaningful_text(generated_text):
        raise AssertionError("Qwen3.5 MLX generation did not produce meaningful decoded text.")

    return {
        "model_id": model_id,
        "prompt": prompt,
        "onnx_path": str(artifacts.onnx_path),
        "graph_path": str(artifacts.graph_path),
        "generated_module": str(artifacts.generated_entrypoint),
        "weights_path": str(artifacts.weights_path),
        "max_abs": max_abs,
        "generated_token_count": len(generated_ids),
        "generated_text": generated_text,
        "full_text": full_text,
        "stop_reason": stop_reason,
        "exported_ops": export_diagnostics.get("exported_ops", ()),
        "unsupported_ops": export_diagnostics.get("unsupported_ops", ()),
        **versions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use ONNX-community Qwen3.5 fp16 checkpoint weights, transpile to MLX, "
            "and run autoregressive inference."
        )
    )
    parser.add_argument(
        "--model-id",
        default=_DEFAULT_MODEL_ID,
        help=(
            "Retained for the synthetic transformer-export lane. "
            f"The default real path uses {_DEFAULT_ONNX_MODEL_ID} for weights."
        ),
    )
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT, help="Prompt text for generation.")
    parser.add_argument("--context-window", type=int, default=_DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--max-new-tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=_DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=_DEFAULT_REPETITION_PENALTY,
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer and ONNX-community checkpoint assets from local cache only.",
    )
    add_output_dir_argument(parser, default="examples/qwen/out/qwen3_5_0_8b_mlx")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import mlx.core  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance only
        raise SystemExit("This demo requires optional deps: torch, mlx, and transformers.") from exc

    result = run_demo(
        args.output_dir,
        prompt=args.prompt,
        model_id=args.model_id,
        context_window=args.context_window,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        local_files_only=args.local_files_only,
    )
    print("Qwen3.5 conversion + generation (MLX) succeeded.")
    print(
        "Versions: "
        f"torch={result['torch_version']}, "
        f"transformers={result['transformers_version']}, "
        f"onnx={result['onnx_version']}, "
        f"mlx={result['mlx_version']}"
    )
    print(f"Model id: {result['model_id']}")
    print(f"ONNX: {result['onnx_path']}")
    print(f"Graph IR: {result['graph_path']}")
    print(f"Generated: {result['generated_module']}")
    print(f"Weights: {result['weights_path']}")
    print(f"Max abs diff: {result['max_abs']:.6e}")
    print(f"Generated tokens: {result['generated_token_count']}")
    print(f"Stop reason: {result['stop_reason']}")
    print(f"Generated text: {result['generated_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
