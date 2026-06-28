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
from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig

_DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
_DEFAULT_ONNX_MODEL_ID = "onnx-community/Qwen3.5-0.8B-ONNX"
_DEFAULT_PROMPT = "Write one short sentence about compilers."
_DEFAULT_CONTEXT_WINDOW = 128
_DEFAULT_MAX_NEW_TOKENS = 12
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_TOP_K = 0
_DEFAULT_REPETITION_PENALTY = 1.0
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


def _has_meaningful_text(text: str) -> bool:
    return bool(text.strip()) and any(ch.isalnum() for ch in text)


def _runtime_versions() -> dict[str, str]:
    import jax
    import torch
    import transformers

    return {
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "onnx_version": str(onnx.__version__),
        "jax_version": str(jax.__version__),
    }


def _preferred_torch_device() -> Any:
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_tiny_qwen3_5_config() -> Any:
    from transformers import Qwen3_5TextConfig

    hidden_size = 64
    num_attention_heads = 4
    head_dim = hidden_size // num_attention_heads
    num_hidden_layers = 2
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=256,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        head_dim=head_dim,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=num_attention_heads,
        linear_num_value_heads=num_attention_heads,
        layer_types=["full_attention"] * num_hidden_layers,
        use_cache=False,
        attention_dropout=0.0,
    )


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


def _prepare_model(model: Any) -> Any:
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
    if hasattr(model, "eval"):
        return model.eval()
    return model


def _load_pretrained_model_and_tokenizer(
    model_id: str,
    *,
    local_files_only: bool,
) -> tuple[Any, Any]:
    try:
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance only
        raise SystemExit(
            "This example requires the optional dev dependencies.\nRun: uv sync --dev"
        ) from exc

    try:
        config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
        if hasattr(config, "text_config") and not hasattr(config, "vocab_size"):
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                local_files_only=local_files_only,
            )
        else:
            model = Qwen3_5ForCausalLM.from_pretrained(
                model_id,
                local_files_only=local_files_only,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    except OSError as exc:  # pragma: no cover - depends on local cache/network state
        raise SystemExit(
            f"Could not load {model_id!r} from transformers. "
            "Make sure the checkpoint is cached locally or available from Hugging Face."
        ) from exc

    tokenizer = _prepare_tokenizer(tokenizer)
    model = _prepare_model(model)
    if hasattr(model, "config"):
        model.config.pad_token_id = int(tokenizer.pad_token_id)
    return model, tokenizer


def _build_export_model(model: Any) -> Any:
    import torch
    import torch.nn as nn

    class QwenExportWrapper(nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            cache_position = torch.arange(
                input_ids.shape[-1],
                dtype=torch.long,
                device=input_ids.device,
            )
            outputs = self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                cache_position=cache_position,
            )
            return outputs.logits

    return QwenExportWrapper(model).eval()


def _build_inputs(
    ids: list[int],
    *,
    seq_len: int,
    pad_id: int,
) -> tuple[dict[str, np.ndarray[Any, Any]], int]:
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


def _export_onnx(
    path: str | Path, *, model: Any, sample_inputs: dict[str, np.ndarray[Any, Any]]
) -> Path:
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


def _inspect_onnx(onnx_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from tnnx.ingest.op_map import ONNX_TO_SEMANTIC

    exported = onnx.load(onnx_path, load_external_data=False)
    exported_ops = tuple(sorted({node.op_type for node in exported.graph.node}))
    unsupported = tuple(sorted(set(exported_ops) - (set(ONNX_TO_SEMANTIC.keys()) | {"Constant"})))
    return exported_ops, unsupported


def _sample_next_token(
    logits: np.ndarray[Any, Any],
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


def _resolve_prebuilt_snapshot(*, local_files_only: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance only
        raise SystemExit(
            "This example requires huggingface_hub for the prebuilt ONNX checkpoint.\n"
            "Run: uv sync --dev"
        ) from exc

    try:
        return Path(
            snapshot_download(
                repo_id=_DEFAULT_ONNX_MODEL_ID,
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "chat_template.jinja",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "onnx/decoder_model_merged_fp16.onnx",
                    "onnx/decoder_model_merged_fp16.onnx_data*",
                    "onnx/embed_tokens_fp16.onnx",
                    "onnx/embed_tokens_fp16.onnx_data*",
                ],
                local_files_only=local_files_only,
            )
        )
    except OSError as exc:  # pragma: no cover - depends on local cache/network state
        raise SystemExit(
            f"Could not load {_DEFAULT_ONNX_MODEL_ID!r}. "
            "Cache the ONNX-community snapshot locally or allow network access."
        ) from exc


def _load_prebuilt_tokenizer(snapshot_root: Path) -> Any:
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance only
        raise SystemExit(
            "This example requires transformers for tokenizer loading.\nRun: uv sync --dev"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot_root), local_files_only=True)
        return _prepare_tokenizer(tokenizer)
    except Exception:
        tokenizer_file = snapshot_root / "tokenizer.json"
        if tokenizer_file.exists():
            try:
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
                return _prepare_tokenizer(tokenizer)
            except Exception:
                pass
        tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_MODEL_ID, local_files_only=True)
        return _prepare_tokenizer(tokenizer)


def _symlink_asset(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def _specialize_decoder_to_decode_only(source_path: Path, out_path: Path) -> Path:
    model = onnx.load(source_path, load_external_data=False)
    rewritten: list[Any] = []
    extra_value_info: list[Any] = []

    for node in model.graph.node:
        if node.op_type != "If":
            clone = onnx.NodeProto()
            clone.CopyFrom(node)
            rewritten.append(clone)
            continue

        then_branch = None
        for attr in node.attribute:
            if attr.name == "then_branch":
                then_branch = onnx.helper.get_attribute_value(attr)
                break
        if then_branch is None:
            raise ValueError(f"If node {node.name or node.output[0]!r} is missing then_branch.")
        for branch_node in then_branch.node:
            clone = onnx.NodeProto()
            clone.CopyFrom(branch_node)
            rewritten.append(clone)
        for value_info in then_branch.value_info:
            clone = onnx.ValueInfoProto()
            clone.CopyFrom(value_info)
            extra_value_info.append(clone)

    model.graph.ClearField("node")
    model.graph.node.extend(rewritten)

    existing_value_info = {value_info.name for value_info in model.graph.value_info}
    for value_info in extra_value_info:
        if value_info.name in existing_value_info:
            continue
        model.graph.value_info.append(value_info)
        existing_value_info.add(value_info.name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path)
    return out_path


def _materialize_prebuilt_onnx_assets(
    out_dir: str | Path, snapshot_root: Path
) -> tuple[Path, Path]:
    assets_dir = Path(out_dir) / "prebuilt_onnx"
    assets_dir.mkdir(parents=True, exist_ok=True)

    embed_source = snapshot_root / "onnx" / "embed_tokens_fp16.onnx"
    embed_local = assets_dir / embed_source.name
    _symlink_asset(embed_source, embed_local)
    for data_path in sorted((snapshot_root / "onnx").glob("embed_tokens_fp16.onnx_data*")):
        _symlink_asset(data_path, assets_dir / data_path.name)

    decoder_source = snapshot_root / "onnx" / "decoder_model_merged_fp16.onnx"
    for data_path in sorted((snapshot_root / "onnx").glob("decoder_model_merged_fp16.onnx_data*")):
        _symlink_asset(data_path, assets_dir / data_path.name)

    decoder_local = assets_dir / "decoder_model_merged_fp16_decode.onnx"
    _specialize_decoder_to_decode_only(decoder_source, decoder_local)
    return embed_local, decoder_local


def _onnx_elem_dtype(elem_type: int) -> np.dtype[Any]:
    if elem_type == onnx.TensorProto.FLOAT:
        return np.dtype(np.float32)
    if elem_type == onnx.TensorProto.FLOAT16:
        return np.dtype(np.float16)
    if elem_type == onnx.TensorProto.INT32:
        return np.dtype(np.int32)
    if elem_type == onnx.TensorProto.INT64:
        return np.dtype(np.int64)
    raise ValueError(f"Unsupported ONNX input dtype enum: {elem_type}")


def _onnx_input_specs(onnx_path: Path) -> dict[str, tuple[np.dtype[Any], list[int | str]]]:
    model = onnx.load(onnx_path, load_external_data=False)
    specs: dict[str, tuple[np.dtype[Any], list[int | str]]] = {}
    for value in model.graph.input:
        tensor_type = value.type.tensor_type
        dims: list[int | str] = []
        for dim in tensor_type.shape.dim:
            if dim.dim_param:
                dims.append(dim.dim_param)
            elif dim.dim_value:
                dims.append(int(dim.dim_value))
            else:
                dims.append("?")
        specs[value.name] = (_onnx_elem_dtype(int(tensor_type.elem_type)), dims)
    return specs


def _decoder_state_from_specs(
    specs: dict[str, tuple[np.dtype[Any], list[int | str]]],
) -> dict[str, np.ndarray[Any, Any]]:
    state: dict[str, np.ndarray[Any, Any]] = {}
    for name, (dtype, dims) in sorted(specs.items()):
        if name in {"inputs_embeds", "attention_mask", "position_ids"}:
            continue
        resolved: list[int] = []
        for dim in dims:
            if isinstance(dim, int):
                resolved.append(dim)
            elif dim == "batch_size":
                resolved.append(1)
            elif dim == "past_sequence_length":
                resolved.append(0)
            else:
                resolved.append(1)
        state[name] = np.zeros(tuple(resolved), dtype=dtype)
    return state


def _build_decode_step_inputs(
    *,
    state: dict[str, np.ndarray[Any, Any]],
    inputs_embeds: np.ndarray[Any, Any],
    position: int,
) -> dict[str, np.ndarray[Any, Any]]:
    total_len = int(position) + 1
    inputs = dict(state)
    inputs["inputs_embeds"] = np.asarray(inputs_embeds)
    inputs["attention_mask"] = np.ones((1, total_len), dtype=np.int64)
    pos = np.asarray([position], dtype=np.int64).reshape(1, 1)
    inputs["position_ids"] = np.stack((pos, pos, pos), axis=0)
    return inputs


def _next_decoder_state(outputs: dict[str, Any]) -> dict[str, np.ndarray[Any, Any]]:
    next_state: dict[str, np.ndarray[Any, Any]] = {}
    for name, value in outputs.items():
        if name == "logits":
            continue
        if name.startswith("present_conv."):
            key = "past_conv." + name.removeprefix("present_conv.")
        elif name.startswith("present_recurrent."):
            key = "past_recurrent." + name.removeprefix("present_recurrent.")
        elif name.startswith("present.") and (name.endswith(".key") or name.endswith(".value")):
            key = "past_key_values" + name.removeprefix("present")
        else:
            continue
        next_state[key] = np.asarray(value)
    return next_state


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

    embed_generated_dir = Path(out_dir) / "generated_qwen_embed_tokens_jax"
    decoder_generated_dir = Path(out_dir) / "generated_qwen_decoder_merged_jax"
    embed_manifest = transpile_onnx(
        str(embed_onnx_path),
        "jax",
        str(embed_generated_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )
    decoder_manifest = transpile_onnx(
        str(decoder_onnx_path),
        "jax",
        str(decoder_generated_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )

    embed_module = load_generated_module(
        embed_generated_dir / "model_jax.py",
        module_name="generated_qwen_embed_tokens_jax",
    )
    embed_params = embed_module.load_weights(str(embed_manifest.weights_file))
    decoder_module = load_generated_module(
        decoder_generated_dir / "model_jax.py",
        module_name="generated_qwen_decoder_merged_jax",
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
        inputs_embeds = np.asarray(
            embed_module.forward(embed_params, token_inputs)["inputs_embeds"]
        )
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
        raise AssertionError("Qwen3.5 JAX generation returned zero decoded tokens.")
    if not _has_meaningful_text(generated_text):
        raise AssertionError("Qwen3.5 JAX generation did not produce meaningful decoded text.")

    return {
        "model_id": _DEFAULT_ONNX_MODEL_ID,
        "prompt": prompt,
        "onnx_path": str(decoder_onnx_path),
        "embed_onnx_path": str(embed_onnx_path),
        "graph_path": str(decoder_generated_dir / "graph_ir.json"),
        "generated_module": str(decoder_generated_dir / "model_jax.py"),
        "weights_path": str(decoder_manifest.weights_file),
        "embed_generated_module": str(embed_generated_dir / "model_jax.py"),
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
    out_dir: str | Path = "examples/qwen/out/qwen3_5_0_8b",
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

    export_model = _build_export_model(model).to(_preferred_torch_device())
    input_ids_t = torch.from_numpy(sample_inputs["input_ids"]).to(_preferred_torch_device())
    attention_mask_t = torch.from_numpy(sample_inputs["attention_mask"]).to(
        _preferred_torch_device()
    )
    with torch.no_grad():
        expected = (
            export_model(
                input_ids_t,
                attention_mask_t,
            )
            .detach()
            .cpu()
            .numpy()
        )
    actual = np.asarray(module.forward(params, sample_inputs)["logits"])
    max_abs = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"Qwen3.5 JAX parity failed: max_abs={max_abs}, rtol={rtol}, atol={atol}"
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
        raise AssertionError("Qwen3.5 JAX generation returned zero decoded tokens.")
    if not _has_meaningful_text(generated_text):
        raise AssertionError("Qwen3.5 JAX generation did not produce meaningful decoded text.")

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
            "Load a pretrained Qwen3.5 checkpoint from transformers, transpile it to JAX, "
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
        help="Load the tokenizer and checkpoint from the local Hugging Face cache only.",
    )
    add_output_dir_argument(parser, default="examples/qwen/out/qwen3_5_0_8b")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import jax  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guidance only
        raise SystemExit("This demo requires optional deps: torch, jax, and transformers.") from exc

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
    print("Qwen3.5 conversion + generation succeeded.")
    print(
        "Versions: "
        f"torch={result['torch_version']}, "
        f"transformers={result['transformers_version']}, "
        f"onnx={result['onnx_version']}, "
        f"jax={result['jax_version']}"
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
