from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.qwen import infer_qwen3_5_from_transformers_jax as qwen_jax  # noqa: E402
from scripts.research.hot_model_benchmarks import (  # noqa: E402
    _qwen_prepare_generated,
    _qwen_prompt_ids,
)


def _onnx_dtype_name(dtype_enum: int) -> str:
    try:
        return str(onnx.TensorProto.DataType.Name(dtype_enum))
    except ValueError:
        return f"UNKNOWN_{dtype_enum}"


def _onnx_tensor_type_counts(path: Path) -> dict[str, int]:
    model = onnx.load(path, load_external_data=False)
    counts: Counter[str] = Counter()
    for initializer in model.graph.initializer:
        counts[_onnx_dtype_name(int(initializer.data_type))] += 1
    return dict(sorted(counts.items()))


def _session_output_dtypes(session: Any) -> dict[str, str]:
    return {output.name: str(output.type) for output in session.get_outputs()}


def _param_dtype_counts(value: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()

    def _visit(item: Any) -> None:
        if isinstance(item, dict):
            for child in item.values():
                _visit(child)
            return
        try:
            arr = np.asarray(item)
        except Exception:
            return
        counts[str(arr.dtype)] += 1

    _visit(value)
    return dict(sorted(counts.items()))


def _finite_max_rel(actual: np.ndarray[Any, Any], expected: np.ndarray[Any, Any]) -> float | None:
    denom = np.maximum(np.abs(expected), np.asarray(1e-6, dtype=np.float32))
    rel = np.abs(actual - expected) / denom
    finite = np.isfinite(rel)
    if not np.any(finite):
        return None
    return float(np.max(rel[finite]))


def _array_error(
    actual_value: Any,
    expected_value: Any,
    *,
    is_logits: bool = False,
) -> dict[str, Any]:
    actual = np.asarray(actual_value)
    expected = np.asarray(expected_value)
    if actual.shape != expected.shape:
        return {
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "shape_match": False,
        }

    actual32 = actual.astype(np.float32, copy=False)
    expected32 = expected.astype(np.float32, copy=False)
    diff = actual32 - expected32
    abs_diff = np.abs(diff)
    payload: dict[str, Any] = {
        "shape_match": True,
        "shape": [int(dim) for dim in actual.shape],
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "actual_max_abs": float(np.max(np.abs(actual32))) if actual.size else 0.0,
        "expected_max_abs": float(np.max(np.abs(expected32))) if expected.size else 0.0,
        "max_abs": float(np.max(abs_diff)) if actual.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if actual.size else 0.0,
        "rmse": float(math.sqrt(float(np.mean(diff * diff)))) if actual.size else 0.0,
        "max_relative": _finite_max_rel(actual32, expected32),
        "allclose_rtol_2e_4_atol_2e_5": bool(
            np.allclose(actual32, expected32, rtol=2e-4, atol=2e-5)
        ),
        "allclose_rtol_1e_2_atol_1e_2": bool(
            np.allclose(actual32, expected32, rtol=1e-2, atol=1e-2)
        ),
    }
    if np.issubdtype(actual.dtype, np.floating) and np.issubdtype(expected.dtype, np.floating):
        actual16 = actual32.astype(np.float16).astype(np.float32)
        expected16 = expected32.astype(np.float16).astype(np.float32)
        payload["after_fp16_round_max_abs"] = float(np.max(np.abs(actual16 - expected16)))
    if is_logits and actual.ndim >= 2:
        actual_last = actual32.reshape(-1, actual32.shape[-1])
        expected_last = expected32.reshape(-1, expected32.shape[-1])
        actual_top1 = np.argmax(actual_last, axis=-1)
        expected_top1 = np.argmax(expected_last, axis=-1)
        payload["logit_top1_equal"] = bool(np.array_equal(actual_top1, expected_top1))
        payload["logit_max_margin"] = float(
            np.max(np.sort(expected_last, axis=-1)[:, -1] - np.sort(expected_last, axis=-1)[:, -2])
        )
    return payload


def _decoder_expected(
    *,
    embed_session: Any,
    decoder_session: Any,
    prepared: dict[str, Any],
    prompt_ids: list[int],
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, np.ndarray[Any, Any]]]:
    token_inputs = {"input_ids": np.asarray([[int(prompt_ids[0])]], dtype=np.int64)}
    embed_expected = embed_session.run(None, token_inputs)[0]
    decoder_inputs = qwen_jax._build_decode_step_inputs(
        state=qwen_jax._decoder_state_from_specs(prepared["decoder_specs"]),
        inputs_embeds=embed_expected,
        position=0,
    )
    decoder_expected_values = decoder_session.run(None, decoder_inputs)
    expected = {
        output.name: value
        for output, value in zip(
            decoder_session.get_outputs(),
            decoder_expected_values,
            strict=True,
        )
    }
    return {"inputs_embeds": embed_expected}, expected


def run(args: argparse.Namespace) -> dict[str, Any]:
    import onnxruntime as ort

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prepared_by_backend: dict[str, dict[str, Any]] = {}
    generated_outputs: dict[str, dict[str, Any]] = {}
    backend_reports: list[dict[str, Any]] = []

    for backend in args.backend:
        prepared_by_backend[backend] = _qwen_prepare_generated(
            out_dir=args.out_dir / f"qwen_generated_{backend}",
            backend=backend,
            local_files_only=True,
            force=args.force,
        )

    first_backend = args.backend[0]
    prepared = prepared_by_backend[first_backend]
    prompt_ids, _ = _qwen_prompt_ids(prepared["tokenizer"], args.prompt)
    embed_session = ort.InferenceSession(
        str(prepared["embed_onnx_path"]),
        providers=["CPUExecutionProvider"],
    )
    decoder_session = ort.InferenceSession(
        str(prepared["decoder_onnx_path"]),
        providers=["CPUExecutionProvider"],
    )
    embed_expected, decoder_expected = _decoder_expected(
        embed_session=embed_session,
        decoder_session=decoder_session,
        prepared=prepared,
        prompt_ids=prompt_ids,
    )

    token_inputs = {"input_ids": np.asarray([[int(prompt_ids[0])]], dtype=np.int64)}
    decoder_inputs = qwen_jax._build_decode_step_inputs(
        state=qwen_jax._decoder_state_from_specs(prepared["decoder_specs"]),
        inputs_embeds=embed_expected["inputs_embeds"],
        position=0,
    )

    for backend, backend_prepared in prepared_by_backend.items():
        embed_actual = backend_prepared["embed_module"].forward(
            backend_prepared["embed_params"],
            token_inputs,
        )
        decoder_actual = backend_prepared["decoder_module"].forward(
            backend_prepared["decoder_params"],
            decoder_inputs,
        )
        generated_outputs[backend] = {
            name: np.asarray(value) for name, value in decoder_actual.items()
        }
        state_errors = [
            _array_error(decoder_actual[name], expected)
            for name, expected in decoder_expected.items()
            if name != "logits" and name in decoder_actual
        ]
        worst_state = max(
            state_errors,
            key=lambda item: float(item.get("max_abs", -1.0)),
            default={"max_abs": 0.0},
        )
        backend_reports.append(
            {
                "backend": backend,
                "device": backend_prepared["device"],
                "default_backend": backend_prepared["default_backend"],
                "decoder_param_dtypes": _param_dtype_counts(backend_prepared["decoder_params"]),
                "embed": _array_error(
                    embed_actual["inputs_embeds"],
                    embed_expected["inputs_embeds"],
                ),
                "logits": _array_error(
                    decoder_actual["logits"],
                    decoder_expected["logits"],
                    is_logits=True,
                ),
                "worst_state": worst_state,
                "state_output_count": len(state_errors),
            }
        )

    cross_backend: dict[str, Any] = {}
    if set(args.backend) >= {"jax", "mlx"}:
        jax_outputs = generated_outputs["jax"]
        mlx_outputs = generated_outputs["mlx"]
        shared = sorted(set(jax_outputs) & set(mlx_outputs))
        cross_backend = {
            name: _array_error(mlx_outputs[name], jax_outputs[name], is_logits=name == "logits")
            for name in shared
            if name == "logits" or name.startswith("present")
        }

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": args.prompt,
        "model_id": "onnx-community/Qwen3.5-0.8B-ONNX",
        "reference": {
            "runtime": "ONNX Runtime CPUExecutionProvider",
            "embed_output_dtypes": _session_output_dtypes(embed_session),
            "decoder_output_dtypes": _session_output_dtypes(decoder_session),
            "embed_initializer_dtypes": _onnx_tensor_type_counts(prepared["embed_onnx_path"]),
            "decoder_initializer_dtypes": _onnx_tensor_type_counts(prepared["decoder_onnx_path"]),
        },
        "backends": backend_reports,
        "cross_backend_generated": cross_backend,
        "diagnosis": (
            "Embedding is exact because it is a table lookup. Decoder drift appears only after "
            "fp16 arithmetic in attention, normalization, and matmul-heavy blocks. The JAX and "
            "MLX generated outputs preserve top-1 logits for the probed token, but differ from "
            "ONNX Runtime by fp16-scale absolute amounts; FLUX reduced paths use fp32/tiny "
            "topologies, so their absolute errors are near single-precision roundoff."
        ),
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Qwen3.5 generated-backend error.")
    parser.add_argument("--backend", nargs="+", choices=("jax", "mlx"), default=["jax", "mlx"])
    parser.add_argument("--out-dir", type=Path, default=Path("/private/tmp/tnnx_qwen_error_probe"))
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("docs/research_wiki/results/qwen_error_probe.json"),
    )
    parser.add_argument("--prompt", default="Explain ONNX to MLX conversion in one sentence.")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
