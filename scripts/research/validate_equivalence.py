from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptfm_cases import available_cases, write_case_onnx
from tnnx.api import transpile_onnx
from tnnx.config import DEFAULT_PASSES, CompileConfig, CompilePass, ResourceBudget


def _parse_passes(value: str) -> tuple[CompilePass, ...]:
    normalized = value.strip().lower()
    if normalized in {"", "none"}:
        return ()
    requested = tuple(item.strip() for item in normalized.split(",") if item.strip())
    unknown = [item for item in requested if item not in DEFAULT_PASSES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unsupported pass(es): {', '.join(unknown)}. "
            f"Supported: {', '.join(DEFAULT_PASSES)} or none."
        )
    return requested  # type: ignore[return-value]


def _optional_import(name: str) -> Any | None:
    try:
        return __import__(name, fromlist=["*"])
    except ModuleNotFoundError:
        return None


def _target_runtime_available(target: str) -> bool:
    if target == "jax":
        return _optional_import("jax") is not None
    if target == "mlx":
        return _optional_import("mlx.core") is not None
    return False


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated module at {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_numpy(value: Any, target: str) -> np.ndarray:
    if hasattr(value, "block_until_ready"):
        value = value.block_until_ready()
    if target == "mlx":
        mx = _optional_import("mlx.core")
        if mx is not None:
            mx.eval(value)
            mx.synchronize()
    return np.asarray(value)


def _source_path(out_dir: Path, target: str) -> Path:
    if target == "jax":
        return out_dir / "model_jax.py"
    if target == "mlx":
        return out_dir / "model_mlx.py"
    raise ValueError(f"Equivalence runtime is not implemented for target {target!r}.")


def _compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual32 = np.asarray(actual, dtype=np.float32)
    expected32 = np.asarray(expected, dtype=np.float32)
    diff = np.abs(actual32 - expected32)
    denom = np.maximum(np.abs(expected32), np.float32(1e-6))
    return {
        "max_abs_error": float(np.max(diff)),
        "max_relative_error": float(np.max(diff / denom)),
        "actual_shape": list(actual32.shape),
        "expected_shape": list(expected32.shape),
        "dtype": str(actual32.dtype),
    }


def _within_tolerance(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> bool:
    actual32 = np.asarray(actual, dtype=np.float32)
    expected32 = np.asarray(expected, dtype=np.float32)
    allowed = atol + rtol * np.abs(expected32)
    return bool(np.all(np.abs(actual32 - expected32) <= allowed))


def run(args: argparse.Namespace) -> int:
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for case_name in args.cases:
        onnx_path = out_root / "onnx" / f"{case_name}.onnx"
        case = write_case_onnx(case_name, onnx_path, seed=args.seed)
        for target in args.targets:
            if not _target_runtime_available(target):
                rows.append(
                    {
                        "case": case_name,
                        "graph_name": case.graph_name,
                        "backend": target,
                        "seed": args.seed,
                        "status": "skipped",
                        "reason": f"{target} runtime is not importable",
                    }
                )
                if args.require_targets:
                    failures.append(f"{case_name}/{target}/runtime_missing")
                continue
            out_dir = out_root / case_name / target
            cfg = CompileConfig(
                enabled_passes=args.passes,
                resource_budget=ResourceBudget(
                    target_hardware=args.target_hardware or f"{target}-runtime",
                    preferred_dtype=args.preferred_dtype,
                    memory_budget_mb=args.memory_budget_mb,
                    latency_priority=args.latency_priority,
                    notes="numerical equivalence validation",
                ),
            )
            try:
                _ = transpile_onnx(str(onnx_path), target, str(out_dir), config=cfg)
                module = _load_module(
                    _source_path(out_dir, target),
                    f"tnnx_equiv_{case_name}_{target}",
                )
                params = module.load_weights(str(out_dir / "weights.npz"))
                output = module.forward(params, case.inputs)[case.output_name]
                actual = _to_numpy(output, target)
                expected = case.expected[case.output_name]
                stats = _compare(actual, expected)
                passed = stats["actual_shape"] == stats["expected_shape"] and _within_tolerance(
                    actual,
                    expected,
                    atol=case.atol,
                    rtol=case.rtol,
                )
                status = "passed" if passed else "failed"
                if not passed:
                    failures.append(f"{case_name}/{target}")
                rows.append(
                    {
                        "case": case_name,
                        "graph_name": case.graph_name,
                        "description": case.description,
                        "backend": target,
                        "seed": args.seed,
                        "passes": list(args.passes),
                        "status": status,
                        "output_name": case.output_name,
                        "tolerance": {"atol": case.atol, "rtol": case.rtol},
                        **stats,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{case_name}/{target}/exception")
                rows.append(
                    {
                        "case": case_name,
                        "graph_name": case.graph_name,
                        "backend": target,
                        "seed": args.seed,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": rows,
        "failures": failures,
        "note": "Skipped rows are dependency availability statements, not equivalence evidence.",
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    passed = sum(1 for row in rows if row.get("status") == "passed")
    skipped = sum(1 for row in rows if row.get("status") == "skipped")
    print(
        f"wrote {args.result_json}: passed={passed} skipped={skipped} "
        f"failed_or_error={len(failures)}"
    )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate generated JAX/MLX outputs against NumPy reference outputs.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(available_cases()),
        choices=available_cases(),
    )
    parser.add_argument("--targets", nargs="+", default=["jax", "mlx"], choices=["jax", "mlx"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--passes", type=_parse_passes, default=DEFAULT_PASSES)
    parser.add_argument("--target-hardware", default="")
    parser.add_argument("--preferred-dtype", default="float32")
    parser.add_argument("--memory-budget-mb", type=int, default=None)
    parser.add_argument(
        "--latency-priority",
        default="auditability",
        choices=["balanced", "low_latency", "low_memory", "auditability"],
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("generated/research/equivalence"),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("generated/research/equivalence.json"),
    )
    parser.add_argument("--require-targets", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
