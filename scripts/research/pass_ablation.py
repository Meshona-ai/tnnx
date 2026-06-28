from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import hashlib
import json
import sys
import time
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

PASS_MODES: dict[str, tuple[tuple[CompilePass, ...], bool]] = {
    "none": ((), False),
    "prune": (("prune",), False),
    "normalize": (("normalize",), False),
    "shape_prop": (("normalize", "shape_prop"), True),
    "all": (DEFAULT_PASSES, True),
}


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_npz_contents(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(data.files):
            value = np.ascontiguousarray(data[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(value.tobytes())
    return digest.hexdigest()


def _source_path(out_dir: Path, target: str) -> Path:
    if target == "jax":
        return out_dir / "model_jax.py"
    if target == "mlx":
        return out_dir / "model_mlx.py"
    raise ValueError(f"Unsupported target: {target}")


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _graph_stats(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"node_count": 0, "tensor_count": 0, "tensors_with_shape": 0}
    graph = json.loads(path.read_text(encoding="utf-8"))
    tensors = graph.get("tensors", {})
    if not isinstance(tensors, dict):
        tensors = {}
    return {
        "node_count": len(graph.get("nodes", [])),
        "tensor_count": len(tensors),
        "tensors_with_shape": sum(
            1
            for payload in tensors.values()
            if isinstance(payload, dict) and bool(payload.get("shape"))
        ),
    }


def run(args: argparse.Namespace) -> int:
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    onnx_path = out_root / "onnx" / f"{args.case}.onnx"
    case = write_case_onnx(args.case, onnx_path, seed=args.seed)
    for mode in args.modes:
        passes, infer_shapes = PASS_MODES[mode]
        out_dir = out_root / args.case / args.target / mode
        cfg = CompileConfig(
            enabled_passes=passes,
            infer_shapes=infer_shapes,
            resource_budget=ResourceBudget(
                target_hardware=args.target_hardware or f"{args.target}-ablation",
                preferred_dtype=args.preferred_dtype,
                memory_budget_mb=args.memory_budget_mb,
                latency_priority=args.latency_priority,
                notes=f"pass ablation mode={mode}",
            ),
        )
        start = time.perf_counter()
        try:
            manifest = transpile_onnx(str(onnx_path), args.target, str(out_dir), config=cfg)
            compile_ms = (time.perf_counter() - start) * 1000.0
            graph_path = out_dir / "graph_ir.json"
            source_path = _source_path(out_dir, args.target)
            rows.append(
                {
                    "case": args.case,
                    "graph_name": case.graph_name,
                    "backend": args.target,
                    "mode": mode,
                    "passes": list(passes),
                    "infer_shapes": infer_shapes,
                    "seed": args.seed,
                    "status": "completed",
                    "compile_ms": compile_ms,
                    "generated_source_loc": _line_count(source_path),
                    "manifest_files": [path.name for path in manifest.files],
                    "source_sha256": _sha256_file(source_path),
                    "graph_ir_sha256": _sha256_file(graph_path),
                    "compile_metadata_sha256": _sha256_file(out_dir / "compile_metadata.json"),
                    "weights_content_sha256": _sha256_npz_contents(out_dir / "weights.npz"),
                    **_graph_stats(graph_path),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{mode}/{type(exc).__name__}")
            rows.append(
                {
                    "case": args.case,
                    "graph_name": case.graph_name,
                    "backend": args.target,
                    "mode": mode,
                    "passes": list(passes),
                    "infer_shapes": infer_shapes,
                    "seed": args.seed,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "note": (
            "compile_ms is local script wall time, not a hardware performance result. "
            "Use latency benchmarks for inference-performance claims."
        ),
        "rows": rows,
        "failures": failures,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    print(
        f"wrote {args.result_json}: completed={len(rows) - len(failures)} failures={len(failures)}"
    )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real compile-pass ablations and emit table-ready artifact metrics.",
    )
    parser.add_argument("--case", default="tiny_transformer", choices=available_cases())
    parser.add_argument("--target", default="jax", choices=["jax", "mlx"])
    parser.add_argument("--modes", nargs="+", default=list(PASS_MODES), choices=tuple(PASS_MODES))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-hardware", default="")
    parser.add_argument("--preferred-dtype", default="float32")
    parser.add_argument("--memory-budget-mb", type=int, default=None)
    parser.add_argument(
        "--latency-priority",
        default="auditability",
        choices=["balanced", "low_latency", "low_memory", "auditability"],
    )
    parser.add_argument("--out-root", type=Path, default=Path("generated/research/pass_ablation"))
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("generated/research/pass_ablation.json"),
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
