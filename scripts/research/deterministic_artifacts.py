from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import hashlib
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_npz_contents(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(data.files):
            value = np.ascontiguousarray(data[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(value.tobytes())
    return digest.hexdigest()


def _artifact_hashes(out_dir: Path, target: str) -> dict[str, dict[str, str | int]]:
    source_name = {
        "jax": "model_jax.py",
        "mlx": "model_mlx.py",
    }[target]
    names = ["compile_metadata.json", "graph_ir.json", "weights.npz", source_name]
    hashes: dict[str, dict[str, str | int]] = {}
    for name in names:
        path = out_dir / name
        if not path.exists():
            continue
        entry: dict[str, str | int] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if name == "weights.npz":
            entry["content_sha256"] = _sha256_npz_contents(path)
        hashes[name] = entry
    return hashes


def _stable(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return reference == candidate


def run(args: argparse.Namespace) -> int:
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for case_name in args.cases:
        onnx_path = out_root / "onnx" / f"{case_name}.onnx"
        case = write_case_onnx(case_name, onnx_path, seed=args.seed)
        for target in args.targets:
            reference_hashes: dict[str, Any] | None = None
            for repeat_idx in range(args.repeats):
                run_dir = out_root / case_name / target / f"run_{repeat_idx:02d}"
                cfg = CompileConfig(
                    enabled_passes=args.passes,
                    resource_budget=ResourceBudget(
                        target_hardware=args.target_hardware or f"{target}-artifact",
                        preferred_dtype=args.preferred_dtype,
                        memory_budget_mb=args.memory_budget_mb,
                        latency_priority=args.latency_priority,
                        notes="deterministic artifact validation",
                    ),
                )
                manifest = transpile_onnx(str(onnx_path), target, str(run_dir), config=cfg)
                hashes = _artifact_hashes(run_dir, target)
                stable_against_first = (
                    True
                    if reference_hashes is None
                    else _stable(
                        reference_hashes,
                        hashes,
                    )
                )
                if reference_hashes is None:
                    reference_hashes = hashes
                elif not stable_against_first:
                    failures.append(f"{case_name}/{target}/run_{repeat_idx:02d}")
                rows.append(
                    {
                        "case": case_name,
                        "graph_name": case.graph_name,
                        "backend": target,
                        "repeat": repeat_idx,
                        "seed": args.seed,
                        "passes": list(args.passes),
                        "manifest_files": [path.name for path in manifest.files],
                        "artifact_hashes": hashes,
                        "stable_against_first": stable_against_first,
                    }
                )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "note": "Timestamp is report metadata only and is not included in artifact hashes.",
        "repeats": args.repeats,
        "rows": rows,
        "failures": failures,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    stable_rows = sum(1 for row in rows if row["stable_against_first"])
    print(
        f"wrote {args.result_json} with {stable_rows}/{len(rows)} stable repeat rows "
        f"across cases={','.join(args.cases)} targets={','.join(args.targets)}"
    )
    if failures and not args.allow_unstable:
        print("unstable repeated artifacts: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate byte-stable generated artifacts across repeated transpilation runs.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(available_cases()),
        choices=available_cases(),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["jax", "mlx"],
        choices=["jax", "mlx"],
    )
    parser.add_argument("--repeats", type=int, default=3)
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
        default=Path("generated/research/deterministic_artifacts"),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("generated/research/deterministic_artifacts.json"),
    )
    parser.add_argument("--allow-unstable", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
