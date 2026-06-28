from __future__ import annotations

import argparse

from .api import transpile_onnx
from .config import DEFAULT_PASSES, CompileConfig, CompilePass, ResourceBudget


def _parse_passes(value: str) -> tuple[CompilePass, ...]:
    normalized = value.strip().lower()
    if normalized in {"", "none", "noopt", "no-op"}:
        return ()
    requested = tuple(item.strip() for item in normalized.split(",") if item.strip())
    unknown = [item for item in requested if item not in DEFAULT_PASSES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unsupported pass(es): {', '.join(unknown)}. "
            f"Supported passes: {', '.join(DEFAULT_PASSES)} or none."
        )
    return requested  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tnnx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transpile = subparsers.add_parser("transpile")
    transpile.add_argument("--onnx", required=True)
    transpile.add_argument("--target", required=True, choices=["jax", "mlx"])
    transpile.add_argument("--out", required=True)
    transpile.add_argument("--weights", default="weights.npz")
    transpile.add_argument("--entry", default="forward")
    transpile.add_argument(
        "--graph-ir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit graph_ir.json artifact (default: enabled).",
    )
    transpile.add_argument(
        "--passes",
        type=_parse_passes,
        default=DEFAULT_PASSES,
        help=(
            "Comma-separated compile passes to enable. "
            "Supported: prune,normalize,shape_prop; use 'none' for no passes."
        ),
    )
    transpile.add_argument(
        "--target-hardware",
        default="unspecified",
        help="Hardware tag recorded in compile_metadata.json.",
    )
    transpile.add_argument(
        "--preferred-dtype",
        default=None,
        help="Preferred dtype recorded as a resource constraint; not a dtype rewrite pass.",
    )
    transpile.add_argument(
        "--memory-budget-mb",
        type=int,
        default=None,
        help="Memory budget recorded in compile_metadata.json.",
    )
    transpile.add_argument(
        "--latency-priority",
        default="balanced",
        choices=["balanced", "low_latency", "low_memory", "auditability"],
        help="Optimization priority recorded in compile_metadata.json.",
    )
    transpile.add_argument(
        "--resource-note",
        default="",
        help="Free-form resource-budget note recorded in compile_metadata.json.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "transpile":
        transpile_onnx(
            onnx_path=args.onnx,
            target=args.target,
            out_dir=args.out,
            config=CompileConfig(
                entrypoint=args.entry,
                weights_filename=args.weights,
                emit_graph_ir=args.graph_ir,
                enabled_passes=args.passes,
                resource_budget=ResourceBudget(
                    target_hardware=args.target_hardware,
                    preferred_dtype=args.preferred_dtype,
                    memory_budget_mb=args.memory_budget_mb,
                    latency_priority=args.latency_priority,
                    notes=args.resource_note,
                ),
            ),
        )
        return 0
    print(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
