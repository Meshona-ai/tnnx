from __future__ import annotations

import argparse

from examples.common import add_output_dir_argument

from .smoke import format_smoke_job_lines, run_jobs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export upstream-backed model-zoo smoke references to ONNX and transpile them."
        )
    )
    parser.add_argument("--target", choices=["jax", "mlx"], default="jax")
    parser.add_argument("--model", default=None, help="Run a single top-level model-zoo entry.")
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Also attempt experimental jobs such as YOLO and the Llama smoke wrapper.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the planned smoke jobs without exporting or transpiling.",
    )
    add_output_dir_argument(parser, default="examples/out/model_zoo")
    return parser


def main() -> int:
    args = _parser().parse_args()

    if args.list:
        print("=== Model Zoo Smoke Transpile Jobs ===")
        for line in format_smoke_job_lines(target=args.target, out_root=args.output_dir):
            print(line)
        return 0

    print("=== Model Zoo Smoke Transpile Run ===")
    for result in run_jobs(
        target=args.target,
        model_name=args.model,
        include_experimental=args.include_experimental,
        out_root=args.output_dir,
    ):
        print(f"{result.job.model_name} :: {result.job.variant_name} -> {result.status}")
        print(f"  {result.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
