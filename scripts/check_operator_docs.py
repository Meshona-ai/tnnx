from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "operators.md"
EQUALITY_DISPATCH = re.compile(r'if node\.op == "([A-Z0-9_]+)"')
SET_DISPATCH = re.compile(r"if node\.op in \{([^}]*)\}")


def _imports_ready() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _dispatched_ops(module: object) -> set[str]:
    source = inspect.getsource(module._emit_node_expr)  # type: ignore[attr-defined]
    ops = set(EQUALITY_DISPATCH.findall(source))
    for group in SET_DISPATCH.findall(source):
        ops.update(re.findall(r'"([A-Z0-9_]+)"', group))
    return ops


def _generate() -> str:
    _imports_ready()
    from tnnx.codegen import jax_codegen, mlx_codegen
    from tnnx.ingest.op_map import ONNX_TO_SEMANTIC
    from tnnx.ir.schema import SEMANTIC_SCHEMAS

    sources: dict[str, list[str]] = {op: [] for op in SEMANTIC_SCHEMAS}
    for onnx_name, semantic in sorted(ONNX_TO_SEMANTIC.items()):
        sources.setdefault(semantic, []).append(onnx_name)
    jax_ops = _dispatched_ops(jax_codegen)
    mlx_ops = _dispatched_ops(mlx_codegen)

    lines = [
        "# Supported Operators",
        "",
        (
            "Generated from `src/tnnx/ir/schema.py`, `src/tnnx/ingest/op_map.py`, "
            "and backend dispatch."
        ),
        "",
        f"- Semantic schema ops: {len(SEMANTIC_SCHEMAS)}",
        f"- ONNX spellings mapped: {len(ONNX_TO_SEMANTIC)}",
        f"- JAX dispatch coverage: {len(jax_ops & set(SEMANTIC_SCHEMAS))}/{len(SEMANTIC_SCHEMAS)}",
        f"- MLX dispatch coverage: {len(mlx_ops & set(SEMANTIC_SCHEMAS))}/{len(SEMANTIC_SCHEMAS)}",
        "",
        "| Semantic op | ONNX source | JAX | MLX | Disposition |",
        "| --- | --- | --- | --- | --- |",
    ]
    for op in sorted(SEMANTIC_SCHEMAS):
        source = ", ".join(sources.get(op, [])) or "internal/generated IR only"
        disposition = "ready" if source != "internal/generated IR only" else "internal-only"
        lines.append(
            f"| {op} | {source} | {'yes' if op in jax_ops else 'no'} | "
            f"{'yes' if op in mlx_ops else 'no'} | {disposition} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    expected = _generate()
    if args.write:
        DOC_PATH.write_text(expected, encoding="utf-8")
        print(f"Wrote {DOC_PATH.relative_to(ROOT)}")
        return 0
    actual = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
    if actual != expected:
        print(
            "docs/operators.md is stale. Run: uv run python scripts/check_operator_docs.py --write",
            file=sys.stderr,
        )
        return 1
    print("Operator docs are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
