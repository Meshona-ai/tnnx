from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_ROOT = ROOT / "docs" / "llm_context"
CODE_INDEX = CONTEXT_ROOT / "code_index.json"
SYMBOL_INDEX = CONTEXT_ROOT / "symbol_index.jsonl"
GRAPH_EDGES = CONTEXT_ROOT / "graph_edges.tsv"
EXCLUDED_PREFIXES = (".codex/plans/",)
GENERATED_INDEXES = {
    "docs/llm_context/code_index.json",
    "docs/llm_context/graph_edges.tsv",
    "docs/llm_context/symbol_index.jsonl",
}


def _git_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return sorted(
        line
        for line in result.stdout.splitlines()
        if line and not line.startswith(EXCLUDED_PREFIXES) and (ROOT / line).exists()
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _sha(path: Path, rel: str) -> str | None:
    if rel in GENERATED_INDEXES:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_count(text: str | None) -> int | None:
    if text is None:
        return None
    return len(text.splitlines())


def _subsystem(rel: str) -> str:
    if rel.startswith("src/tnnx/ingest/"):
        return "ingest"
    if rel.startswith("src/tnnx/ir/"):
        return "ir"
    if rel.startswith("src/tnnx/passes/"):
        return "graph-passes"
    if rel.startswith("src/tnnx/codegen/"):
        return "codegen"
    if rel.startswith("src/tnnx/"):
        return "public-runtime"
    if rel.startswith("examples/"):
        return "examples"
    if rel.startswith("tests/"):
        return "tests"
    if rel.startswith("docs/llm_context/"):
        return "llm-context"
    if rel.startswith("docs/") or rel == "README.md":
        return "docs"
    if rel.startswith("scripts/"):
        return "validation-scripts"
    if rel.startswith(".github/"):
        return "ci"
    return "repo-config"


def _classification(rel: str) -> str:
    if rel.startswith("src/tnnx/") or rel == "README.md":
        return "public"
    if rel.startswith("tests/"):
        return "test"
    return "internal"


def _disposition(rel: str) -> str:
    if rel.startswith("docs/llm_context/") or rel == "llms.txt":
        return "LLM-CONTEXT"
    if rel.startswith("tests/") or rel.startswith("src/tnnx/") or rel.startswith("examples/"):
        return "KEEP"
    return "KEEP"


def _purpose(rel: str) -> str:
    subsystem = _subsystem(rel)
    return {
        "ci": "Continuous integration workflow",
        "codegen": "Backend code generation",
        "docs": "Repository documentation",
        "examples": "Example/model path",
        "graph-passes": "Graph pass implementation",
        "ingest": "ONNX ingest",
        "ir": "GraphIR schema and serialization",
        "llm-context": "LLM context pack",
        "public-runtime": "Public runtime/API",
        "tests": "Regression or validation test",
        "validation-scripts": "Reusable validation gate",
    }.get(subsystem, "Repository configuration/process file")


def _context_pages(rel: str) -> list[str]:
    pages = ["docs/llm_context/index.md"]
    if rel.startswith("src/tnnx/codegen/"):
        pages.append("docs/llm_context/backends_jax_mlx.md")
    if rel.startswith("src/tnnx/ingest/") or rel.startswith("src/tnnx/ir/"):
        pages.append("docs/llm_context/operators.md")
    if rel.startswith("examples/"):
        pages.append("docs/llm_context/model_zoo.md")
    if rel.startswith("tests/") or rel.startswith("scripts/") or rel.startswith(".github/"):
        pages.append("docs/llm_context/validation.md")
    if rel in {"src/tnnx/api.py", "src/tnnx/cli.py", "src/tnnx/config.py"}:
        pages.append("docs/llm_context/public_api_cli.md")
    return sorted(set(pages))


def _ast_data(rel: str, text: str | None) -> tuple[list[str], list[str], list[dict[str, str]]]:
    if text is None or not rel.endswith(".py"):
        return [], [], []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], [], []
    imports: list[str] = []
    exports: list[str] = []
    symbols: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            imports.append(module)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            exports.append(node.name)
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            if node.name.startswith("test_"):
                kind = "test"
            symbols.append(
                {
                    "file": rel,
                    "symbol": node.name,
                    "kind": kind,
                    "line": str(node.lineno),
                    "signature": node.name,
                    "subsystem": _subsystem(rel),
                }
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    exports.append(target.id)
                    symbols.append(
                        {
                            "file": rel,
                            "symbol": target.id,
                            "kind": "constant",
                            "line": str(node.lineno),
                            "signature": target.id,
                            "subsystem": _subsystem(rel),
                        }
                    )
    return sorted(set(imports)), sorted(set(exports)), symbols


def _code_index(files: list[str]) -> tuple[dict[str, Any], list[dict[str, str]], list[list[str]]]:
    entries: list[dict[str, Any]] = []
    symbols: list[dict[str, str]] = []
    edges: list[list[str]] = [["source", "relationship", "target", "detail"]]
    for rel in files:
        path = ROOT / rel
        text = _read_text(path)
        imports, exports, file_symbols = _ast_data(rel, text)
        symbols.extend(file_symbols)
        for imported in imports:
            edges.append([rel, "imports", imported, "python AST import"])
        entries.append(
            {
                "path": rel,
                "file_kind": path.suffix.removeprefix(".") or "file",
                "subsystem": _subsystem(rel),
                "classification": _classification(rel),
                "disposition": _disposition(rel),
                "purpose": _purpose(rel),
                "context_pages": _context_pages(rel),
                "imports": imports,
                "exported_symbols": exports,
                "line_count": None if rel in GENERATED_INDEXES else _line_count(text),
                "size_bytes": None if rel in GENERATED_INDEXES else path.stat().st_size,
                "sha256": _sha(path, rel),
            }
        )
    return (
        {
            "schema_version": 1,
            "repo_sha": "working-tree",
            "files": entries,
        },
        sorted(symbols, key=lambda item: (item["file"], int(item["line"]), item["symbol"])),
        edges,
    )


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def _listed_context_pages() -> set[str]:
    text = (CONTEXT_ROOT / "index.md").read_text(encoding="utf-8")
    listed = set()
    for rel in CONTEXT_ROOT.glob("*.md"):
        if f"`{rel.name}`" in text:
            listed.add(rel.name)
    return listed


def _validate_index_links() -> list[str]:
    actual = {path.name for path in CONTEXT_ROOT.glob("*.md")}
    listed = _listed_context_pages()
    missing = sorted(actual - listed - {"index.md"})
    return [f"docs/llm_context/index.md does not list {name}" for name in missing]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    code_index, symbols, edges = _code_index(_git_files())
    expected_code = json.dumps(code_index, indent=2, sort_keys=True) + "\n"
    expected_symbols = "\n".join(json.dumps(item, sort_keys=True) for item in symbols) + "\n"

    if args.write:
        CODE_INDEX.write_text(expected_code, encoding="utf-8")
        SYMBOL_INDEX.write_text(expected_symbols, encoding="utf-8")
        _write_tsv(GRAPH_EDGES, edges)
        print("Wrote LLM context machine indexes.")
        return 0

    errors = _validate_index_links()
    if CODE_INDEX.read_text(encoding="utf-8") != expected_code:
        errors.append("docs/llm_context/code_index.json is stale")
    if SYMBOL_INDEX.read_text(encoding="utf-8") != expected_symbols:
        errors.append("docs/llm_context/symbol_index.jsonl is stale")
    actual_edges = GRAPH_EDGES.read_text(encoding="utf-8")
    expected_edges = "\n".join("\t".join(row) for row in edges) + "\n"
    if actual_edges != expected_edges:
        errors.append("docs/llm_context/graph_edges.tsv is stale")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print("Run: uv run python scripts/check_context_pack.py --write", file=sys.stderr)
        return 1
    print("Context pack indexes are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
