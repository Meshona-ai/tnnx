from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PATHS = (
    "docs/research_wiki/articles/tnnx_paper.md",
    "docs/research_wiki/articles/tnnx_paper.tex",
    "docs/research_wiki/results/benchmark_source_of_truth.md",
    "docs/research_wiki/results/tnnx_evidence.md",
    "docs/research_wiki/results/tnnx_evidence.json",
    "docs/research_wiki/topics/anonymized-artifact-release.md",
    "docs/research_wiki/topics/resource-budget-story.md",
    "docs/research_wiki/topics/threats-to-validity.md",
)

PATTERNS = (
    ("local_absolute_path", re.compile(r"(/Users/|/home/)[^\s)`>,]+")),
    ("email_address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    (
        "non_anonymous_repo_url",
        re.compile(r"github\.com/[^/\s)`>,]+/(tnnx|transpiler)[/\w.-]*", re.IGNORECASE),
    ),
)


@dataclass(frozen=True, slots=True)
class Finding:
    path: Path
    line: int
    kind: str
    text: str


def _iter_files(paths: Iterable[str]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.is_dir():
            yield from sorted(child for child in path.rglob("*") if child.is_file())
        elif path.exists():
            yield path
        else:
            yield path


def scan(paths: Iterable[str]) -> list[Finding]:
    findings: list[Finding] = []
    for path in _iter_files(paths):
        if not path.exists():
            findings.append(Finding(path, 0, "missing_file", "Path does not exist."))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path.name.endswith(".tex") and r"\author{Anonymous Authors}" not in text:
            findings.append(
                Finding(path, 0, "latex_author", "Expected \\author{Anonymous Authors}.")
            )
        for line_no, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(Finding(path, line_no, kind, line.strip()))
    return findings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan TNNX paper artifacts for double-blind anonymity leaks."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_PATHS),
        help="Files or directories to scan. Defaults to submission-facing research docs.",
    )
    return parser


def main() -> int:
    findings = scan(_parser().parse_args().paths)
    if not findings:
        print("Anonymity check passed.")
        return 0
    print("Anonymity check failed:")
    for finding in findings:
        if finding.path.is_relative_to(REPO_ROOT):
            rel = finding.path.relative_to(REPO_ROOT)
        else:
            rel = finding.path
        location = f"{rel}:{finding.line}" if finding.line else str(rel)
        print(f"- {location} [{finding.kind}] {finding.text}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
