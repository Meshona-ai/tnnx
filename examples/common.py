from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig

_DEFAULT_OUTPUT_DIR = Path("examples/out")
_TARGET_ENTRYPOINTS = {
    "jax": "model_jax.py",
    "mlx": "model_mlx.py",
}


@dataclass(frozen=True, slots=True)
class ExampleArtifacts:
    output_dir: Path
    onnx_path: Path
    target: str | None = None
    generated_dir: Path | None = None
    graph_path: Path | None = None
    generated_entrypoint: Path | None = None
    weights_path: Path | None = None


def add_output_dir_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str | Path = _DEFAULT_OUTPUT_DIR,
) -> None:
    parser.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        default=str(default),
        help="Directory for exported ONNX and transpiled artifacts.",
    )


def add_target_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = "jax",
) -> None:
    parser.add_argument(
        "--target",
        choices=("jax", "mlx"),
        default=default,
        help="Transpile target for the exported ONNX graph.",
    )


def load_generated_module(path: str | Path, *, module_name: str) -> Any:
    source = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_and_transpile(
    *,
    output_dir: str | Path,
    onnx_name: str,
    export_fn: Callable[[Path], Path],
    after_export: Callable[[Path], None] | None = None,
    target: str,
    generated_dir_name: str | None = None,
    config: CompileConfig | None = None,
) -> ExampleArtifacts:
    root = Path(output_dir)
    onnx_path = export_fn(root / onnx_name)
    if after_export is not None:
        after_export(onnx_path)
    generated_dir = root / (generated_dir_name or f"generated_{Path(onnx_name).stem}_{target}")
    manifest = transpile_onnx(
        str(onnx_path),
        target,
        str(generated_dir),
        config=config,
    )
    entrypoint_name = _TARGET_ENTRYPOINTS[target]
    graph_path = generated_dir / "graph_ir.json"
    return ExampleArtifacts(
        output_dir=root,
        onnx_path=onnx_path,
        target=target,
        generated_dir=generated_dir,
        graph_path=graph_path if graph_path.exists() else None,
        generated_entrypoint=generated_dir / entrypoint_name,
        weights_path=manifest.weights_file,
    )


def print_artifact_summary(title: str, artifacts: ExampleArtifacts) -> None:
    print(f"=== {title} ===")
    print(f"Output dir: {artifacts.output_dir}")
    print(f"ONNX: {artifacts.onnx_path}")
    if artifacts.target is None:
        return
    print(f"Target: {artifacts.target}")
    if artifacts.graph_path is not None:
        print(f"Graph IR: {artifacts.graph_path}")
    if artifacts.generated_entrypoint is not None:
        print(f"Generated entrypoint: {artifacts.generated_entrypoint}")
    if artifacts.weights_path is not None:
        print(f"Weights: {artifacts.weights_path}")
