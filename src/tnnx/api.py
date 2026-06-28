from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .codegen.jax_codegen import emit_jax_module
from .codegen.mlx_codegen import emit_mlx_module
from .config import DEFAULT_PASSES, ArtifactManifest, CompileConfig, CompilePass
from .ingest.onnx_reader import load_onnx_to_ir
from .ir.serde import dumps_graph_json
from .ir.types import GraphIR
from .passes.normalize import normalize_graph
from .passes.prune import prune_dead_nodes
from .passes.shape_prop import propagate_shapes
from .runtime.weights import save_weights_npz

_RUNTIME_VALUE_MAX_SIZE = 64


def _runtime_values_from_weights(
    weights: dict[str, np.ndarray[Any, Any]],
) -> dict[str, list[int | float]]:
    runtime_values: dict[str, list[int | float]] = {}
    for name, value in weights.items():
        if value.size > _RUNTIME_VALUE_MAX_SIZE:
            continue
        flattened = value.reshape(-1).tolist()
        if not isinstance(flattened, list) or not flattened:
            continue
        scalars: list[int | float] = []
        for item in flattened:
            if isinstance(item, bool):
                scalars.append(int(item))
            elif isinstance(item, int | float):
                scalars.append(item)
        if scalars:
            runtime_values[name] = scalars
    return runtime_values


def _validate_enabled_passes(enabled_passes: tuple[CompilePass, ...]) -> None:
    unknown = [pass_name for pass_name in enabled_passes if pass_name not in DEFAULT_PASSES]
    if unknown:
        raise ValueError(f"Unsupported compile pass(es): {', '.join(unknown)}")


def _apply_configured_passes(
    ir: GraphIR,
    weights: dict[str, np.ndarray[Any, Any]],
    cfg: CompileConfig,
) -> tuple[GraphIR, list[str]]:
    _validate_enabled_passes(cfg.enabled_passes)
    applied: list[str] = []
    for pass_name in cfg.enabled_passes:
        if pass_name == "prune":
            ir = prune_dead_nodes(ir)
            applied.append(pass_name)
        elif pass_name == "normalize":
            ir = normalize_graph(ir)
            applied.append(pass_name)
        elif pass_name == "shape_prop":
            if cfg.infer_shapes:
                ir = propagate_shapes(ir, runtime_values=_runtime_values_from_weights(weights))
                applied.append(pass_name)
            else:
                applied.append("shape_prop_skipped_infer_shapes_false")
    return ir, applied


def _write_compile_metadata(
    output_dir: Path,
    *,
    target: str,
    ir: GraphIR,
    cfg: CompileConfig,
    applied_passes: list[str],
) -> Path:
    payload = {
        "target": target,
        "graph_name": ir.name,
        "opset": ir.opset,
        "inputs": list(ir.inputs),
        "outputs": list(ir.outputs),
        "node_count": len(ir.nodes),
        "tensor_count": len(ir.tensors),
        "compile_config": cfg.to_metadata(),
        "metadata_only_config_fields": ["deterministic", "emit_shape_asserts", "opset"],
        "applied_passes": applied_passes,
        "resource_adaptation_status": (
            "explicit_budget_metadata_only; no automatic resource-search planner"
        ),
    }
    path = output_dir / "compile_metadata.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return path


def transpile_onnx(
    onnx_path: str,
    target: str,
    out_dir: str,
    *,
    config: CompileConfig | None = None,
) -> ArtifactManifest:
    cfg = config or CompileConfig()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ir, weights = load_onnx_to_ir(onnx_path, infer_shapes=cfg.infer_shapes)
    ir, applied_passes = _apply_configured_passes(ir, weights, cfg)
    generated_files: list[Path] = []
    if cfg.emit_graph_ir:
        graph_ir_path = output_dir / "graph_ir.json"
        graph_ir_path.write_text(dumps_graph_json(ir), encoding="utf-8")
        generated_files.append(graph_ir_path)
    generated_files.append(
        _write_compile_metadata(
            output_dir,
            target=target,
            ir=ir,
            cfg=cfg,
            applied_passes=applied_passes,
        )
    )

    weights_path = output_dir / cfg.weights_filename
    save_weights_npz(weights_path, weights)

    if target == "jax":
        generated_files.append(emit_jax_module(ir, output_dir, entrypoint=cfg.entrypoint))
    elif target == "mlx":
        generated_files.append(emit_mlx_module(ir, output_dir, entrypoint=cfg.entrypoint))
    else:
        raise ValueError(f"Unsupported target: {target}")

    generated_files.append(weights_path)
    return ArtifactManifest(
        target=target,
        files=generated_files,
        weights_file=weights_path,
        entrypoint=cfg.entrypoint,
    )
