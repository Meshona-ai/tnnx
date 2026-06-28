from __future__ import annotations

import sys
from pathlib import Path


def _flux_source():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.source as source

    return source


def test_flux_transformer_source_exports_supported_graph(tmp_path: Path) -> None:
    source = _flux_source()
    spec = source.build_demo_export_specs()["transformer"]
    onnx_path = source.export_flux_submodule_onnx(
        "transformer",
        tmp_path / "flux_transformer.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
    )
    unsupported = source.load_unsupported_onnx_ops(onnx_path)

    assert onnx_path.exists()
    assert not unsupported, f"Unsupported ONNX ops: {unsupported}"
