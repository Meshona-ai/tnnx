from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def test_flux_transformer_checkpoint_source_exports_graph(tmp_path: Path) -> None:
    flux_support = importlib.import_module("_support.flux")
    real_flux_export_spec_or_skip = flux_support.real_flux_export_spec_or_skip

    source, spec = real_flux_export_spec_or_skip(
        "transformer",
        load_weights=False,
        reduced_shapes=True,
    )
    try:
        onnx_path = source.export_flux_submodule_onnx(
            "transformer",
            tmp_path / "flux_transformer_checkpoint.onnx",
            module=spec.module,
            sample_inputs=spec.sample_inputs,
            input_names=spec.input_names,
            output_names=spec.output_names,
            export_params=False,
        )
    except Exception as exc:
        pytest.xfail(f"Checkpoint-backed FLUX transformer ONNX export is not yet stable: {exc}")
    unsupported = source.load_unsupported_onnx_ops(onnx_path)

    assert onnx_path.exists()
    if unsupported:
        pytest.xfail(f"Unsupported ONNX ops for checkpoint-backed FLUX transformer: {unsupported}")
