from __future__ import annotations

import importlib
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def test_flux_text_encoder_source_exports_supported_graph(tmp_path: Path) -> None:
    flux_support = importlib.import_module("_support.flux")
    real_flux_export_spec_or_skip = flux_support.real_flux_export_spec_or_skip

    source, spec = real_flux_export_spec_or_skip("text_encoder", load_weights=False)
    onnx_path = source.export_flux_submodule_onnx(
        "text_encoder",
        tmp_path / "flux_text_encoder.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
        export_params=False,
    )
    unsupported = source.load_unsupported_onnx_ops(onnx_path)

    assert onnx_path.exists()
    assert not unsupported, f"Unsupported ONNX ops: {unsupported}"
