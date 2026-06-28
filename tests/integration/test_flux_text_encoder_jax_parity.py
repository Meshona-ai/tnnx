from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def test_flux_text_encoder_jax_parity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import examples.flux.source as source
    from examples.common import load_generated_module
    from tnnx.api import transpile_onnx
    from tnnx.config import CompileConfig

    _ = pytest.importorskip("jax")
    spec = source.build_dummy_flux2_export_specs()["text_encoder"]

    onnx_path = source.export_flux_submodule_onnx(
        "text_encoder",
        tmp_path / "flux_text_encoder.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
    )
    manifest = transpile_onnx(
        str(onnx_path),
        "jax",
        str(tmp_path / "generated_flux_text_encoder_jax"),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )

    module = load_generated_module(
        tmp_path / "generated_flux_text_encoder_jax" / "model_jax.py",
        module_name="generated_flux_text_encoder_jax",
    )
    params = module.load_weights(str(manifest.weights_file))

    input_ids = spec.sample_inputs[0]
    expected = spec.module(input_ids).detach().cpu().numpy()
    actual = np.asarray(
        module.forward(
            params,
            {"input_ids": input_ids.detach().cpu().numpy()},
        )["hidden"]
    )

    assert np.allclose(actual, expected, rtol=1e-3, atol=1e-3), (
        "Synthetic FLUX.2 text_encoder JAX parity failed: "
        f"max_abs={np.max(np.abs(actual - expected))}"
    )
