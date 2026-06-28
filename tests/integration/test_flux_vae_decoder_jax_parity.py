from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig


def _example_modules():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.source as source
    from examples.common import load_generated_module

    return load_generated_module, source


def test_flux_vae_decoder_jax_parity(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    load_generated_module, source = _example_modules()

    spec = source.build_demo_export_specs()["vae_decoder"]
    onnx_path = source.export_flux_submodule_onnx(
        "vae_decoder",
        tmp_path / "flux_vae_decoder.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
    )
    manifest = transpile_onnx(
        str(onnx_path),
        "jax",
        str(tmp_path / "generated_flux_vae_decoder_jax"),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )

    module = load_generated_module(
        tmp_path / "generated_flux_vae_decoder_jax" / "model_jax.py",
        module_name="generated_flux_vae_decoder_jax",
    )
    params = module.load_weights(str(manifest.weights_file))

    latents = spec.sample_inputs[0]
    expected = spec.module(latents).detach().cpu().numpy()
    actual = np.asarray(
        module.forward(
            params,
            {"latents": latents.detach().cpu().numpy()},
        )["image"]
    )

    assert np.allclose(actual, expected, rtol=5e-4, atol=5e-4), (
        f"Reduced FLUX VAE JAX parity failed: max_abs={np.max(np.abs(actual - expected))}"
    )
