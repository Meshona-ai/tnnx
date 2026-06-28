from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def test_flux_vae_checkpoint_runtime_jax(tmp_path: Path) -> None:
    flux_support = importlib.import_module("_support.flux")
    load_flux_example_modules = flux_support.load_flux_example_modules
    real_flux_export_spec_or_skip = flux_support.real_flux_export_spec_or_skip

    _ = pytest.importorskip("jax")
    load_generated_module, source = load_flux_example_modules()
    _, spec = real_flux_export_spec_or_skip("vae_decoder")

    onnx_path = source.export_flux_submodule_onnx(
        "vae_decoder",
        tmp_path / "flux_vae_checkpoint.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
    )
    unsupported = source.load_unsupported_onnx_ops(onnx_path)

    assert unsupported == ()

    out_dir = tmp_path / "generated_flux_vae_checkpoint_jax"
    manifest = transpile_onnx(
        str(onnx_path),
        "jax",
        str(out_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )
    module = load_generated_module(
        out_dir / "model_jax.py",
        module_name="generated_flux_vae_checkpoint_jax",
    )
    params = module.load_weights(str(manifest.weights_file))
    latents = np.asarray(spec.sample_inputs[0].detach().cpu().numpy(), dtype=np.float32)
    expected = np.asarray(
        spec.module(*spec.sample_inputs).detach().cpu().numpy(),
        dtype=np.float32,
    )
    actual = np.asarray(
        module.forward(params, {"latents": latents})["image"],
        dtype=np.float32,
    )
    diff = np.abs(actual - expected)

    assert actual.shape == expected.shape
    assert float(np.max(diff)) <= 1e-2
    assert float(np.mean(diff)) <= 1e-3
