from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _real_flux_vae_export_spec_or_skip() -> tuple[Any, Any]:
    submodule = "vae_decoder"
    if os.getenv("RUN_FLUX_E2E", "0") != "1":
        pytest.skip(f"Set RUN_FLUX_E2E=1 to run checkpoint-backed FLUX {submodule} checks.")

    import examples.flux.source as source

    try:
        snapshot = source.resolve_flux_snapshot()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    if not source.snapshot_has_submodule_weights(snapshot, submodule):
        pytest.skip(f"Missing {submodule} weights under snapshot: {snapshot}")

    try:
        spec = source._real_export_spec(submodule)
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))

    return source, spec


def test_flux_vae_checkpoint_transpile_jax(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    source, spec = _real_flux_vae_export_spec_or_skip()

    onnx_path = source.export_flux_submodule_onnx(
        "vae_decoder",
        tmp_path / "flux_vae_checkpoint.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
    )
    unsupported = source.load_unsupported_onnx_ops(onnx_path)

    assert onnx_path.exists()
    assert unsupported == ()

    out_dir = tmp_path / "generated_flux_vae_checkpoint_jax"
    manifest = transpile_onnx(
        str(onnx_path),
        "jax",
        str(out_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )

    assert manifest.target == "jax"
    assert manifest.weights_file is not None
    assert manifest.weights_file.exists()
    assert (out_dir / "model_jax.py").exists()
