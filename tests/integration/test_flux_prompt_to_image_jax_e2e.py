from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def test_flux_prompt_to_image_jax_e2e_checkpoint_contract(tmp_path: Path) -> None:
    if os.getenv("RUN_FLUX_E2E", "0") != "1":
        pytest.skip("Set RUN_FLUX_E2E=1 to run the reduced checkpoint-backed FLUX artifact gate.")

    flux_support = importlib.import_module("_support.flux")
    source = flux_support.load_flux_source()

    try:
        snapshot = source.resolve_flux_snapshot()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    import examples.flux.transpile_and_generate_jax as demo

    report = demo.prepare_flux_jax_checkpoint_artifacts(
        tmp_path / "out",
        checkpoint_reduced_shapes=True,
    )
    expected_submodules = list(source.real_submodule_export_order(snapshot=snapshot))

    assert report["submodule_order"] == expected_submodules
    assert report["ready_count"] == len(expected_submodules)
    assert report["blocked_count"] == 0
    assert report["missing_count"] == 0
    for submodule in expected_submodules:
        assert report["submodules"][submodule]["status"] == "ready"
    assert report["submodules"]["transformer"]["used_reduced_config"] is True
    assert report["submodules"]["text_encoder"]["used_reduced_config"] is True
    assert report["submodules"]["vae_decoder"]["used_reduced_config"] is False
