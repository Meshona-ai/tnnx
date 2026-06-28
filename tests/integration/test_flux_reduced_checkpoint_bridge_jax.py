from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def test_flux_reduced_checkpoint_bridge_jax(tmp_path: Path) -> None:
    if os.getenv("RUN_FLUX_E2E", "0") != "1":
        pytest.skip("Set RUN_FLUX_E2E=1 to run the reduced checkpoint-backed FLUX bridge test.")
    _ = pytest.importorskip("jax")

    import examples.flux as flux

    result = flux.run_flux_jax_reduced_checkpoint_bridge_demo(tmp_path)

    assert Path(str(result["image_path"])).exists()
    assert Path(str(result["pytorch_reference_image_path"])).exists()
    assert int(result["image_size"]) > 0
    assert float(result["pixel_std"]) > 0.0
    assert float(result["max_abs"]) <= 8e-2
    assert float(result["mean_abs"]) <= 8e-3
