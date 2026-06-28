from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def test_flux_tiny_config_e2e_jax(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")

    import examples.flux as flux

    try:
        result = flux.run_flux_jax_tiny_config_e2e_demo(tmp_path)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    assert Path(str(result["image_path"])).exists()
    assert Path(str(result["pytorch_reference_image_path"])).exists()
    assert int(result["image_size"]) > 0
    assert int(result["token_length"]) > 0
    assert float(result["max_abs"]) <= 5e-2
    assert float(result["mean_abs"]) <= 5e-3
