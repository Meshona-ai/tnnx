from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image


def _run_flux_prompt_to_image_demo():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.flux import run_flux_jax_prompt_to_image_demo

    return run_flux_jax_prompt_to_image_demo


def test_flux_jax_prompt_to_image_smoke(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    run_flux_jax_prompt_to_image_demo = _run_flux_prompt_to_image_demo()

    result = run_flux_jax_prompt_to_image_demo(tmp_path, steps=2)
    image_path = Path(str(result["image_path"]))

    assert result["prompt_source"] == "text_encoder"
    assert image_path.exists()
    assert Path(str(result["text_encoder_module"])).exists()
    assert Path(str(result["pytorch_reference_image_path"])).exists()
    with Image.open(image_path) as image:
        assert image.size == (int(result["image_size"]), int(result["image_size"]))

    assert float(result["text_encoder_max_abs"]) < 1e-6
    assert int(result["token_length"]) > 0
    assert float(result["pytorch_reference_pixel_std"]) > 1e-3
    assert float(result["pixel_std"]) > 1e-3
