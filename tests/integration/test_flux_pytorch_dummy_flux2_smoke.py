from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


def _run_flux_pytorch_dummy_flux2_demo():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.flux import run_flux_pytorch_dummy_flux2_demo

    return run_flux_pytorch_dummy_flux2_demo


def test_flux_pytorch_dummy_flux2_smoke(tmp_path: Path) -> None:
    run_flux_pytorch_dummy_flux2_demo = _run_flux_pytorch_dummy_flux2_demo()

    result = run_flux_pytorch_dummy_flux2_demo(tmp_path, steps=2)
    image_path = Path(str(result["image_path"]))

    assert result["prompt_source"] == "dummy_flux2_text_encoder"
    assert int(result["token_length"]) == 4
    assert image_path.exists()
    with Image.open(image_path) as image:
        assert image.size == (int(result["image_size"]), int(result["image_size"]))

    assert float(result["pixel_std"]) > 1e-3
