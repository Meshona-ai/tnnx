from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _flux_modules():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.source as source
    from examples.flux import run_flux_jax_demo

    return run_flux_jax_demo, source


def test_flux_jax_image_smoke_accepts_custom_prompt_embeddings(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    run_flux_jax_demo, source = _flux_modules()

    prompt = np.full((1, source.DEMO_SEQUENCE, source.DEMO_DIM), 0.25, dtype=np.float32)
    pooled = np.full((1, source.DEMO_DIM), -0.5, dtype=np.float32)
    result = run_flux_jax_demo(
        tmp_path,
        steps=2,
        prompt_embeddings=prompt,
        pooled_prompt=pooled,
    )
    image_path = Path(str(result["image_path"]))

    assert result["prompt_source"] == "custom"
    assert image_path.exists()
    with Image.open(image_path) as image:
        assert image.size == (int(result["image_size"]), int(result["image_size"]))
    assert float(result["pixel_std"]) > 1e-3


def test_flux_jax_image_smoke_rejects_partial_custom_prompt_inputs(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    run_flux_jax_demo, source = _flux_modules()

    prompt = np.zeros((1, source.DEMO_SEQUENCE, source.DEMO_DIM), dtype=np.float32)
    with pytest.raises(
        ValueError,
        match="prompt_embeddings and pooled_prompt must be provided together",
    ):
        run_flux_jax_demo(tmp_path, steps=1, prompt_embeddings=prompt)
