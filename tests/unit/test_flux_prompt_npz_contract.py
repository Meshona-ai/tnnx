from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _flux_demo_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.transpile_and_generate_jax as demo

    return demo


def test_flux_prompt_npz_contract_loads_expected_arrays(tmp_path: Path) -> None:
    demo = _flux_demo_module()
    fixture = tmp_path / "prompt_fixture.npz"
    expected_prompt = np.ones((1, 4, 8), dtype=np.float32)
    expected_pooled = np.zeros((1, 8), dtype=np.float32)
    np.savez(fixture, prompt_embeddings=expected_prompt, pooled_prompt=expected_pooled)

    prompt, pooled = demo._load_prompt_npz(fixture)

    assert np.array_equal(prompt, expected_prompt)
    assert np.array_equal(pooled, expected_pooled)


def test_flux_prompt_npz_contract_requires_both_prompt_arrays(tmp_path: Path) -> None:
    demo = _flux_demo_module()
    fixture = tmp_path / "prompt_fixture_bad.npz"
    np.savez(fixture, prompt_embeddings=np.ones((1, 4, 8), dtype=np.float32))

    with pytest.raises(
        ValueError,
        match="Prompt fixture must contain 'prompt_embeddings' and 'pooled_prompt'",
    ):
        demo._load_prompt_npz(fixture)
