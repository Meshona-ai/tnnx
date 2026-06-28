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


def test_flux_token_ids_npz_contract_loads_expected_arrays(tmp_path: Path) -> None:
    demo = _flux_demo_module()
    fixture = tmp_path / "token_fixture.npz"
    expected_ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    np.savez(fixture, input_ids=expected_ids)

    input_ids = demo._load_token_ids_npz(fixture)

    assert np.array_equal(input_ids, expected_ids)


def test_flux_token_ids_npz_contract_requires_token_array(tmp_path: Path) -> None:
    demo = _flux_demo_module()
    fixture = tmp_path / "token_fixture_bad.npz"
    np.savez(fixture, unexpected=np.array([[0, 1, 2, 3]], dtype=np.int64))

    with pytest.raises(
        ValueError,
        match="Token fixture must contain 'input_ids'",
    ):
        demo._load_token_ids_npz(fixture)
