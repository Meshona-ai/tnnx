from __future__ import annotations

import numpy as np

from tnnx.runtime.weights import load_weights_npz, save_weights_npz


def test_weight_roundtrip_contract(tmp_path: str) -> None:
    weights = {
        "w": np.array([[1.0, 2.0]], dtype=np.float32),
        "b": np.array([3.0], dtype=np.float32),
    }
    path = tmp_path / "weights.npz"
    save_weights_npz(path, weights)
    loaded = load_weights_npz(path)

    assert sorted(loaded.keys()) == ["b", "w"]
    assert loaded["w"].dtype == np.float32
