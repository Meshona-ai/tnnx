from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_weights_npz(path: str | Path, weights: dict[str, np.ndarray[Any, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: weights[key] for key in sorted(weights.keys())}
    np.savez(out_path, **ordered)  # type: ignore[arg-type]


def load_weights_npz(path: str | Path) -> dict[str, np.ndarray[Any, Any]]:
    data = np.load(Path(path))
    return {key: data[key] for key in sorted(data.files)}
