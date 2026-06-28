from __future__ import annotations

import numpy as np
import mlx.core as mx

def load_weights(path: str) -> dict[str, mx.array]:
    data = np.load(path)
    params = {k: mx.array(data[k]) for k in sorted(data.files)}
    return params

def forward(
    params: dict[str, mx.array],
    inputs: dict[str, mx.array],
) -> dict[str, mx.array]:
    tensors: dict[str, mx.array] = {}
    tensors["x"] = mx.asarray(inputs["x"])
    tensors["y"] = (mx.matmul(tensors["x"], params["w"]) + params["b"])
    return {"y": tensors["y"]}
