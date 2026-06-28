from __future__ import annotations

import numpy as np
import mlx.core as mx

def _onnx_slice(data, starts, ends, axes=None, steps=None):
    arr = np.asarray(data)
    starts_l = [int(v) for v in np.asarray(starts).tolist()]
    ends_l = [int(v) for v in np.asarray(ends).tolist()]
    if axes is None:
        axes_l = list(range(len(starts_l)))
    else:
        axes_l = [int(v) for v in np.asarray(axes).tolist()]
    if steps is None:
        steps_l = [1] * len(starts_l)
    else:
        steps_l = [int(v) for v in np.asarray(steps).tolist()]
    slices = [slice(None)] * arr.ndim
    for s, e, a, st in zip(starts_l, ends_l, axes_l, steps_l, strict=False):
        slices[a] = slice(s, e, st)
    return mx.array(arr[tuple(slices)])

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
    tensors["g"] = mx.take(tensors["x"], mx.array(params["indices"]), axis=1)
    tensors["y"] = _onnx_slice(tensors["g"], params["starts"], params["ends"], params["axes"], params["steps"])
    return {"y": tensors["y"]}
