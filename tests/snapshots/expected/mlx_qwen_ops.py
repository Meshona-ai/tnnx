from __future__ import annotations

import numpy as np
import mlx.core as mx

def _onnx_cumsum(data, axis, *, exclusive=False, reverse=False):
    arr = np.asarray(data)
    axis_i = int(np.asarray(axis).reshape(-1)[0])
    if reverse:
        arr = np.flip(arr, axis=axis_i)
    out = np.cumsum(arr, axis=axis_i)
    if exclusive:
        shifted = np.zeros_like(out)
        head = [slice(None)] * out.ndim
        tail = [slice(None)] * out.ndim
        head[axis_i] = slice(1, None)
        tail[axis_i] = slice(0, -1)
        shifted[tuple(head)] = out[tuple(tail)]
        out = shifted
    if reverse:
        out = np.flip(out, axis=axis_i)
    return mx.array(out)

def _onnx_reduce_sum(data, axes=None, *, keepdims=True):
    if axes is None:
        axes_t = None
    else:
        axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]
        axes_t = tuple(axes_l) if axes_l else None
    return mx.array(np.sum(np.asarray(data), axis=axes_t, keepdims=keepdims))

def _onnx_scatter_nd(data, indices, updates):
    arr = np.asarray(data).copy()
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim == 0:
        raise ValueError('ScatterND indices must have rank >= 1.')
    if idx.shape[-1] == 0:
        raise ValueError('ScatterND indices must include at least one index dimension.')
    flat_idx = idx.reshape(-1, idx.shape[-1])
    slice_shape = tuple(arr.shape[int(idx.shape[-1]) :])
    flat_updates = np.asarray(updates).reshape((flat_idx.shape[0],) + slice_shape)
    if slice_shape:
        for row_idx, row in enumerate(flat_idx):
            arr[tuple(row.tolist())] = flat_updates[row_idx]
    else:
        arr[tuple(flat_idx.T)] = flat_updates.reshape(-1)
    return mx.array(arr)

def _onnx_trilu(data, diagonal=None, *, upper=True):
    diag = 0 if diagonal is None else int(np.asarray(diagonal).reshape(-1)[0])
    arr = np.asarray(data)
    if upper:
        return mx.array(np.triu(arr, k=diag))
    return mx.array(np.tril(arr, k=diag))

def _onnx_mod(left, right, *, fmod=False):
    if fmod:
        return mx.array(np.fmod(np.asarray(left), np.asarray(right)))
    return mx.remainder(left, right)

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
    tensors["soft"] = mx.log(1.0 + mx.exp(tensors["x"]))
    tensors["cum"] = _onnx_cumsum(tensors["soft"], params["axis"], exclusive=True, reverse=True)
    tensors["mod"] = _onnx_mod(tensors["cum"], params["mod_rhs"], fmod=True)
    tensors["sum_unused"] = _onnx_reduce_sum(tensors["mod"], params["reduce_axes"], keepdims=True)
    tensors["tri"] = _onnx_trilu(tensors["mod"], params["diag"], upper=False)
    tensors["y"] = _onnx_scatter_nd(tensors["tri"], params["indices"], params["updates"])
    return {"y": tensors["y"]}
