from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn

def _onnx_cumsum(data, axis, *, exclusive=False, reverse=False):
    axis_i = int(np.asarray(axis).reshape(-1)[0])
    arr = data
    if reverse:
        arr = jnp.flip(arr, axis=axis_i)
    out = jnp.cumsum(arr, axis=axis_i)
    if exclusive:
        head = jnp.take(out, jnp.asarray([0], dtype=jnp.int32), axis=axis_i)
        zeros = jnp.zeros_like(head)
        tail_slices = [slice(None)] * out.ndim
        tail_slices[axis_i] = slice(0, -1)
        out = jnp.concatenate((zeros, out[tuple(tail_slices)]), axis=axis_i)
    if reverse:
        out = jnp.flip(out, axis=axis_i)
    return out

def _onnx_reduce_sum(data, axes=None, *, keepdims=True):
    if axes is None:
        axes_t = None
    else:
        axes_l = [int(v) for v in np.asarray(axes).reshape(-1).tolist()]
        axes_t = tuple(axes_l) if axes_l else None
    return jnp.sum(data, axis=axes_t, keepdims=keepdims)

def _onnx_mod(left, right, *, fmod=False):
    if fmod:
        return jnp.fmod(left, right)
    return jnp.mod(left, right)

def _onnx_scatter_nd(data, indices, updates):
    idx = jnp.asarray(indices, dtype=jnp.int32)
    if idx.ndim == 0:
        raise ValueError('ScatterND indices must have rank >= 1.')
    if idx.shape[-1] == 0:
        raise ValueError('ScatterND indices must include at least one index dimension.')
    flat_idx = jnp.reshape(idx, (-1, idx.shape[-1]))
    slice_shape = tuple(data.shape[int(idx.shape[-1]) :])
    flat_updates = jnp.reshape(jnp.asarray(updates), (-1,) + slice_shape)
    index_tuple = tuple(flat_idx[:, axis] for axis in range(int(idx.shape[-1])))
    if slice_shape:
        return data.at[index_tuple].set(flat_updates)
    return data.at[index_tuple].set(jnp.reshape(flat_updates, (-1,)))

def _onnx_trilu(data, diagonal=None, *, upper=True):
    diag = 0 if diagonal is None else int(np.asarray(diagonal).reshape(-1)[0])
    if upper:
        return jnp.triu(data, k=diag)
    return jnp.tril(data, k=diag)

def _load_weight_value(value):
    if value.dtype.kind in {'i', 'u', 'b'}:
        return value
    if value.dtype.kind == 'V' and value.dtype.itemsize == 2:
        bits = value.view(np.uint16).reshape(value.shape)
        return jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16)
    return jnp.asarray(value)

def load_weights(path: str) -> dict[str, jnp.ndarray]:
    data = np.load(path)
    return {
        k: _load_weight_value(data[k])
        for k in sorted(data.files)
    }

def forward(
    params: dict[str, jnp.ndarray],
    inputs: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    tensors: dict[str, jnp.ndarray] = {}
    tensors["x"] = jnp.asarray(inputs["x"])
    tensors["soft"] = jnn.softplus(tensors["x"])
    tensors["cum"] = _onnx_cumsum(tensors["soft"], params["axis"], exclusive=True, reverse=True)
    tensors["mod"] = _onnx_mod(tensors["cum"], params["mod_rhs"], fmod=True)
    tensors["sum_unused"] = _onnx_reduce_sum(tensors["mod"], params["reduce_axes"], keepdims=True)
    tensors["tri"] = _onnx_trilu(tensors["mod"], params["diag"], upper=False)
    tensors["y"] = _onnx_scatter_nd(tensors["tri"], params["indices"], params["updates"])
    return {"y": tensors["y"]}

_STATIC_META_PARAM_NAMES = frozenset(('axis', 'diag', 'reduce_axes'))
_FORWARD_JIT_CACHE = {}

def _static_meta_cache_key(static_params):
    return tuple(
        (key, np.asarray(value).dtype.str, tuple(np.asarray(value).shape), np.asarray(value).tobytes())
        for key, value in sorted(static_params.items())
    )

def forward_jit(params: dict[str, jnp.ndarray], inputs: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    static_params = {
        key: params[key]
        for key in _STATIC_META_PARAM_NAMES
        if key in params
    }
    dynamic_params = {
        key: value
        for key, value in params.items()
        if key not in _STATIC_META_PARAM_NAMES
    }
    cache_key = _static_meta_cache_key(static_params)
    compiled = _FORWARD_JIT_CACHE.get(cache_key)
    if compiled is None:
        def _closed_forward(actual_params, actual_inputs):
            merged = dict(static_params)
            merged.update(actual_params)
            return forward(merged, actual_inputs)

        compiled = jax.jit(_closed_forward)
        _FORWARD_JIT_CACHE[cache_key] = compiled
    return compiled(dynamic_params, inputs)
