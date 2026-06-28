from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn

def _onnx_slice(data, starts, ends, axes=None, steps=None):
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
    slices = [slice(None)] * data.ndim
    for s, e, a, st in zip(starts_l, ends_l, axes_l, steps_l, strict=False):
        slices[a] = slice(s, e, st)
    return data[tuple(slices)]

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
    tensors["g"] = jnp.take(tensors["x"], jnp.asarray(params["indices"], dtype=jnp.int32), axis=1)
    tensors["y"] = _onnx_slice(tensors["g"], params["starts"], params["ends"], params["axes"], params["steps"])
    return {"y": tensors["y"]}

_STATIC_META_PARAM_NAMES = frozenset(('axes', 'ends', 'starts', 'steps'))
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
