from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn

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
    tensors["y"] = (jnp.matmul(tensors["x"], params["w"]) + params["b"])
    return {"y": tensors["y"]}

forward_jit = jax.jit(forward)
