from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn

def _onnx_rotary_embedding(data, position_ids, cos_cache, sin_cache, *, rotary_dim, num_heads, interleaved=False):
    del num_heads
    if interleaved:
        raise ValueError('RotaryEmbedding interleaved mode is not supported yet.')
    rotary_i = int(rotary_dim)
    if rotary_i <= 0:
        return data
    half = rotary_i // 2
    pos = jnp.asarray(position_ids, dtype=jnp.int32)
    cos = jnp.take(jnp.asarray(cos_cache), pos, axis=0)
    sin = jnp.take(jnp.asarray(sin_cache), pos, axis=0)
    cos = jnp.expand_dims(cos, axis=1)
    sin = jnp.expand_dims(sin, axis=1)
    left = data[..., :half]
    right = data[..., half:rotary_i]
    rotated = jnp.concatenate((
        left * cos - right * sin,
        right * cos + left * sin,
    ), axis=-1)
    return jnp.concatenate((rotated, data[..., rotary_i:]), axis=-1)

def _onnx_skip_rmsnorm(data, skip, gamma, epsilon=1e-5, *, return_residual=True):
    residual = data + skip
    rms = jnp.mean(jnp.square(residual), axis=-1, keepdims=True)
    normalized = (residual / jnp.sqrt(rms + epsilon)) * gamma
    if return_residual:
        return normalized, residual
    return normalized

def _onnx_group_query_attention(
    query,
    key,
    value,
    past_key,
    past_value,
    attention_bias=None,
    *,
    num_heads=1,
    kv_num_heads=1,
    scale=1.0,
    softcap=0.0,
):
    batch, query_len, q_width = query.shape
    num_heads_i = int(num_heads)
    kv_heads_i = int(kv_num_heads)
    if num_heads_i <= 0 or kv_heads_i <= 0 or q_width % num_heads_i != 0:
        raise ValueError('Invalid GroupQueryAttention head configuration.')
    head_dim = q_width // num_heads_i
    if key.shape[-1] != kv_heads_i * head_dim or value.shape[-1] != kv_heads_i * head_dim:
        raise ValueError('GroupQueryAttention key/value width mismatch.')
    if num_heads_i % kv_heads_i != 0:
        raise ValueError('num_heads must be divisible by kv_num_heads.')
    query_h = jnp.transpose(jnp.reshape(query, (batch, query_len, num_heads_i, head_dim)), (0, 2, 1, 3))
    key_h = jnp.transpose(jnp.reshape(key, (batch, query_len, kv_heads_i, head_dim)), (0, 2, 1, 3))
    value_h = jnp.transpose(jnp.reshape(value, (batch, query_len, kv_heads_i, head_dim)), (0, 2, 1, 3))
    present_key = jnp.concatenate((past_key, key_h), axis=2)
    present_value = jnp.concatenate((past_value, value_h), axis=2)
    repeats = num_heads_i // kv_heads_i
    full_key = jnp.repeat(present_key, repeats, axis=1)
    full_value = jnp.repeat(present_value, repeats, axis=1)
    scores = jnp.einsum('bhqd,bhkd->bhqk', query_h, full_key) * float(scale)
    if float(softcap) > 0.0:
        scores = jnp.tanh(scores / float(softcap)) * float(softcap)
    if attention_bias is not None:
        scores = scores + jnp.asarray(attention_bias, dtype=scores.dtype)
    probs = jnn.softmax(scores, axis=-1)
    context = jnp.einsum('bhqk,bhkd->bhqd', probs, full_value)
    context = jnp.reshape(jnp.transpose(context, (0, 2, 1, 3)), (batch, query_len, q_width))
    return context, present_key, present_value

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
    tensors["rot_in"] = jnp.asarray(inputs["rot_in"])
    tensors["pos"] = jnp.asarray(inputs["pos"])
    tensors["skip"] = jnp.asarray(inputs["skip"])
    tensors["q"] = jnp.asarray(inputs["q"])
    tensors["k"] = jnp.asarray(inputs["k"])
    tensors["v"] = jnp.asarray(inputs["v"])
    tensors["past_k"] = jnp.asarray(inputs["past_k"])
    tensors["past_v"] = jnp.asarray(inputs["past_v"])
    tensors["bias"] = jnp.asarray(inputs["bias"])
    tensors["rot_out"] = _onnx_rotary_embedding(tensors["rot_in"], tensors["pos"], params["cos_cache"], params["sin_cache"], rotary_dim=4, num_heads=2, interleaved=False)
    _node_n1 = _onnx_skip_rmsnorm(tensors["rot_out"], tensors["skip"], params["gamma"], epsilon=1e-06, return_residual=True)
    tensors["skip_norm"] = _node_n1[0]
    tensors["skip_residual"] = _node_n1[1]
    _node_n2 = _onnx_group_query_attention(tensors["q"], tensors["k"], tensors["v"], tensors["past_k"], tensors["past_v"], attention_bias=tensors["bias"], num_heads=2, kv_num_heads=1, scale=0.5, softcap=0.0)
    tensors["attn_out"] = _node_n2[0]
    tensors["present_k"] = _node_n2[1]
    tensors["present_v"] = _node_n2[2]
    return {"skip_norm": tensors["skip_norm"], "attn_out": tensors["attn_out"], "present_k": tensors["present_k"], "present_v": tensors["present_v"]}

forward_jit = jax.jit(forward)
