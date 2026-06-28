from __future__ import annotations

import numpy as np
import mlx.core as mx

def _onnx_rotary_embedding(data, position_ids, cos_cache, sin_cache, *, rotary_dim, num_heads, interleaved=False):
    del num_heads
    if interleaved:
        raise ValueError('RotaryEmbedding interleaved mode is not supported yet.')
    arr = np.asarray(data)
    rotary_i = int(rotary_dim)
    if rotary_i <= 0:
        return mx.array(arr)
    half = rotary_i // 2
    pos = np.asarray(position_ids, dtype=np.int64)
    cos = np.take(np.asarray(cos_cache), pos, axis=0)
    sin = np.take(np.asarray(sin_cache), pos, axis=0)
    cos = np.expand_dims(cos, axis=1)
    sin = np.expand_dims(sin, axis=1)
    left = arr[..., :half]
    right = arr[..., half:rotary_i]
    rotated = np.concatenate((
        left * cos - right * sin,
        right * cos + left * sin,
    ), axis=-1)
    out = np.concatenate((rotated, arr[..., rotary_i:]), axis=-1)
    return mx.array(out.astype(arr.dtype, copy=False))

def _onnx_skip_rmsnorm(data, skip, gamma, epsilon=1e-5, *, return_residual=True):
    data_arr = np.asarray(data)
    skip_arr = np.asarray(skip)
    residual = data_arr + skip_arr
    rms = np.mean(residual * residual, axis=-1, keepdims=True)
    normalized = (residual / np.sqrt(rms + float(epsilon))) * np.asarray(gamma)
    if return_residual:
        return mx.array(normalized), mx.array(residual)
    return mx.array(normalized)

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
    q = np.asarray(query)
    k = np.asarray(key)
    v = np.asarray(value)
    pk = np.asarray(past_key)
    pv = np.asarray(past_value)
    batch, query_len, q_width = q.shape
    num_heads_i = int(num_heads)
    kv_heads_i = int(kv_num_heads)
    if num_heads_i <= 0 or kv_heads_i <= 0 or q_width % num_heads_i != 0:
        raise ValueError('Invalid GroupQueryAttention head configuration.')
    head_dim = q_width // num_heads_i
    if k.shape[-1] != kv_heads_i * head_dim or v.shape[-1] != kv_heads_i * head_dim:
        raise ValueError('GroupQueryAttention key/value width mismatch.')
    if num_heads_i % kv_heads_i != 0:
        raise ValueError('num_heads must be divisible by kv_num_heads.')
    qh = q.reshape(batch, query_len, num_heads_i, head_dim).transpose(0, 2, 1, 3)
    kh = k.reshape(batch, query_len, kv_heads_i, head_dim).transpose(0, 2, 1, 3)
    vh = v.reshape(batch, query_len, kv_heads_i, head_dim).transpose(0, 2, 1, 3)
    present_key = np.concatenate((pk, kh), axis=2)
    present_value = np.concatenate((pv, vh), axis=2)
    repeats = num_heads_i // kv_heads_i
    full_key = np.repeat(present_key, repeats, axis=1)
    full_value = np.repeat(present_value, repeats, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', qh, full_key) * float(scale)
    if float(softcap) > 0.0:
        scores = np.tanh(scores / float(softcap)) * float(softcap)
    if attention_bias is not None:
        scores = scores + np.asarray(attention_bias, dtype=scores.dtype)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    context = np.einsum('bhqk,bhkd->bhqd', probs, full_value)
    context = context.transpose(0, 2, 1, 3).reshape(batch, query_len, q_width)
    return mx.array(context), mx.array(present_key), mx.array(present_value)

def load_weights(path: str) -> dict[str, mx.array]:
    data = np.load(path)
    params = {k: mx.array(data[k]) for k in sorted(data.files)}
    return params

def forward(
    params: dict[str, mx.array],
    inputs: dict[str, mx.array],
) -> dict[str, mx.array]:
    tensors: dict[str, mx.array] = {}
    tensors["rot_in"] = mx.asarray(inputs["rot_in"])
    tensors["pos"] = mx.asarray(inputs["pos"])
    tensors["skip"] = mx.asarray(inputs["skip"])
    tensors["q"] = mx.asarray(inputs["q"])
    tensors["k"] = mx.asarray(inputs["k"])
    tensors["v"] = mx.asarray(inputs["v"])
    tensors["past_k"] = mx.asarray(inputs["past_k"])
    tensors["past_v"] = mx.asarray(inputs["past_v"])
    tensors["bias"] = mx.asarray(inputs["bias"])
    tensors["rot_out"] = _onnx_rotary_embedding(tensors["rot_in"], tensors["pos"], params["cos_cache"], params["sin_cache"], rotary_dim=4, num_heads=2, interleaved=False)
    _node_n1 = _onnx_skip_rmsnorm(tensors["rot_out"], tensors["skip"], params["gamma"], epsilon=1e-06, return_residual=True)
    tensors["skip_norm"] = _node_n1[0]
    tensors["skip_residual"] = _node_n1[1]
    _node_n2 = _onnx_group_query_attention(tensors["q"], tensors["k"], tensors["v"], tensors["past_k"], tensors["past_v"], attention_bias=tensors["bias"], num_heads=2, kv_num_heads=1, scale=0.5, softcap=0.0)
    tensors["attn_out"] = _node_n2[0]
    tensors["present_k"] = _node_n2[1]
    tensors["present_v"] = _node_n2[2]
    return {"skip_norm": tensors["skip_norm"], "attn_out": tensors["attn_out"], "present_k": tensors["present_k"], "present_v": tensors["present_v"]}
