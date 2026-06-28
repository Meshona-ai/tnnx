from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _flux_demo_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.transpile_and_generate_jax as demo

    return demo


def test_flux_prompt_bridge_contract_uses_single_text_hidden() -> None:
    demo = _flux_demo_module()
    hidden = np.array([[[1.0, 3.0], [5.0, 7.0]]], dtype=np.float32)

    combined, pooled = demo._prompt_inputs_from_single_text_hidden(hidden)

    assert np.array_equal(combined, hidden)
    assert np.array_equal(pooled, np.array([[3.0, 5.0]], dtype=np.float32))


def test_flux_prompt_bridge_contract_repeats_to_target_length() -> None:
    demo = _flux_demo_module()
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    expanded, pooled = demo._prompt_inputs_from_single_text_hidden(hidden, target_seq_len=5)

    assert expanded.shape == (1, 5, 2)
    assert np.array_equal(expanded[:, :2, :], hidden)
    assert np.array_equal(expanded[:, 2:4, :], hidden)
    assert np.array_equal(expanded[:, 4:, :], hidden[:, :1, :])
    assert np.array_equal(pooled, np.mean(expanded, axis=1))


def test_flux_prompt_bridge_contract_builds_single_encoder_prompt_inputs() -> None:
    demo = _flux_demo_module()
    default_input_ids = np.array([[0, 1]], dtype=np.int64)

    text_ids, text_hidden, prompt_embeddings, pooled = demo._build_single_encoder_prompt_inputs(
        default_input_ids=default_input_ids,
        provided_input_ids=np.array([[3, 5]], dtype=np.int64),
        target_seq_len=3,
        encode_hidden=lambda token_ids: np.expand_dims(token_ids.astype(np.float32), axis=-1),
    )

    assert np.array_equal(text_ids, np.array([[3, 5]], dtype=np.int64))
    assert np.array_equal(text_hidden, np.array([[[3.0], [5.0]]], dtype=np.float32))
    assert np.array_equal(prompt_embeddings, np.array([[[3.0], [5.0], [3.0]]], dtype=np.float32))
    assert np.array_equal(pooled, np.array([[11.0 / 3.0]], dtype=np.float32))


def test_flux_prompt_bridge_contract_rejects_non_rank_three_hidden() -> None:
    demo = _flux_demo_module()
    hidden = np.zeros((1, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="Expected 3D text hidden tensor"):
        demo._prompt_inputs_from_single_text_hidden(hidden)
