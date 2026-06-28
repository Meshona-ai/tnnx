from __future__ import annotations

from collections.abc import Sequence


def build_window(ids: Sequence[int], seq_len: int, pad_id: int) -> list[int]:
    window = [int(token) for token in ids[-seq_len:]]
    if len(window) < seq_len:
        window = [pad_id] * (seq_len - len(window)) + window
    return window
