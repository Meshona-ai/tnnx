from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


class _FakeQwenTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.bos_token = "<bos>"
        self.bos_token_id = 2
        self._stoi = {
            " ": 3,
            "a": 4,
            "c": 5,
            "d": 6,
            "e": 7,
            "i": 8,
            "l": 9,
            "m": 10,
            "o": 11,
            "p": 12,
            "r": 13,
            "t": 14,
            "w": 15,
        }
        self._itos = {value: key for key, value in self._stoi.items()}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [self._stoi.get(ch, 15) for ch in text]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        chars: list[str] = []
        for token in ids:
            if skip_special_tokens and token in {
                self.pad_token_id,
                self.eos_token_id,
                self.bos_token_id,
            }:
                continue
            chars.append(self._itos.get(int(token), "x"))
        return "".join(chars)


def test_qwen3_5_synthetic_example_mlx(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("mlx.core")
    _ = pytest.importorskip("transformers")
    from transformers import Qwen3_5ForCausalLM

    from examples.qwen.infer_qwen3_5_from_transformers_mlx import (
        build_tiny_qwen3_5_config,
        run_demo,
    )

    torch.manual_seed(7)
    np.random.seed(7)
    config = build_tiny_qwen3_5_config()
    model = Qwen3_5ForCausalLM(config).eval()

    result = run_demo(
        tmp_path,
        prompt="write code",
        model_id="synthetic-qwen3.5",
        context_window=16,
        max_new_tokens=4,
        model=model,
        tokenizer=_FakeQwenTokenizer(),
    )

    assert Path(str(result["onnx_path"])).exists()
    assert Path(str(result["graph_path"])).exists()
    assert Path(str(result["generated_module"])).exists()
    assert Path(str(result["weights_path"])).exists()
    assert result["unsupported_ops"] == ()
    assert "ScatterND" in result["exported_ops"]
    assert float(result["max_abs"]) <= 3e-3
    assert int(result["generated_token_count"]) >= 1
    assert str(result["generated_text"]).strip() != ""
    assert any(ch.isalnum() for ch in str(result["generated_text"]))
    assert str(result["torch_version"]) != ""
    assert str(result["transformers_version"]) != ""
    assert str(result["onnx_version"]) != ""
    assert str(result["mlx_version"]) != ""


def test_qwen3_5_real_example_mlx(tmp_path: Path) -> None:
    if os.getenv("RUN_QWEN_MLX_E2E", "0") != "1":
        pytest.skip("Set RUN_QWEN_MLX_E2E=1 to run the full Qwen3.5 MLX path.")

    _ = pytest.importorskip("torch")
    _ = pytest.importorskip("mlx.core")
    _ = pytest.importorskip("transformers")
    from examples.qwen.infer_qwen3_5_from_transformers_mlx import run_demo

    result = run_demo(
        tmp_path,
        prompt="hello",
        context_window=32,
        max_new_tokens=8,
    )

    assert Path(str(result["onnx_path"])).exists()
    assert Path(str(result["embed_onnx_path"])).exists()
    assert Path(str(result["graph_path"])).exists()
    assert Path(str(result["generated_module"])).exists()
    assert Path(str(result["weights_path"])).exists()
    assert result["unsupported_ops"] == ()
    assert int(result["generated_token_count"]) >= 1
    assert str(result["generated_text"]).strip() != ""
    assert any(ch.isalnum() for ch in str(result["generated_text"]))
