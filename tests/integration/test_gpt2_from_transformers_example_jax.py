from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self._stoi = {
            " ": 2,
            "H": 3,
            "e": 4,
            "l": 5,
            "o": 6,
            "f": 7,
            "r": 8,
            "m": 9,
            "t": 10,
            "n": 11,
            "x": 12,
        }
        self._itos = {value: key for key, value in self._stoi.items()}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [self._stoi.get(ch, 2) for ch in text]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        chars: list[str] = []
        for token in ids:
            if skip_special_tokens and token in {self.pad_token_id, self.eos_token_id}:
                continue
            chars.append(self._itos.get(int(token), "?"))
        return "".join(chars)


def test_gpt2_from_transformers_example_jax(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("jax")
    from examples.nanogpt.infer_gpt2_from_transformers_jax import run_demo

    class FakeGPT2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(n_positions=8)
            self.embed = torch.nn.Embedding(16, 8)
            self.proj = torch.nn.Linear(8, 16)

        def forward(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            use_cache: bool = False,
        ) -> SimpleNamespace:
            _ = attention_mask
            _ = use_cache
            logits = self.proj(self.embed(input_ids))
            return SimpleNamespace(logits=logits)

    result = run_demo(
        tmp_path,
        prompt="Hello",
        model_id="fake-gpt2",
        context_window=8,
        max_new_tokens=4,
        model=FakeGPT2(),
        tokenizer=_FakeTokenizer(),
    )
    assert Path(str(result["onnx_path"])).exists()
    assert Path(str(result["graph_path"])).exists()
    assert Path(str(result["generated_module"])).exists()
    assert Path(str(result["weights_path"])).exists()
    assert int(result["generated_token_count"]) <= 4
