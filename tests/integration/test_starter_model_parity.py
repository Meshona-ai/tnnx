from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("generated_starter_model", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StarterTransformerBlock:
    @staticmethod
    def build(torch: Any) -> Any:
        class TransformerEncoderBlock(torch.nn.Module):
            def __init__(
                self,
                d_model: int = 256,
                n_heads: int = 4,
                d_ff: int = 512,
                dropout: float = 0.1,
            ) -> None:
                super().__init__()
                self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                self.ff = torch.nn.Sequential(
                    torch.nn.Linear(d_model, d_ff),
                    torch.nn.GELU(),
                    torch.nn.Linear(d_ff, d_model),
                )
                self.norm1 = torch.nn.LayerNorm(d_model)
                self.norm2 = torch.nn.LayerNorm(d_model)
                self.dropout = torch.nn.Dropout(dropout)

            def forward(self, x: Any) -> Any:
                attn_out, _ = self.attn(x, x, x)
                x = self.norm1(x + self.dropout(attn_out))
                x = self.norm2(x + self.dropout(self.ff(x)))
                return x

        return TransformerEncoderBlock().eval()


def test_starter_model_has_cross_backend_runtime_parity(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("jax")
    _ = pytest.importorskip("mlx.core")

    torch.manual_seed(7)
    np.random.seed(7)

    model = _StarterTransformerBlock.build(torch)
    sample = torch.randn(1, 8, 256, dtype=torch.float32)
    with torch.no_grad():
        expected = model(sample).detach().cpu().numpy()

    onnx_path = tmp_path / "starter_model.onnx"
    torch.onnx.export(
        model,
        sample,
        onnx_path,
        input_names=["x"],
        output_names=["y"],
        opset_version=18,
        dynamo=False,
    )

    actual_by_target: dict[str, np.ndarray[Any, Any]] = {}
    for target, rtol, atol in (("jax", 1e-5, 1e-6), ("mlx", 2e-5, 2e-6)):
        out_dir = tmp_path / f"generated_{target}"
        manifest = transpile_onnx(
            str(onnx_path),
            target,
            str(out_dir),
            config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
        )
        module = _load_module(out_dir / f"model_{target}.py")
        params = module.load_weights(str(manifest.weights_file))
        actual = np.asarray(module.forward(params, {"x": sample.detach().cpu().numpy()})["y"])
        actual_by_target[target] = actual
        assert np.allclose(actual, expected, rtol=rtol, atol=atol), (
            f"Starter model parity failed for {target}: "
            f"max_abs={np.max(np.abs(actual - expected))}, rtol={rtol}, atol={atol}"
        )

    assert np.allclose(actual_by_target["jax"], actual_by_target["mlx"], rtol=2e-5, atol=2e-6), (
        "Cross-backend parity failed between JAX and MLX for starter model: "
        f"max_abs={np.max(np.abs(actual_by_target['jax'] - actual_by_target['mlx']))}"
    )
