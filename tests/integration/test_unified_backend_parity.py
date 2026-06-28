from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig


@dataclass(frozen=True, slots=True)
class ParityCase:
    name: str
    input_shape: tuple[int, ...]


NUMERIC_CASES: list[tuple[str, ParityCase]] = [
    ("jax", ParityCase(name="mlp", input_shape=(2, 8))),
    ("mlx", ParityCase(name="mlp", input_shape=(2, 8))),
    ("jax", ParityCase(name="conv", input_shape=(1, 3, 16, 16))),
    ("mlx", ParityCase(name="conv", input_shape=(1, 3, 16, 16))),
    ("jax", ParityCase(name="residual", input_shape=(2, 12))),
    ("mlx", ParityCase(name="residual", input_shape=(2, 12))),
    ("jax", ParityCase(name="indexing", input_shape=(3, 6))),
    ("mlx", ParityCase(name="indexing", input_shape=(3, 6))),
    ("jax", ParityCase(name="conv1d", input_shape=(1, 80, 32))),
    ("mlx", ParityCase(name="conv1d", input_shape=(1, 80, 32))),
]


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("generated_backend_model", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rtol_atol(target: str, dtype: np.dtype[Any]) -> tuple[float, float]:
    # Float32 parity defaults: tighter for JAX, slightly relaxed for MLX.
    if dtype == np.float32:
        if target == "jax":
            return 1e-5, 1e-6
        if target == "mlx":
            return 2e-5, 2e-6
    return 1e-6, 1e-7


def _model_for_case(case: ParityCase, torch: Any) -> Any:
    nn = torch.nn

    if case.name == "mlp":

        class TinyMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(8, 16)
                self.fc2 = nn.Linear(16, 4)

            def forward(self, x: Any) -> Any:
                return self.fc2(torch.relu(self.fc1(x)))

        return TinyMLP().eval()

    if case.name == "conv":

        class TinyConv(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
                self.fc = nn.Linear(8 * 16 * 16, 10)

            def forward(self, x: Any) -> Any:
                y = torch.relu(self.conv(x))
                y = y.reshape(y.shape[0], -1)
                return self.fc(y)

        return TinyConv().eval()

    if case.name == "residual":

        class ResidualMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(12, 12)
                self.ff1 = nn.Linear(12, 24)
                self.ff2 = nn.Linear(24, 12)

            def forward(self, x: Any) -> Any:
                skip = self.proj(x)
                y = self.ff2(torch.relu(self.ff1(skip)))
                return y + skip

        return ResidualMLP().eval()

    if case.name == "indexing":

        class IndexingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("idx", torch.tensor([0, 2, 4], dtype=torch.long))

            def forward(self, x: Any) -> Any:
                gathered = x[:, self.idx]
                return gathered[:, 1:3]

        return IndexingModel().eval()

    if case.name == "conv1d":

        class TinyConv1D(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv1d(80, 96, kernel_size=3, stride=2, padding=1)
                self.out = nn.Conv1d(96, 64, kernel_size=1)

            def forward(self, x: Any) -> Any:
                return torch.relu(self.out(torch.relu(self.conv(x))))

        return TinyConv1D().eval()

    raise ValueError(f"Unknown case: {case.name}")


def _require_backend_runtime(target: str) -> None:
    if target == "jax":
        _ = pytest.importorskip("jax")
    elif target == "mlx":
        _ = pytest.importorskip("mlx.core")
    else:
        raise ValueError(f"Unexpected target in numeric parity: {target}")


def _export_to_onnx(model: Any, sample: Any, out_path: Path, torch: Any) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        out_path,
        input_names=["x"],
        output_names=["y"],
        opset_version=18,
        dynamo=False,
    )
    return out_path


@pytest.mark.parametrize(("target", "case"), NUMERIC_CASES)
def test_unified_numeric_parity(target: str, case: ParityCase, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _require_backend_runtime(target)

    torch.manual_seed(7)
    np.random.seed(7)

    model = _model_for_case(case, torch)
    sample = torch.randn(*case.input_shape, dtype=torch.float32)
    with torch.no_grad():
        expected = model(sample).detach().cpu().numpy()

    onnx_path = _export_to_onnx(model, sample, tmp_path / f"{case.name}.onnx", torch)
    out_dir = tmp_path / f"generated_{target}_{case.name}"
    manifest = transpile_onnx(
        str(onnx_path),
        target,
        str(out_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )

    module_path = out_dir / f"model_{target}.py"
    module = _load_module(module_path)
    params = module.load_weights(str(manifest.weights_file))
    actual = module.forward(params, {"x": sample.detach().cpu().numpy()})["y"]
    actual_np = np.asarray(actual)

    rtol, atol = _rtol_atol(target, expected.dtype)
    assert np.allclose(actual_np, expected, rtol=rtol, atol=atol), (
        f"Parity failed for {target}/{case.name}: "
        f"max_abs={np.max(np.abs(actual_np - expected))}, rtol={rtol}, atol={atol}"
    )
