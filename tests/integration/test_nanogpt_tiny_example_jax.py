from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest

from tnnx.api import transpile_onnx
from tnnx.config import CompileConfig
from tnnx.ingest.op_map import ONNX_TO_SEMANTIC


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("generated_nanogpt_jax_model", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nanogpt_tiny_jax_export_transpile_parity(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _ = torch
    _ = pytest.importorskip("jax")
    from examples.model_nanogpt_tiny import (
        NanoGPTTinyConfig,
        TinyNanoGPT,
        build_demo_tokens,
        export_onnx,
    )

    torch.manual_seed(7)
    np.random.seed(7)
    cfg = NanoGPTTinyConfig()
    model = TinyNanoGPT(cfg).eval()
    tokens = build_demo_tokens(cfg, seed=7)
    onnx_path = export_onnx(tmp_path / "nanogpt_tiny.onnx", model=model, sample_tokens=tokens)

    exported = onnx.load(onnx_path)
    exported_ops = {node.op_type for node in exported.graph.node}
    unsupported = sorted(exported_ops - (set(ONNX_TO_SEMANTIC.keys()) | {"Constant"}))
    assert not unsupported, f"NanoGPT tiny export has unsupported ops: {unsupported}"

    out_dir = tmp_path / "generated_nanogpt_tiny_jax"
    manifest = transpile_onnx(
        str(onnx_path),
        "jax",
        str(out_dir),
        config=CompileConfig(entrypoint="forward", weights_filename="weights.npz"),
    )
    mod = _load_module(out_dir / "model_jax.py")
    params = mod.load_weights(str(manifest.weights_file))

    expected = model(tokens).detach().cpu().numpy()
    actual = np.asarray(mod.forward(params, {"idx": tokens.detach().cpu().numpy()})["y"])
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_nanogpt_tiny_demo_script_runner(tmp_path: Path) -> None:
    _ = pytest.importorskip("torch")
    _ = pytest.importorskip("jax")
    from examples.run_nanogpt_tiny_jax import run_demo

    result = run_demo(tmp_path)
    assert str(result["generated_module"]).endswith("model_jax.py")
    assert float(result["max_abs"]) <= 1e-5
