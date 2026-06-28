from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _flux_demo_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.transpile_and_generate_jax as demo

    return demo


def test_flux_checkpoint_asset_inspection_reads_snapshot_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    snapshot = tmp_path / "snapshot"
    transformer_dir = snapshot / "transformer"
    vae_dir = snapshot / "vae"
    text_dir = snapshot / "text_encoder"
    transformer_dir.mkdir(parents=True)
    vae_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)

    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2Transformer2DModel",
                "in_channels": 16,
                "joint_attention_dim": 32,
                "axes_dims_rope": [4, 4, 4],
            }
        ),
        encoding="utf-8",
    )
    (transformer_dir / "diffusion_pytorch_model.safetensors").write_text(
        "weights",
        encoding="utf-8",
    )
    (vae_dir / "config.json").write_text(
        json.dumps({"_class_name": "AutoencoderKLFlux2", "latent_channels": 16}),
        encoding="utf-8",
    )
    (text_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "Qwen3ForCausalLM",
                "model_type": "qwen3",
                "hidden_size": 64,
                "max_position_embeddings": 128,
            }
        ),
        encoding="utf-8",
    )
    (text_dir / "pytorch_model.bin").write_text("weights", encoding="utf-8")

    monkeypatch.setattr(demo.flux_source, "resolve_flux_snapshot", lambda model_id=None: snapshot)

    report = demo.inspect_flux_checkpoint_assets(tmp_path / "out")

    assert report["ready_count"] == 2
    assert report["missing_count"] == 1
    assert report["submodules"]["transformer"]["status"] == "ready"
    assert report["submodules"]["transformer"]["metadata"] == {
        "_class_name": "Flux2Transformer2DModel",
        "in_channels": 16,
        "joint_attention_dim": 32,
        "axes_dims_rope": [4, 4, 4],
    }
    assert report["submodules"]["vae_decoder"]["status"] == "missing"
    assert report["submodules"]["vae_decoder"]["has_config"] is True
    assert report["submodules"]["vae_decoder"]["has_weights"] is False
    assert report["submodules"]["text_encoder"]["status"] == "ready"
    assert report["submodules"]["text_encoder"]["metadata"] == {
        "_class_name": "Qwen3ForCausalLM",
        "model_type": "qwen3",
        "hidden_size": 64,
        "max_position_embeddings": 128,
    }


def test_flux_checkpoint_asset_inspection_cli_mode_uses_report_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()

    def _fake_inspect(out_root, *, model_id=None):
        report_path = Path(out_root) / "flux_checkpoint_asset_report.json"
        return {
            "model_id": model_id,
            "snapshot": "/tmp/fake-snapshot",
            "report_path": str(report_path),
            "ready_count": 2,
            "missing_count": 1,
            "error": None,
            "submodule_order": ["vae_decoder", "transformer", "text_encoder"],
            "submodules": {
                "vae_decoder": {"status": "ready", "has_config": True, "has_weights": True},
                "transformer": {"status": "ready", "has_config": True, "has_weights": True},
                "text_encoder": {"status": "missing", "has_config": True, "has_weights": False},
            },
        }

    monkeypatch.setattr(demo, "inspect_flux_checkpoint_assets", _fake_inspect)

    exit_code = demo.main(["--out", str(tmp_path), "--inspect-checkpoint-assets"])

    assert exit_code == 0
