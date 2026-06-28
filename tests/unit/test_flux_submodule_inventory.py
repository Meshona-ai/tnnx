from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _flux_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux as flux

    return flux


def _flux_source_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.source as source

    return source


def test_flux_model_candidates_cover_documented_4b_targets() -> None:
    flux = _flux_module()

    assert flux.DEFAULT_FLUX_MODEL_ID == "black-forest-labs/FLUX.2-klein-4b-fp8"
    assert flux.DEFAULT_REAL_FLUX_MODEL_ID == "black-forest-labs/FLUX.2-klein-4B"
    assert flux.FLUX_MODEL_CANDIDATES == (
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-4b-fp8",
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-base-4b-nvfp4",
    )


def test_flux_submodule_order_matches_staged_rollout() -> None:
    flux = _flux_module()

    assert flux.SUBMODULE_EXPORT_ORDER == (
        "vae_decoder",
        "transformer",
        "text_encoder",
    )


def test_flux_real_submodule_order_is_flux2_only(tmp_path: Path) -> None:
    source = _flux_source_module()

    flux2_snapshot = tmp_path / "flux2"
    flux2_snapshot.mkdir()

    assert source.real_submodule_export_order(snapshot=flux2_snapshot) == (
        "vae_decoder",
        "transformer",
        "text_encoder",
    )


def test_flux_required_models_doc_tracks_model_constants() -> None:
    flux = _flux_module()
    doc = Path("examples/flux/required_models.md").read_text(encoding="utf-8")
    for model_id in flux.FLUX_MODEL_CANDIDATES:
        assert model_id in doc


def test_flux_snapshot_setup_doc_tracks_primary_target_and_env_contract() -> None:
    flux = _flux_module()
    doc = Path("examples/flux/snapshot_setup.md").read_text(encoding="utf-8")

    assert flux.DEFAULT_FLUX_MODEL_ID in doc
    assert flux.DEFAULT_REAL_FLUX_MODEL_ID in doc
    assert "TNNX_FLUX_SNAPSHOT" in doc
    assert "--inspect-checkpoint-assets" in doc
    assert "--prepare-checkpoint-artifacts" in doc


def test_flux_readme_tracks_tiny_config_workflow() -> None:
    doc = Path("examples/flux/README.md").read_text(encoding="utf-8")

    assert "run_flux_pytorch_tiny_config_e2e_demo" in doc
    assert "run_flux_jax_tiny_config_e2e_demo" in doc
    assert "--use-tiny-config" in doc
    assert "--use-tiny-config-torch" in doc


def test_flux_public_example_entrypoints_remain_exported() -> None:
    flux = _flux_module()

    assert callable(flux.run_flux_jax_demo)
    assert callable(flux.run_flux_pytorch_prompt_to_image_demo)
    assert callable(flux.run_flux_pytorch_dummy_flux2_demo)
    assert callable(flux.run_flux_jax_prompt_to_image_demo)
    assert callable(flux.run_flux_jax_dummy_flux2_demo)
    assert callable(flux.run_flux_jax_reduced_checkpoint_bridge_demo)
    assert callable(flux.run_flux_pytorch_tiny_config_e2e_demo)
    assert callable(flux.run_flux_jax_tiny_config_e2e_demo)
    assert callable(flux.prepare_flux_jax_checkpoint_artifacts)


def test_flux_dummy_flux2_specs_track_single_encoder_topology() -> None:
    source = _flux_source_module()
    specs = source.build_dummy_flux2_export_specs()

    assert set(specs) == {"vae_decoder", "transformer", "text_encoder"}
    assert specs["transformer"].sample_inputs[0].shape == (
        1,
        source.DUMMY_FLUX2_SEQUENCE,
        source.DUMMY_FLUX2_DIM,
    )
    assert specs["text_encoder"].sample_inputs[0].shape == (
        1,
        source.DUMMY_FLUX2_TOKEN_SEQ_LEN,
    )


def test_flux_onnx_inventory_helpers_track_supported_demo_graph(tmp_path: Path) -> None:
    source = _flux_source_module()
    spec = source.build_demo_export_specs()["vae_decoder"]
    onnx_path = source.export_flux_submodule_onnx(
        "vae_decoder",
        tmp_path / "flux_vae_decoder.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
    )

    inventory = source.load_onnx_op_inventory(onnx_path)
    unsupported = source.load_unsupported_onnx_ops(onnx_path)

    assert inventory
    assert unsupported == ()


def test_flux_text_encoder_export_uses_dynamo(tmp_path: Path, monkeypatch) -> None:
    source = _flux_source_module()
    spec = source.build_demo_export_specs()["text_encoder"]
    captured: dict[str, object] = {}

    def _fake_export(module: Any, args: Any, target: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        Path(target).write_text("fake", encoding="utf-8")

    monkeypatch.setattr(source.torch.onnx, "export", _fake_export)

    onnx_path = source.export_flux_submodule_onnx(
        "text_encoder",
        tmp_path / "flux_text_encoder.onnx",
        module=spec.module,
        sample_inputs=spec.sample_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
    )

    assert onnx_path.exists()
    assert captured["dynamo"] is True
