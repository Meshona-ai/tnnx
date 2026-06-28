from __future__ import annotations

import sys
from pathlib import Path

import onnx
import pytest


def _flux_source():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.source as source

    return source


def test_flux_transformer_input_builder_builds_inputs_for_tiny_transformer(
    tmp_path: Path,
) -> None:
    _ = pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

    source = _flux_source()
    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        guidance_embeds=True,
        axes_dims_rope=(2, 2, 4),
    ).eval()

    sample_map = source.build_flux_transformer_sample_inputs(
        transformer,
        image_seq_len=4,
        text_seq_len=6,
    )
    wrapper = source.FluxTransformerForwardWrapper(
        transformer,
        include_pooled_projections=True,
        include_guidance=True,
    ).eval()
    output = wrapper(
        sample_map["hidden_states"],
        sample_map["encoder_hidden_states"],
        sample_map["timestep"],
        sample_map["img_ids"],
        sample_map["txt_ids"],
        sample_map["guidance"],
        sample_map["pooled_projections"],
    )

    assert output.shape == (1, 4, 4)
    assert tuple(sample_map.keys()) == (
        "hidden_states",
        "encoder_hidden_states",
        "timestep",
        "img_ids",
        "txt_ids",
        "pooled_projections",
        "guidance",
    )

    onnx_path = source.export_flux_submodule_onnx(
        "transformer",
        tmp_path / "tiny_flux_transformer.onnx",
        module=wrapper,
        sample_inputs=(
            sample_map["hidden_states"],
            sample_map["encoder_hidden_states"],
            sample_map["timestep"],
            sample_map["img_ids"],
            sample_map["txt_ids"],
            sample_map["guidance"],
            sample_map["pooled_projections"],
        ),
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
            "pooled_projections",
        ),
        output_names=("denoised",),
    )
    exported = onnx.load(onnx_path)

    assert onnx_path.exists()
    assert len(exported.graph.node) > 0


def test_flux_transformer_input_builder_supports_flux2_signature() -> None:
    _ = pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    source = _flux_source()
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2),
    ).eval()

    sample_map = source.build_flux_transformer_sample_inputs(
        transformer,
        image_seq_len=4,
        text_seq_len=6,
    )
    wrapper = source.FluxTransformerForwardWrapper(
        transformer,
        include_pooled_projections=False,
        include_guidance=True,
    ).eval()
    output = wrapper(
        sample_map["hidden_states"],
        sample_map["encoder_hidden_states"],
        sample_map["timestep"],
        sample_map["img_ids"],
        sample_map["txt_ids"],
        guidance=sample_map["guidance"],
    )

    assert output.shape == (1, 4, 4)
    assert tuple(sample_map.keys()) == (
        "hidden_states",
        "encoder_hidden_states",
        "timestep",
        "img_ids",
        "txt_ids",
        "guidance",
    )
    assert "pooled_projections" not in sample_map


def test_flux_transformer_graph_only_export_skips_weight_sidecar(tmp_path: Path) -> None:
    _ = pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    source = _flux_source()
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2),
    ).eval()
    sample_map = source.build_flux_transformer_sample_inputs(
        transformer,
        image_seq_len=source.CHECKPOINT_SMOKE_IMAGE_SEQ_LEN,
        text_seq_len=source.CHECKPOINT_SMOKE_TEXT_SEQ_LEN,
    )
    wrapper = source.FluxTransformerForwardWrapper(
        transformer,
        include_pooled_projections=False,
        include_guidance=True,
    ).eval()

    onnx_path = source.export_flux_submodule_onnx(
        "transformer",
        tmp_path / "tiny_flux_transformer_graph_only.onnx",
        module=wrapper,
        sample_inputs=(
            sample_map["hidden_states"],
            sample_map["encoder_hidden_states"],
            sample_map["timestep"],
            sample_map["img_ids"],
            sample_map["txt_ids"],
            sample_map["guidance"],
        ),
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
        ),
        output_names=("denoised",),
        export_params=False,
    )

    assert onnx_path.exists()
    assert not (tmp_path / "tiny_flux_transformer_graph_only.onnx.data").exists()
