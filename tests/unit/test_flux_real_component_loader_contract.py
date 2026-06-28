from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def _flux_source_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.source as source

    return source


def test_flux_real_component_loader_uses_low_memory_transformer_load(monkeypatch) -> None:
    source = _flux_source_module()
    snapshot = Path("/tmp/fake-flux-snapshot")
    captured: dict[str, object] = {}

    class _FakeVae:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("VAE loader should not be called in this test.")

    class _FakeTransformer:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            captured["path"] = path
            captured["kwargs"] = dict(kwargs)
            return "fake-transformer"

    monkeypatch.setattr(source, "resolve_flux_snapshot", lambda model_id=None: snapshot)
    monkeypatch.setattr(
        source,
        "_require_diffusers_flux2_components",
        lambda: (_FakeVae, _FakeTransformer),
    )
    monkeypatch.setattr(
        source,
        "_require_transformers",
        lambda: SimpleNamespace(AutoModel=None),
    )

    result = source._load_real_component("transformer")

    assert result == "fake-transformer"
    assert captured["path"] == str(snapshot / "transformer")
    assert captured["kwargs"] == {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }


def test_flux_reduced_transformer_config_shrinks_large_dims() -> None:
    source = _flux_source_module()

    reduced = source._reduced_transformer_config(
        {
            "patch_size": 2,
            "in_channels": 128,
            "out_channels": 256,
            "num_layers": 8,
            "num_single_layers": 48,
            "attention_head_dim": 128,
            "num_attention_heads": 48,
            "joint_attention_dim": 7680,
            "timestep_guidance_channels": 256,
            "pooled_projection_dim": 4096,
            "pooled_projection_embed_dim": 3072,
            "axes_dims_rope": [32, 32, 32, 32],
        }
    )

    assert reduced["patch_size"] == 1
    assert reduced["in_channels"] == 32
    assert reduced["out_channels"] == 32
    assert reduced["num_layers"] == 1
    assert reduced["num_single_layers"] == 1
    assert reduced["attention_head_dim"] == 16
    assert reduced["num_attention_heads"] == 4
    assert reduced["joint_attention_dim"] == 64
    assert reduced["timestep_guidance_channels"] == 16
    assert reduced["pooled_projection_dim"] == 64
    assert reduced["pooled_projection_embed_dim"] == 64
    assert reduced["axes_dims_rope"] == (4, 4, 4, 4)


def test_flux_reduced_text_encoder_config_shrinks_large_dims() -> None:
    source = _flux_source_module()

    reduced = source._reduced_text_encoder_config(
        {
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 80,
            "max_position_embeddings": 40960,
            "vocab_size": 151936,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_window_layers": 28,
            "sliding_window": 256,
            "rope_scaling": {"type": "yarn", "original_max_position_embeddings": 40960},
        }
    )

    assert reduced["hidden_size"] == 64
    assert reduced["intermediate_size"] == 256
    assert reduced["num_hidden_layers"] == 1
    assert reduced["num_attention_heads"] == 4
    assert reduced["num_key_value_heads"] == 4
    assert reduced["head_dim"] == 16
    assert reduced["max_position_embeddings"] == 64
    assert reduced["vocab_size"] == 512
    assert reduced["bos_token_id"] == 511
    assert reduced["eos_token_id"] == 511
    assert reduced["layer_types"] == ["sliding_attention"]
    assert reduced["max_window_layers"] == 1
    assert reduced["sliding_window"] == 64
    assert reduced["rope_scaling"] == {
        "type": "yarn",
        "original_max_position_embeddings": 64,
    }


def test_flux_tiny_transformer_config_shrinks_to_two_head_profile() -> None:
    source = _flux_source_module()

    reduced = source._tiny_transformer_config(
        {
            "patch_size": 2,
            "in_channels": 128,
            "out_channels": 256,
            "num_layers": 8,
            "num_single_layers": 48,
            "attention_head_dim": 128,
            "num_attention_heads": 24,
            "joint_attention_dim": 7680,
            "timestep_guidance_channels": 256,
            "pooled_projection_dim": 4096,
            "axes_dims_rope": [32, 32, 32, 32],
        }
    )

    assert reduced["patch_size"] == 1
    assert reduced["in_channels"] == 8
    assert reduced["out_channels"] == 8
    assert reduced["num_layers"] == 2
    assert reduced["num_single_layers"] == 2
    assert reduced["attention_head_dim"] == 8
    assert reduced["num_attention_heads"] == 2
    assert reduced["joint_attention_dim"] == 32
    assert reduced["timestep_guidance_channels"] == 8
    assert reduced["pooled_projection_dim"] == 32
    assert reduced["axes_dims_rope"] == (2, 2, 2, 2)


def test_flux_tiny_text_encoder_config_shrinks_to_two_head_profile() -> None:
    source = _flux_source_module()

    reduced = source._tiny_text_encoder_config(
        {
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 40960,
            "vocab_size": 151936,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_window_layers": 28,
            "sliding_window": 256,
            "rope_scaling": {"type": "yarn", "original_max_position_embeddings": 40960},
        },
        target_hidden_size=32,
    )

    assert reduced["hidden_size"] == 32
    assert reduced["intermediate_size"] == 64
    assert reduced["num_hidden_layers"] == 2
    assert reduced["num_attention_heads"] == 2
    assert reduced["num_key_value_heads"] == 2
    assert reduced["head_dim"] == 16
    assert reduced["max_position_embeddings"] == 32
    assert reduced["vocab_size"] == 128
    assert reduced["bos_token_id"] == 127
    assert reduced["eos_token_id"] == 127
    assert reduced["layer_types"] == ["sliding_attention", "full_attention"]
    assert reduced["max_window_layers"] == 2
    assert reduced["sliding_window"] == 32
    assert reduced["rope_scaling"] == {
        "type": "yarn",
        "original_max_position_embeddings": 32,
    }


def test_flux_tiny_vae_config_shrinks_decoder_profile() -> None:
    source = _flux_source_module()

    reduced = source._tiny_vae_config(
        {
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": ["a", "b", "c", "d"],
            "layers_per_block": 2,
            "latent_channels": 32,
            "sample_size": 1024,
            "patch_size": 2,
            "norm_num_groups": 32,
        }
    )

    assert reduced["block_out_channels"] == (8, 8, 16, 16)
    assert reduced["layers_per_block"] == 1
    assert reduced["latent_channels"] == 8
    assert reduced["sample_size"] == 16
    assert reduced["patch_size"] == [1, 1]
    assert reduced["norm_num_groups"] == 8


def test_flux_real_component_loader_can_use_config_only_transformer_build(monkeypatch) -> None:
    source = _flux_source_module()
    snapshot = Path("/tmp/fake-flux-snapshot")
    captured: dict[str, object] = {}

    class _FakeVae:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("VAE loader should not be called in this test.")

    class _FakeTransformer:
        @classmethod
        def load_config(cls, path: str, **kwargs):
            captured["config_path"] = path
            captured["config_kwargs"] = dict(kwargs)
            return {
                "patch_size": 1,
                "in_channels": 128,
                "out_channels": 128,
                "num_layers": 8,
                "num_single_layers": 48,
                "attention_head_dim": 128,
                "num_attention_heads": 48,
                "joint_attention_dim": 15360,
                "timestep_guidance_channels": 256,
                "pooled_projection_dim": 4096,
                "pooled_projection_embed_dim": 3072,
                "axes_dims_rope": [32, 32, 32, 32],
            }

        @classmethod
        def from_config(cls, config):
            captured["config"] = dict(config)
            return "config-only-transformer"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("Weighted transformer load should not be used in graph-only mode.")

    monkeypatch.setattr(source, "resolve_flux_snapshot", lambda model_id=None: snapshot)
    monkeypatch.setattr(
        source,
        "_require_diffusers_flux2_components",
        lambda: (_FakeVae, _FakeTransformer),
    )
    monkeypatch.setattr(
        source,
        "_require_transformers",
        lambda: SimpleNamespace(AutoModel=None),
    )

    result = source._load_real_component("transformer", load_weights=False)

    assert result == "config-only-transformer"
    assert captured["config_path"] == str(snapshot / "transformer")
    assert captured["config_kwargs"] == {"local_files_only": True}
    assert captured["config"] == {
        "patch_size": 1,
        "in_channels": 32,
        "out_channels": 32,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 4,
        "joint_attention_dim": 64,
        "timestep_guidance_channels": 16,
        "pooled_projection_dim": 64,
        "pooled_projection_embed_dim": 64,
        "axes_dims_rope": (4, 4, 4, 4),
    }


def test_flux_real_component_loader_can_use_config_only_text_encoder_build(monkeypatch) -> None:
    source = _flux_source_module()
    snapshot = Path("/tmp/fake-flux-snapshot")
    captured: dict[str, object] = {}

    class _FakeConfig:
        def __init__(self, **kwargs):
            self._payload = dict(kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to_dict(self):
            return dict(self._payload)

    class _FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            captured["config_path"] = path
            captured["config_kwargs"] = dict(kwargs)
            return _FakeConfig(
                hidden_size=2560,
                intermediate_size=9728,
                num_hidden_layers=36,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=80,
                max_position_embeddings=40960,
                vocab_size=151936,
                layer_types=["sliding_attention", "full_attention"],
            )

    class _FakeAutoModel:
        @classmethod
        def from_config(cls, config):
            captured["config"] = config.to_dict()
            return "config-only-text-encoder"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError(
                "Weighted text-encoder load should not be used in graph-only mode."
            )

    class _FakeVae:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("VAE loader should not be called in this test.")

    class _FakeTransformer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("Transformer loader should not be called in this test.")

    monkeypatch.setattr(source, "resolve_flux_snapshot", lambda model_id=None: snapshot)
    monkeypatch.setattr(
        source,
        "_require_diffusers_flux2_components",
        lambda: (_FakeVae, _FakeTransformer),
    )
    monkeypatch.setattr(
        source,
        "_require_transformers",
        lambda: SimpleNamespace(AutoConfig=_FakeAutoConfig, AutoModel=_FakeAutoModel),
    )

    result = source._load_real_component("text_encoder", load_weights=False)

    assert result == "config-only-text-encoder"
    assert captured["config_path"] == str(snapshot / "text_encoder")
    assert captured["config_kwargs"] == {"local_files_only": True}
    assert captured["config"] == {
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "max_position_embeddings": 64,
        "vocab_size": 512,
        "layer_types": ["sliding_attention"],
    }
