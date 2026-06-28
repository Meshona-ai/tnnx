from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _flux_entrypoint():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.source as source
    from examples.flux.transpile_and_generate_jax import main

    return main, source


def test_flux_jax_cli_accepts_prompt_npz(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    main, source = _flux_entrypoint()

    prompt_fixture = tmp_path / "prompt_fixture.npz"
    np.savez(
        prompt_fixture,
        prompt_embeddings=np.full(
            (1, source.DEMO_SEQUENCE, source.DEMO_DIM),
            0.25,
            dtype=np.float32,
        ),
        pooled_prompt=np.full((1, source.DEMO_DIM), -0.5, dtype=np.float32),
    )

    exit_code = main(
        [
            "--out",
            str(tmp_path),
            "--steps",
            "2",
            "--prompt-npz",
            str(prompt_fixture),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "flux_jax_demo.png").exists()


def test_flux_jax_cli_accepts_use_text_encoders(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    main, _ = _flux_entrypoint()

    exit_code = main(
        [
            "--out",
            str(tmp_path),
            "--steps",
            "2",
            "--use-text-encoders",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "flux_jax_demo.png").exists()


def test_flux_jax_cli_accepts_use_dummy_flux2(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    main, _ = _flux_entrypoint()

    exit_code = main(
        [
            "--out",
            str(tmp_path),
            "--steps",
            "2",
            "--use-dummy-flux2",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "flux_jax_dummy_flux2.png").exists()


def test_flux_jax_cli_accepts_use_tiny_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = pytest.importorskip("jax")
    main, _ = _flux_entrypoint()
    import examples.flux.transpile_and_generate_jax as demo

    captured: dict[str, object] = {}

    def _fake_tiny(out_root, *, model_id=None):
        captured["out_root"] = str(out_root)
        captured["model_id"] = model_id
        image_path = tmp_path / "flux_jax_tiny_config_e2e.png"
        ref_path = tmp_path / "flux_pytorch_tiny_config_e2e.png"
        image_path.write_bytes(b"png")
        ref_path.write_bytes(b"png")
        return {
            "text_encoder_onnx": str(tmp_path / "text_encoder.onnx"),
            "transformer_onnx": str(tmp_path / "transformer.onnx"),
            "vae_onnx": str(tmp_path / "vae.onnx"),
            "image_path": str(image_path),
            "pytorch_reference_image_path": str(ref_path),
            "prompt_source": "tiny_config_text_encoder",
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "pixel_std": 1.0,
        }

    monkeypatch.setattr(demo, "run_flux_jax_tiny_config_e2e_demo", _fake_tiny)

    exit_code = main(
        [
            "--out",
            str(tmp_path),
            "--use-tiny-config",
            "--model-id",
            "black-forest-labs/FLUX.2-klein-4B",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "out_root": str(tmp_path),
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
    }


def test_flux_jax_cli_accepts_use_tiny_config_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = pytest.importorskip("jax")
    main, _ = _flux_entrypoint()
    import examples.flux.transpile_and_generate_jax as demo

    captured: dict[str, object] = {}

    def _fake_tiny_torch(out_root, *, model_id=None):
        captured["out_root"] = str(out_root)
        captured["model_id"] = model_id
        image_path = tmp_path / "flux_pytorch_tiny_config_e2e.png"
        image_path.write_bytes(b"png")
        return {
            "image_path": str(image_path),
            "image_size": 16,
            "prompt_source": "tiny_config_text_encoder",
            "token_length": 8,
            "pixel_std": 1.0,
        }

    monkeypatch.setattr(demo, "run_flux_pytorch_tiny_config_e2e_demo", _fake_tiny_torch)

    exit_code = main(
        [
            "--out",
            str(tmp_path),
            "--use-tiny-config-torch",
            "--model-id",
            "black-forest-labs/FLUX.2-klein-4B",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "out_root": str(tmp_path),
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
    }


def test_flux_jax_cli_accepts_reduced_checkpoint_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = pytest.importorskip("jax")
    main, _ = _flux_entrypoint()
    import examples.flux.transpile_and_generate_jax as demo

    captured: dict[str, object] = {}

    def _fake_bridge(out_root, *, model_id=None):
        captured["out_root"] = str(out_root)
        captured["model_id"] = model_id
        image_path = tmp_path / "flux_jax_reduced_checkpoint_bridge.png"
        ref_path = tmp_path / "flux_pytorch_reduced_checkpoint_bridge.png"
        image_path.write_bytes(b"png")
        ref_path.write_bytes(b"png")
        return {
            "text_encoder_onnx": str(tmp_path / "text_encoder.onnx"),
            "transformer_onnx": str(tmp_path / "transformer.onnx"),
            "vae_onnx": str(tmp_path / "vae.onnx"),
            "image_path": str(image_path),
            "pytorch_reference_image_path": str(ref_path),
            "prompt_source": "reduced_checkpoint_bridge",
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "pixel_std": 1.0,
        }

    monkeypatch.setattr(demo, "run_flux_jax_reduced_checkpoint_bridge_demo", _fake_bridge)

    exit_code = main(
        [
            "--out",
            str(tmp_path),
            "--use-reduced-checkpoint-bridge",
            "--model-id",
            "black-forest-labs/FLUX.2-klein-4B",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "out_root": str(tmp_path),
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
    }


def test_flux_jax_cli_accepts_token_ids_npz(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    main, source = _flux_entrypoint()

    token_fixture = tmp_path / "token_fixture.npz"
    np.savez(
        token_fixture,
        input_ids=np.array([list(range(source.DEMO_SEQUENCE))], dtype=np.int64),
    )

    exit_code = main(
        [
            "--out",
            str(tmp_path),
            "--steps",
            "2",
            "--use-text-encoders",
            "--token-ids-npz",
            str(token_fixture),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "flux_jax_demo.png").exists()


def test_flux_jax_cli_rejects_prompt_npz_with_text_encoders(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    main, source = _flux_entrypoint()

    prompt_fixture = tmp_path / "prompt_fixture.npz"
    np.savez(
        prompt_fixture,
        prompt_embeddings=np.zeros((1, source.DEMO_SEQUENCE, source.DEMO_DIM), dtype=np.float32),
        pooled_prompt=np.zeros((1, source.DEMO_DIM), dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match="--prompt-npz cannot be combined with --use-text-encoders",
    ):
        main(
            [
                "--out",
                str(tmp_path),
                "--use-text-encoders",
                "--prompt-npz",
                str(prompt_fixture),
            ]
        )


def test_flux_jax_cli_rejects_token_ids_npz_without_text_encoders(tmp_path: Path) -> None:
    _ = pytest.importorskip("jax")
    main, source = _flux_entrypoint()

    token_fixture = tmp_path / "token_fixture.npz"
    np.savez(
        token_fixture,
        input_ids=np.array([list(range(source.DEMO_SEQUENCE))], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="--token-ids-npz requires --use-text-encoders"):
        main(
            [
                "--out",
                str(tmp_path),
                "--token-ids-npz",
                str(token_fixture),
            ]
        )


def test_flux_jax_cli_rejects_use_dummy_flux2_with_other_generation_flags(
    tmp_path: Path,
) -> None:
    _ = pytest.importorskip("jax")
    main, _ = _flux_entrypoint()

    with pytest.raises(
        ValueError,
        match="--use-dummy-flux2 cannot be combined with other prompt/image generation flags",
    ):
        main(
            [
                "--out",
                str(tmp_path),
                "--use-dummy-flux2",
                "--use-text-encoders",
            ]
        )


def test_flux_jax_cli_rejects_use_tiny_config_with_other_generation_flags(
    tmp_path: Path,
) -> None:
    _ = pytest.importorskip("jax")
    main, _ = _flux_entrypoint()

    with pytest.raises(
        ValueError,
        match="--use-tiny-config cannot be combined with other prompt/image generation flags",
    ):
        main(
            [
                "--out",
                str(tmp_path),
                "--use-tiny-config",
                "--use-text-encoders",
            ]
        )


def test_flux_jax_cli_rejects_use_tiny_config_torch_with_other_generation_flags(
    tmp_path: Path,
) -> None:
    _ = pytest.importorskip("jax")
    main, _ = _flux_entrypoint()

    with pytest.raises(
        ValueError,
        match=(
            "--use-tiny-config-torch cannot be combined with other prompt/image generation flags"
        ),
    ):
        main(
            [
                "--out",
                str(tmp_path),
                "--use-tiny-config-torch",
                "--use-tiny-config",
            ]
        )


def test_flux_jax_cli_rejects_reduced_checkpoint_bridge_with_other_generation_flags(
    tmp_path: Path,
) -> None:
    _ = pytest.importorskip("jax")
    main, _ = _flux_entrypoint()

    with pytest.raises(
        ValueError,
        match=(
            "--use-reduced-checkpoint-bridge cannot be combined with other "
            "prompt/image generation flags"
        ),
    ):
        main(
            [
                "--out",
                str(tmp_path),
                "--use-reduced-checkpoint-bridge",
                "--use-text-encoders",
            ]
        )
