from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest


def _flux_demo_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.transpile_and_generate_jax as demo

    return demo


def test_flux_checkpoint_prepare_report_tracks_ready_blocked_and_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    demo_specs = demo.build_demo_export_specs()

    monkeypatch.setattr(demo.flux_source, "resolve_flux_snapshot", lambda model_id=None: snapshot)
    monkeypatch.setattr(
        demo.flux_source,
        "snapshot_has_submodule_weights",
        lambda resolved_snapshot, submodule: submodule != "vae_decoder",
    )
    monkeypatch.setattr(
        demo.flux_source,
        "_real_export_spec",
        lambda submodule, model_id=None, **kwargs: demo_specs[submodule],
    )

    def _fake_export(submodule, out_path, **_kwargs):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fake-{submodule}", encoding="utf-8")
        return path

    monkeypatch.setattr(demo.flux_source, "export_flux_submodule_onnx", _fake_export)
    monkeypatch.setattr(
        demo.flux_source,
        "load_unsupported_onnx_ops",
        lambda onnx_path: ("FakeUnsupported",) if "transformer" in str(onnx_path) else (),
    )

    def _fake_transpile(onnx_path, target, out_dir, *, config):
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_file = output_dir / config.weights_filename
        weights_file.write_text("weights", encoding="utf-8")
        return SimpleNamespace(target=target, weights_file=weights_file)

    monkeypatch.setattr(demo, "transpile_onnx", _fake_transpile)
    monkeypatch.setattr(
        demo,
        "run_export_spec_pytorch_reference",
        lambda spec: {"output_shapes": [list(spec.sample_inputs[0].shape)]},
    )

    report = demo.prepare_flux_jax_checkpoint_artifacts(tmp_path / "out")

    assert report["submodule_order"] == [
        "vae_decoder",
        "transformer",
        "text_encoder",
    ]
    assert report["ready_count"] == 1
    assert report["blocked_count"] == 1
    assert report["missing_count"] == 1
    assert report["submodules"]["transformer"]["status"] == "blocked"
    assert report["submodules"]["transformer"]["unsupported_ops"] == ["FakeUnsupported"]
    assert report["submodules"]["transformer"]["used_reduced_config"] is False
    assert report["submodules"]["transformer"]["reference_output_shapes"] == [
        [1, demo.flux_source.DEMO_SEQUENCE, demo.flux_source.DEMO_DIM]
    ]
    assert report["submodules"]["vae_decoder"]["status"] == "missing"
    assert report["submodules"]["vae_decoder"]["used_reduced_config"] is False
    assert report["submodules"]["vae_decoder"]["reference_output_shapes"] == []
    assert report["submodules"]["text_encoder"]["status"] == "ready"
    assert report["submodules"]["text_encoder"]["used_reduced_config"] is False
    assert report["submodules"]["text_encoder"]["reference_output_shapes"] == [
        [1, demo.flux_source.DEMO_SEQUENCE]
    ]
    assert "text_encoder_2" not in report["submodules"]

    report_path = Path(str(report["report_path"]))
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["submodule_order"] == [
        "vae_decoder",
        "transformer",
        "text_encoder",
    ]
    assert payload["ready_count"] == 1
    assert payload["blocked_count"] == 1
    assert payload["missing_count"] == 1


def test_flux_checkpoint_prepare_report_filters_single_submodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    demo_specs = demo.build_demo_export_specs()

    monkeypatch.setattr(demo.flux_source, "resolve_flux_snapshot", lambda model_id=None: snapshot)
    monkeypatch.setattr(
        demo.flux_source,
        "snapshot_has_submodule_weights",
        lambda resolved_snapshot, submodule: True,
    )
    monkeypatch.setattr(
        demo.flux_source,
        "_real_export_spec",
        lambda submodule, model_id=None, **kwargs: demo_specs[submodule],
    )

    def _fake_export(submodule, out_path, **_kwargs):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fake-{submodule}", encoding="utf-8")
        return path

    monkeypatch.setattr(demo.flux_source, "export_flux_submodule_onnx", _fake_export)
    monkeypatch.setattr(demo.flux_source, "load_unsupported_onnx_ops", lambda onnx_path: ())

    def _fake_transpile(onnx_path, target, out_dir, *, config):
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_file = output_dir / config.weights_filename
        weights_file.write_text("weights", encoding="utf-8")
        return SimpleNamespace(target=target, weights_file=weights_file)

    monkeypatch.setattr(demo, "transpile_onnx", _fake_transpile)
    monkeypatch.setattr(
        demo,
        "run_export_spec_pytorch_reference",
        lambda spec: {"output_shapes": [list(spec.sample_inputs[0].shape)]},
    )

    report = demo.prepare_flux_jax_checkpoint_artifacts(
        tmp_path / "out",
        checkpoint_submodule="transformer",
    )

    assert report["submodule_order"] == ["transformer"]
    assert report["ready_count"] == 1
    assert report["blocked_count"] == 0
    assert report["missing_count"] == 0
    assert list(report["submodules"]) == ["transformer"]
    assert report["submodules"]["transformer"]["status"] == "ready"
    assert report["submodules"]["transformer"]["used_reduced_config"] is False


def test_flux_checkpoint_prepare_report_uses_reduced_configs_in_reduced_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    demo_specs = demo.build_demo_export_specs()
    captured_kwargs: dict[str, dict[str, object]] = {}

    monkeypatch.setattr(demo.flux_source, "resolve_flux_snapshot", lambda model_id=None: snapshot)
    monkeypatch.setattr(
        demo.flux_source,
        "snapshot_has_submodule_weights",
        lambda resolved_snapshot, submodule: True,
    )

    def _fake_real_export_spec(submodule, model_id=None, **kwargs):
        captured_kwargs[submodule] = dict(kwargs)
        return demo_specs[submodule]

    monkeypatch.setattr(demo.flux_source, "_real_export_spec", _fake_real_export_spec)

    def _fake_export(submodule, out_path, **_kwargs):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fake-{submodule}", encoding="utf-8")
        return path

    monkeypatch.setattr(demo.flux_source, "export_flux_submodule_onnx", _fake_export)
    monkeypatch.setattr(demo.flux_source, "load_unsupported_onnx_ops", lambda onnx_path: ())

    def _fake_transpile(onnx_path, target, out_dir, *, config):
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_file = output_dir / config.weights_filename
        weights_file.write_text("weights", encoding="utf-8")
        return SimpleNamespace(target=target, weights_file=weights_file)

    monkeypatch.setattr(demo, "transpile_onnx", _fake_transpile)
    monkeypatch.setattr(
        demo,
        "run_export_spec_pytorch_reference",
        lambda spec: {"output_shapes": [list(spec.sample_inputs[0].shape)]},
    )

    report = demo.prepare_flux_jax_checkpoint_artifacts(
        tmp_path / "out",
        checkpoint_reduced_shapes=True,
    )

    assert report["ready_count"] == 3
    assert report["blocked_count"] == 0
    assert report["missing_count"] == 0
    assert report["submodules"]["transformer"]["used_reduced_config"] is True
    assert report["submodules"]["text_encoder"]["used_reduced_config"] is True
    assert report["submodules"]["vae_decoder"]["used_reduced_config"] is False
    assert captured_kwargs["transformer"]["load_weights"] is False
    assert captured_kwargs["transformer"]["transformer_image_seq_len"] == (
        demo.flux_source.CHECKPOINT_SMOKE_IMAGE_SEQ_LEN
    )
    assert captured_kwargs["transformer"]["transformer_text_seq_len"] == (
        demo.flux_source.CHECKPOINT_SMOKE_TEXT_SEQ_LEN
    )
    assert captured_kwargs["text_encoder"]["load_weights"] is False
    assert captured_kwargs["vae_decoder"] == {}


def test_flux_pytorch_reference_shapes_support_bfloat16_outputs() -> None:
    demo = _flux_demo_module()
    torch = pytest.importorskip("torch")

    class _BFloat16Module(torch.nn.Module):
        def forward(self, token_ids: object) -> object:
            return torch.ones((1, 2, 3), dtype=torch.bfloat16)

    spec = demo.ExportSpec(
        module=_BFloat16Module().eval(),
        sample_inputs=(torch.zeros((1, 2), dtype=torch.int64),),
        input_names=("input_ids",),
        output_names=("hidden",),
    )

    reference = demo.run_export_spec_pytorch_reference(spec)

    assert reference["output_shapes"] == [[1, 2, 3]]


def test_flux_checkpoint_prepare_report_cli_mode_uses_report_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()

    def _fake_prepare(
        out_root,
        *,
        model_id=None,
        checkpoint_submodule=None,
        checkpoint_graph_only=False,
        checkpoint_reduced_shapes=False,
    ):
        report_path = Path(out_root) / "flux_checkpoint_jax_report.json"
        return {
            "model_id": model_id,
            "snapshot": "/tmp/fake-snapshot",
            "report_path": str(report_path),
            "ready_count": 1,
            "blocked_count": 2,
            "missing_count": 1,
            "checkpoint_graph_only": checkpoint_graph_only,
            "checkpoint_reduced_shapes": checkpoint_reduced_shapes,
            "error": None,
        }

    monkeypatch.setattr(demo, "prepare_flux_jax_checkpoint_artifacts", _fake_prepare)

    exit_code = demo.main(["--out", str(tmp_path), "--prepare-checkpoint-artifacts"])

    assert exit_code == 0


def test_flux_checkpoint_prepare_report_handles_missing_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()

    def _missing_snapshot(model_id=None):
        raise FileNotFoundError("missing snapshot")

    monkeypatch.setattr(demo.flux_source, "resolve_flux_snapshot", _missing_snapshot)

    report = demo.prepare_flux_jax_checkpoint_artifacts(
        tmp_path / "out",
        model_id="black-forest-labs/FLUX.2-klein-base-4B",
    )

    assert report["model_id"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert report["snapshot"] is None
    assert report["submodule_order"] == [
        "vae_decoder",
        "transformer",
        "text_encoder",
    ]
    assert report["ready_count"] == 0
    assert report["blocked_count"] == 0
    assert report["missing_count"] == 3
    assert report["error"] == "missing snapshot"
    for submodule in ("vae_decoder", "transformer", "text_encoder"):
        assert report["submodules"][submodule]["status"] == "missing"
        assert report["submodules"][submodule]["reason"] == "missing snapshot"
        assert report["submodules"][submodule]["used_reduced_config"] is False

    report_path = Path(str(report["report_path"]))
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["error"] == "missing snapshot"


def test_flux_checkpoint_prepare_report_blocks_on_pytorch_reference_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    demo_specs = demo.build_demo_export_specs()

    monkeypatch.setattr(demo.flux_source, "resolve_flux_snapshot", lambda model_id=None: snapshot)
    monkeypatch.setattr(
        demo.flux_source,
        "snapshot_has_submodule_weights",
        lambda resolved_snapshot, submodule: submodule == "transformer",
    )
    monkeypatch.setattr(
        demo.flux_source,
        "_real_export_spec",
        lambda submodule, model_id=None, **kwargs: demo_specs[submodule],
    )
    monkeypatch.setattr(
        demo,
        "run_export_spec_pytorch_reference",
        lambda spec: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    report = demo.prepare_flux_jax_checkpoint_artifacts(tmp_path / "out")

    assert report["ready_count"] == 0
    assert report["blocked_count"] == 1
    assert report["missing_count"] == 2
    assert report["submodules"]["transformer"]["status"] == "blocked"
    assert report["submodules"]["transformer"]["reason"] == "PyTorch reference failed: boom"
    assert report["submodules"]["transformer"]["reference_output_shapes"] == []


def test_flux_checkpoint_prepare_report_handles_report_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    demo_specs = demo.build_demo_export_specs()

    monkeypatch.setattr(demo.flux_source, "resolve_flux_snapshot", lambda model_id=None: snapshot)
    monkeypatch.setattr(
        demo.flux_source,
        "snapshot_has_submodule_weights",
        lambda resolved_snapshot, submodule: True,
    )
    monkeypatch.setattr(
        demo.flux_source,
        "_real_export_spec",
        lambda submodule, model_id=None, **kwargs: demo_specs[submodule],
    )

    def _fake_export(submodule, out_path, **_kwargs):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fake-{submodule}", encoding="utf-8")
        return path

    monkeypatch.setattr(demo.flux_source, "export_flux_submodule_onnx", _fake_export)
    monkeypatch.setattr(demo.flux_source, "load_unsupported_onnx_ops", lambda onnx_path: ())

    def _fake_transpile(onnx_path, target, out_dir, *, config):
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_file = output_dir / config.weights_filename
        weights_file.write_text("weights", encoding="utf-8")
        return SimpleNamespace(target=target, weights_file=weights_file)

    monkeypatch.setattr(demo, "transpile_onnx", _fake_transpile)
    monkeypatch.setattr(
        demo,
        "run_export_spec_pytorch_reference",
        lambda spec: {"output_shapes": [list(spec.sample_inputs[0].shape)]},
    )
    monkeypatch.setattr(demo, "_write_json_report", lambda report_path, report: "disk full")

    report = demo.prepare_flux_jax_checkpoint_artifacts(
        tmp_path / "out",
        checkpoint_submodule="transformer",
    )

    assert report["ready_count"] == 1
    assert report["blocked_count"] == 0
    assert report["missing_count"] == 0
    assert report["report_write_error"] == "disk full"


def test_flux_checkpoint_prepare_report_cli_passes_model_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    captured: dict[str, object] = {}

    def _fake_prepare(
        out_root,
        *,
        model_id=None,
        checkpoint_submodule=None,
        checkpoint_graph_only=False,
        checkpoint_reduced_shapes=False,
    ):
        captured["out_root"] = out_root
        captured["model_id"] = model_id
        captured["checkpoint_submodule"] = checkpoint_submodule
        captured["checkpoint_graph_only"] = checkpoint_graph_only
        captured["checkpoint_reduced_shapes"] = checkpoint_reduced_shapes
        report_path = Path(out_root) / "flux_checkpoint_jax_report.json"
        return {
            "model_id": model_id,
            "snapshot": "/tmp/fake-snapshot",
            "report_path": str(report_path),
            "ready_count": 1,
            "blocked_count": 2,
            "missing_count": 1,
            "checkpoint_graph_only": checkpoint_graph_only,
            "checkpoint_reduced_shapes": checkpoint_reduced_shapes,
            "error": None,
        }

    monkeypatch.setattr(demo, "prepare_flux_jax_checkpoint_artifacts", _fake_prepare)

    exit_code = demo.main(
        [
            "--out",
            str(tmp_path),
            "--prepare-checkpoint-artifacts",
            "--model-id",
            "black-forest-labs/FLUX.2-klein-base-4B",
        ]
    )

    assert exit_code == 0
    assert captured["model_id"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert captured["checkpoint_submodule"] is None
    assert captured["checkpoint_graph_only"] is False
    assert captured["checkpoint_reduced_shapes"] is False


def test_flux_checkpoint_prepare_report_cli_passes_checkpoint_submodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    captured: dict[str, object] = {}

    def _fake_prepare(
        out_root,
        *,
        model_id=None,
        checkpoint_submodule=None,
        checkpoint_graph_only=False,
        checkpoint_reduced_shapes=False,
    ):
        captured["out_root"] = out_root
        captured["model_id"] = model_id
        captured["checkpoint_submodule"] = checkpoint_submodule
        captured["checkpoint_graph_only"] = checkpoint_graph_only
        captured["checkpoint_reduced_shapes"] = checkpoint_reduced_shapes
        report_path = Path(out_root) / "flux_checkpoint_jax_report.json"
        return {
            "model_id": model_id,
            "snapshot": "/tmp/fake-snapshot",
            "report_path": str(report_path),
            "ready_count": 1,
            "blocked_count": 0,
            "missing_count": 0,
            "checkpoint_graph_only": checkpoint_graph_only,
            "checkpoint_reduced_shapes": checkpoint_reduced_shapes,
            "error": None,
        }

    monkeypatch.setattr(demo, "prepare_flux_jax_checkpoint_artifacts", _fake_prepare)

    exit_code = demo.main(
        [
            "--out",
            str(tmp_path),
            "--prepare-checkpoint-artifacts",
            "--checkpoint-submodule",
            "transformer",
        ]
    )

    assert exit_code == 0
    assert captured["checkpoint_submodule"] == "transformer"
    assert captured["checkpoint_graph_only"] is False
    assert captured["checkpoint_reduced_shapes"] is False


def test_flux_checkpoint_prepare_report_uses_graph_only_and_reduced_transformer_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    demo_specs = demo.build_demo_export_specs()
    captured: dict[str, object] = {}

    monkeypatch.setattr(demo.flux_source, "resolve_flux_snapshot", lambda model_id=None: snapshot)
    monkeypatch.setattr(
        demo.flux_source,
        "snapshot_has_submodule_weights",
        lambda resolved_snapshot, submodule: submodule == "transformer",
    )

    def _fake_real_export_spec(submodule, model_id=None, **kwargs):
        captured["real_export_kwargs"] = dict(kwargs)
        return demo_specs[submodule]

    monkeypatch.setattr(demo.flux_source, "_real_export_spec", _fake_real_export_spec)

    def _fake_export(submodule, out_path, **kwargs):
        captured["export_kwargs"] = dict(kwargs)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fake-{submodule}", encoding="utf-8")
        return path

    monkeypatch.setattr(demo.flux_source, "export_flux_submodule_onnx", _fake_export)
    monkeypatch.setattr(demo.flux_source, "load_unsupported_onnx_ops", lambda onnx_path: ())

    def _fake_transpile(onnx_path, target, out_dir, *, config):
        captured["transpile_called"] = True
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_file = output_dir / config.weights_filename
        weights_file.write_text("weights", encoding="utf-8")
        return SimpleNamespace(target=target, weights_file=weights_file)

    monkeypatch.setattr(demo, "transpile_onnx", _fake_transpile)
    monkeypatch.setattr(
        demo,
        "run_export_spec_pytorch_reference",
        lambda spec: {"output_shapes": [list(spec.sample_inputs[0].shape)]},
    )

    report = demo.prepare_flux_jax_checkpoint_artifacts(
        tmp_path / "out",
        checkpoint_submodule="transformer",
        checkpoint_graph_only=True,
        checkpoint_reduced_shapes=True,
    )

    assert report["ready_count"] == 1
    assert report["checkpoint_graph_only"] is True
    assert report["checkpoint_reduced_shapes"] is True
    assert report["submodules"]["transformer"]["status"] == "ready"
    assert report["submodules"]["transformer"]["generated_dir"] is None
    assert report["submodules"]["transformer"]["weights_file"] is None
    assert report["submodules"]["transformer"]["used_reduced_config"] is True
    assert captured["real_export_kwargs"] == {
        "load_weights": False,
        "transformer_image_seq_len": demo.flux_source.CHECKPOINT_SMOKE_IMAGE_SEQ_LEN,
        "transformer_text_seq_len": demo.flux_source.CHECKPOINT_SMOKE_TEXT_SEQ_LEN,
    }
    export_kwargs = cast(dict[str, object], captured["export_kwargs"])
    assert export_kwargs["export_params"] is False
    assert "transpile_called" not in captured


def test_flux_checkpoint_prepare_report_uses_graph_only_text_encoder_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    demo_specs = demo.build_demo_export_specs()
    captured: dict[str, object] = {}

    monkeypatch.setattr(demo.flux_source, "resolve_flux_snapshot", lambda model_id=None: snapshot)
    monkeypatch.setattr(
        demo.flux_source,
        "snapshot_has_submodule_weights",
        lambda resolved_snapshot, submodule: submodule == "text_encoder",
    )

    def _fake_real_export_spec(submodule, model_id=None, **kwargs):
        captured["real_export_kwargs"] = dict(kwargs)
        return demo_specs[submodule]

    monkeypatch.setattr(demo.flux_source, "_real_export_spec", _fake_real_export_spec)

    def _fake_export(submodule, out_path, **kwargs):
        captured["export_kwargs"] = dict(kwargs)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fake-{submodule}", encoding="utf-8")
        return path

    monkeypatch.setattr(demo.flux_source, "export_flux_submodule_onnx", _fake_export)
    monkeypatch.setattr(demo.flux_source, "load_unsupported_onnx_ops", lambda onnx_path: ())

    def _fake_transpile(onnx_path, target, out_dir, *, config):
        captured["transpile_called"] = True
        raise AssertionError("graph-only text-encoder lane should not reach transpilation")

    monkeypatch.setattr(demo, "transpile_onnx", _fake_transpile)
    monkeypatch.setattr(
        demo,
        "run_export_spec_pytorch_reference",
        lambda spec: {"output_shapes": [list(spec.sample_inputs[0].shape)]},
    )

    report = demo.prepare_flux_jax_checkpoint_artifacts(
        tmp_path / "out",
        checkpoint_submodule="text_encoder",
        checkpoint_graph_only=True,
    )

    assert report["ready_count"] == 1
    assert report["checkpoint_graph_only"] is True
    assert report["submodules"]["text_encoder"]["status"] == "ready"
    assert report["submodules"]["text_encoder"]["generated_dir"] is None
    assert report["submodules"]["text_encoder"]["weights_file"] is None
    assert report["submodules"]["text_encoder"]["used_reduced_config"] is True
    assert captured["real_export_kwargs"] == {"load_weights": False}
    export_kwargs = cast(dict[str, object], captured["export_kwargs"])
    assert export_kwargs["export_params"] is False
    assert "transpile_called" not in captured


def test_flux_checkpoint_prepare_report_cli_passes_graph_only_and_reduced_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    captured: dict[str, object] = {}

    def _fake_prepare(
        out_root,
        *,
        model_id=None,
        checkpoint_submodule=None,
        checkpoint_graph_only=False,
        checkpoint_reduced_shapes=False,
    ):
        captured["checkpoint_graph_only"] = checkpoint_graph_only
        captured["checkpoint_reduced_shapes"] = checkpoint_reduced_shapes
        report_path = Path(out_root) / "flux_checkpoint_jax_report.json"
        return {
            "model_id": model_id,
            "snapshot": "/tmp/fake-snapshot",
            "report_path": str(report_path),
            "ready_count": 1,
            "blocked_count": 0,
            "missing_count": 0,
            "checkpoint_graph_only": checkpoint_graph_only,
            "checkpoint_reduced_shapes": checkpoint_reduced_shapes,
            "error": None,
            "submodule_order": ["transformer"],
            "submodules": {"transformer": {"status": "ready", "reason": None}},
        }

    monkeypatch.setattr(demo, "prepare_flux_jax_checkpoint_artifacts", _fake_prepare)

    exit_code = demo.main(
        [
            "--out",
            str(tmp_path),
            "--prepare-checkpoint-artifacts",
            "--checkpoint-submodule",
            "transformer",
            "--checkpoint-graph-only",
            "--checkpoint-reduced-shapes",
        ]
    )

    assert exit_code == 0
    assert captured["checkpoint_graph_only"] is True
    assert captured["checkpoint_reduced_shapes"] is True


def test_flux_checkpoint_prepare_report_cli_prints_status_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    demo = _flux_demo_module()

    def _fake_prepare(
        out_root,
        *,
        model_id=None,
        checkpoint_submodule=None,
        checkpoint_graph_only=False,
        checkpoint_reduced_shapes=False,
    ):
        report_path = Path(out_root) / "flux_checkpoint_jax_report.json"
        return {
            "model_id": model_id or "black-forest-labs/FLUX.2-klein-4B",
            "snapshot": "/tmp/fake-snapshot",
            "report_path": str(report_path),
            "ready_count": 0,
            "blocked_count": 1,
            "missing_count": 0,
            "checkpoint_graph_only": checkpoint_graph_only,
            "checkpoint_reduced_shapes": checkpoint_reduced_shapes,
            "error": None,
            "report_write_error": "disk full",
            "submodule_order": ["transformer"],
            "submodules": {
                "transformer": {
                    "status": "blocked",
                    "reason": "Unsupported ONNX ops: ['Foo']",
                }
            },
        }

    monkeypatch.setattr(demo, "prepare_flux_jax_checkpoint_artifacts", _fake_prepare)

    exit_code = demo.main(
        [
            "--out",
            str(tmp_path),
            "--prepare-checkpoint-artifacts",
            "--checkpoint-submodule",
            "transformer",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Checkpoint graph only: False" in output
    assert "Checkpoint reduced shapes: False" in output
    assert "Report write error: disk full" in output
    assert "transformer: blocked (Unsupported ONNX ops: ['Foo'])" in output


def test_flux_checkpoint_prepare_report_cli_downloads_assets_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _flux_demo_module()
    captured: dict[str, object] = {}

    def _fake_download(model_id):
        captured["model_id"] = model_id
        return Path("/tmp/fake")

    monkeypatch.setattr(demo, "download_snapshot_from_hub", _fake_download)

    exit_code = demo.main(
        [
            "--out",
            str(tmp_path),
            "--download-checkpoint-assets",
            "--model-id",
            "black-forest-labs/FLUX.2-klein-4B",
        ]
    )

    assert exit_code == 0
    assert captured["model_id"] == "black-forest-labs/FLUX.2-klein-4B"


def test_flux_checkpoint_prepare_report_cli_rejects_download_with_generation_flags(
    tmp_path: Path,
) -> None:
    demo = _flux_demo_module()

    with pytest.raises(
        ValueError,
        match="--download-checkpoint-assets cannot be combined with prompt/image generation flags",
    ):
        demo.main(
            [
                "--out",
                str(tmp_path),
                "--download-checkpoint-assets",
                "--prompt-npz",
                str(tmp_path / "prompt_fixture.npz"),
            ]
        )


def test_flux_checkpoint_prepare_report_cli_rejects_generation_flags(
    tmp_path: Path,
) -> None:
    demo = _flux_demo_module()

    with pytest.raises(
        ValueError,
        match=(
            "--prepare-checkpoint-artifacts cannot be combined with prompt/image generation flags"
        ),
    ):
        demo.main(
            [
                "--out",
                str(tmp_path),
                "--prepare-checkpoint-artifacts",
                "--use-text-encoders",
            ]
        )


def test_flux_checkpoint_prepare_report_cli_rejects_checkpoint_submodule_without_prepare(
    tmp_path: Path,
) -> None:
    demo = _flux_demo_module()

    with pytest.raises(
        ValueError,
        match="--checkpoint-submodule requires --prepare-checkpoint-artifacts.",
    ):
        demo.main(
            [
                "--out",
                str(tmp_path),
                "--checkpoint-submodule",
                "transformer",
            ]
        )


def test_flux_checkpoint_prepare_report_cli_rejects_graph_only_without_prepare(
    tmp_path: Path,
) -> None:
    demo = _flux_demo_module()

    with pytest.raises(
        ValueError,
        match="--checkpoint-graph-only requires --prepare-checkpoint-artifacts.",
    ):
        demo.main(
            [
                "--out",
                str(tmp_path),
                "--checkpoint-graph-only",
            ]
        )


def test_flux_checkpoint_prepare_report_cli_rejects_reduced_shapes_without_prepare(
    tmp_path: Path,
) -> None:
    demo = _flux_demo_module()

    with pytest.raises(
        ValueError,
        match="--checkpoint-reduced-shapes requires --prepare-checkpoint-artifacts.",
    ):
        demo.main(
            [
                "--out",
                str(tmp_path),
                "--checkpoint-reduced-shapes",
            ]
        )
