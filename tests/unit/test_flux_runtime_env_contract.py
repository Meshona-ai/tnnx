from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _runtime_env_module():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.runtime_env as runtime_env

    return runtime_env


def test_flux_runtime_env_resolves_hf_token_from_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_env = _runtime_env_module()
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HUGGING_FACE_TOKEN=test-token\n", encoding="utf-8")

    monkeypatch.delenv(runtime_env.TOKEN_ENV_VAR, raising=False)
    monkeypatch.setattr(runtime_env, "DOTENV_PATH", dotenv_path)

    assert runtime_env.resolve_hf_token() == "test-token"


def test_flux_runtime_env_prefers_hf_token_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_env = _runtime_env_module()
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HUGGING_FACE_TOKEN=dotenv-token\n", encoding="utf-8")

    monkeypatch.setenv(runtime_env.TOKEN_ENV_VAR, "env-token")
    monkeypatch.setattr(runtime_env, "DOTENV_PATH", dotenv_path)

    assert runtime_env.resolve_hf_token() == "env-token"


def test_flux_runtime_env_required_hub_patterns_cover_flux2_layout() -> None:
    runtime_env = _runtime_env_module()

    patterns = runtime_env.required_hub_patterns("black-forest-labs/FLUX.2-klein-4B")

    assert "model_index.json" in patterns
    assert "scheduler/*" in patterns
    assert "tokenizer/*" in patterns
    assert "text_encoder/*" in patterns
    assert "transformer/*" in patterns
    assert "vae/*" in patterns
    assert "tokenizer_2/*" not in patterns
    assert "text_encoder_2/*" not in patterns


def test_flux_runtime_env_required_hub_patterns_cover_fp8_single_file() -> None:
    runtime_env = _runtime_env_module()

    patterns = runtime_env.required_hub_patterns("black-forest-labs/FLUX.2-klein-4b-fp8")

    assert patterns == ("README.md", "LICENSE.md", "*.safetensors")


def test_flux_runtime_env_download_snapshot_from_hub_uses_resolved_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_env = _runtime_env_module()
    captured: dict[str, str | tuple[str, ...]] = {}

    monkeypatch.setattr(runtime_env, "resolve_hf_token", lambda: "secret-token")

    def _fake_snapshot_download(*, repo_id, token, allow_patterns):
        captured["repo_id"] = repo_id
        captured["token"] = token
        captured["allow_patterns"] = tuple(allow_patterns)
        return "/tmp/fake-flux-snapshot"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)

    snapshot = runtime_env.download_snapshot_from_hub("black-forest-labs/FLUX.2-klein-4B")

    assert snapshot == Path("/tmp/fake-flux-snapshot")
    assert captured["repo_id"] == "black-forest-labs/FLUX.2-klein-4B"
    assert captured["token"] == "secret-token"
    assert "transformer/*" in captured["allow_patterns"]
