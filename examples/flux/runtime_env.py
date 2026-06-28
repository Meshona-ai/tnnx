from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
SNAPSHOT_ENV_VAR = "TNNX_FLUX_SNAPSHOT"
TOKEN_ENV_VAR = "HUGGING_FACE_TOKEN"
DOTENV_PATH = Path(".env")
DIFFUSERS_FLUX_HUB_PATTERNS: tuple[str, ...] = (
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "transformer/*",
    "vae/*",
)


def _read_dotenv(path: Path | None = None) -> dict[str, str]:
    dotenv_path = path or DOTENV_PATH
    if not dotenv_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_hf_token() -> str:
    token = os.getenv(TOKEN_ENV_VAR)
    if token:
        return token

    token = _read_dotenv().get(TOKEN_ENV_VAR)
    if token:
        return token

    raise FileNotFoundError(
        f"Missing {TOKEN_ENV_VAR} in the environment or .env. "
        "Set it before downloading FLUX checkpoint assets."
    )


def required_hub_patterns(model_id: str) -> tuple[str, ...]:
    if model_id == "black-forest-labs/FLUX.2-klein-4b-fp8":
        return (
            "README.md",
            "LICENSE.md",
            "*.safetensors",
        )

    return DIFFUSERS_FLUX_HUB_PATTERNS


def download_snapshot_from_hub(
    model_id: str,
    *,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    token = resolve_hf_token()
    patterns = tuple(allow_patterns or required_hub_patterns(model_id))
    return Path(
        snapshot_download(
            repo_id=model_id,
            token=token,
            allow_patterns=list(patterns),
        )
    )


def cache_root_for_model(model_id: str) -> Path:
    return HF_CACHE_ROOT / f"models--{model_id.replace('/', '--')}"


def resolve_snapshot_from_cache(model_id: str) -> Path:
    cache_root = cache_root_for_model(model_id)
    refs_main = cache_root / "refs" / "main"
    snapshots_root = cache_root / "snapshots"

    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        candidate = snapshots_root / revision
        if candidate.exists():
            return candidate

    if snapshots_root.exists():
        snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
        if snapshots:
            return snapshots[-1]

    raise FileNotFoundError(
        f"Missing cached FLUX snapshot for {model_id!r} under {cache_root}. "
        f"Set {SNAPSHOT_ENV_VAR}=<snapshot_dir> to point at a local snapshot."
    )


def resolve_snapshot_root(model_id: str) -> Path:
    override = os.getenv(SNAPSHOT_ENV_VAR)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"{SNAPSHOT_ENV_VAR} points to a missing path: {candidate}")
        return candidate
    return resolve_snapshot_from_cache(model_id)
