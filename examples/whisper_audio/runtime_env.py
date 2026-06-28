from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

UV_ARCHIVE_ROOT = Path.home() / ".cache" / "uv" / "archive-v0"
HF_WHISPER_ROOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--whisper-tiny"


def _root_for_package_init(init_path: Path) -> Path:
    parts = init_path.parts
    if "site-packages" in parts:
        idx = parts.index("site-packages")
        return Path(*parts[: idx + 1])
    return init_path.parent.parent


def _add_first_match(pattern: str) -> Path:
    for candidate in sorted(UV_ARCHIVE_ROOT.glob(pattern)):
        root = _root_for_package_init(candidate)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root
    raise FileNotFoundError(
        f"Could not locate cached package matching {pattern!r} under {UV_ARCHIVE_ROOT}"
    )


def _ensure_importable(module_name: str, fallback_pattern: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    _add_first_match(fallback_pattern)


def ensure_runtime_paths(*, require_mlx: bool) -> None:
    _ensure_importable("torch", "*/lib/python*/site-packages/torch/__init__.py")
    _ensure_importable("safetensors", "*/safetensors/__init__.py")
    _ensure_importable("tokenizers", "*/tokenizers/__init__.py")
    if require_mlx:
        _ensure_importable("mlx.core", "*/lib/python*/site-packages/mlx/core.*.so")


def resolve_model_snapshot() -> Path:
    ref_path = HF_WHISPER_ROOT / "refs" / "main"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing Whisper snapshot ref: {ref_path}")
    revision = ref_path.read_text(encoding="utf-8").strip()
    snapshot = HF_WHISPER_ROOT / "snapshots" / revision
    if not snapshot.exists():
        raise FileNotFoundError(f"Missing Whisper snapshot directory: {snapshot}")
    return snapshot
