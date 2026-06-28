from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_flux_source() -> Any:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.source as source

    return source


def load_flux_example_modules() -> tuple[Any, Any]:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import examples.flux.source as source
    from examples.common import load_generated_module

    return load_generated_module, source


def real_flux_export_spec_or_skip(
    submodule: str,
    *,
    load_weights: bool = True,
    reduced_shapes: bool = False,
) -> tuple[Any, Any]:
    if os.getenv("RUN_FLUX_E2E", "0") != "1":
        pytest.skip(f"Set RUN_FLUX_E2E=1 to run checkpoint-backed FLUX {submodule} checks.")

    source = load_flux_source()
    try:
        snapshot = source.resolve_flux_snapshot()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    if not source.snapshot_has_submodule_weights(snapshot, submodule):
        pytest.skip(f"Missing {submodule} weights under snapshot: {snapshot}")

    try:
        real_export_kwargs: dict[str, Any] = {"load_weights": load_weights}
        if submodule == "transformer" and reduced_shapes:
            real_export_kwargs.update(
                {
                    "transformer_image_seq_len": source.CHECKPOINT_SMOKE_IMAGE_SEQ_LEN,
                    "transformer_text_seq_len": source.CHECKPOINT_SMOKE_TEXT_SEQ_LEN,
                }
            )
        spec = source._real_export_spec(submodule, **real_export_kwargs)
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))

    return source, spec


def real_flux_export_spec_with_loader_or_skip(submodule: str) -> tuple[Any, Any, Any]:
    load_generated_module, source = load_flux_example_modules()
    _, spec = real_flux_export_spec_or_skip(submodule)
    return load_generated_module, source, spec
