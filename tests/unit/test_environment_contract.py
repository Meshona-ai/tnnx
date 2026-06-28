from __future__ import annotations

import importlib.metadata as metadata
import sys


def test_python_version_is_314() -> None:
    assert sys.version_info[:2] == (3, 14)


def test_package_is_installable() -> None:
    assert metadata.version("tnnx") == "0.1.0"


def test_package_imports() -> None:
    import tnnx

    assert hasattr(tnnx, "__version__")
