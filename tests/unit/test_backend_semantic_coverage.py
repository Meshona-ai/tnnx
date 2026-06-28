from __future__ import annotations

import inspect
import re

from tnnx.codegen import jax_codegen, mlx_codegen
from tnnx.ir.schema import SEMANTIC_SCHEMAS

_EQUALITY_DISPATCH = re.compile(r'if node\.op == "([A-Z0-9_]+)"')
_SET_DISPATCH = re.compile(r"if node\.op in \{([^}]*)\}")


def _dispatched_ops(module: object) -> set[str]:
    source = inspect.getsource(module._emit_node_expr)  # type: ignore[attr-defined]
    ops = set(_EQUALITY_DISPATCH.findall(source))
    for group in _SET_DISPATCH.findall(source):
        ops.update(re.findall(r'"([A-Z0-9_]+)"', group))
    return ops


def test_jax_codegen_dispatch_covers_all_semantic_ops() -> None:
    assert _dispatched_ops(jax_codegen) == set(SEMANTIC_SCHEMAS)


def test_mlx_codegen_dispatch_covers_all_semantic_ops() -> None:
    assert _dispatched_ops(mlx_codegen) == set(SEMANTIC_SCHEMAS)
