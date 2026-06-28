from __future__ import annotations

from tnnx.codegen.common import NameGenerator


def test_name_generator_stability() -> None:
    gen = NameGenerator()
    assert gen.next_temp() == "_v0"
    assert gen.next_temp() == "_v1"
    assert gen.next_temp() == "_v2"
