from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "tnnx"

TOTAL_SOURCE_LOC_BUDGET = 6500
IR_INGEST_LOC_BUDGET = 900
JAX_BACKEND_LOC_BUDGET = 1600
MLX_BACKEND_LOC_BUDGET = 1500


def _count_py_lines(directory: Path) -> int:
    if directory.is_file():
        return len(directory.read_text(encoding="utf-8").splitlines())
    total = 0
    for path in directory.rglob("*.py"):
        total += len(path.read_text(encoding="utf-8").splitlines())
    return total


def test_total_source_loc_budget() -> None:
    assert _count_py_lines(SRC) <= TOTAL_SOURCE_LOC_BUDGET


def test_ir_ingest_loc_budget() -> None:
    assert _count_py_lines(SRC / "ir") + _count_py_lines(SRC / "ingest") <= IR_INGEST_LOC_BUDGET


def test_backend_loc_budget() -> None:
    assert _count_py_lines(SRC / "codegen" / "jax_codegen.py") <= JAX_BACKEND_LOC_BUDGET
    assert _count_py_lines(SRC / "codegen" / "mlx_codegen.py") <= MLX_BACKEND_LOC_BUDGET
