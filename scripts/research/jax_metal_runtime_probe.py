from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np


def _optional_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("tnnx_generated_jax_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated module at {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_npz(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in sorted(data.files)}


def _select_output(value: Any, output_name: str) -> Any:
    if isinstance(value, dict):
        return value[output_name]
    if isinstance(value, list | tuple):
        return value[0]
    return value


def _block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def _summarize(times: list[float]) -> dict[str, float | int | None]:
    if not times:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "runs": 0}
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "runs": int(len(times)),
    }


def _measure(
    *,
    label: str,
    function: Any,
    params: Any,
    inputs: dict[str, Any],
    output_name: str,
    expected: np.ndarray[Any, Any] | None,
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    output_value = _select_output(function(params, inputs), output_name)
    _block(output_value)
    first_forward_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(warmups):
        output_value = _select_output(function(params, inputs), output_name)
        _block(output_value)

    times: list[float] = []
    output_value = None
    for _ in range(runs):
        start = time.perf_counter()
        output_value = _select_output(function(params, inputs), output_name)
        _block(output_value)
        times.append((time.perf_counter() - start) * 1000.0)

    max_abs_error = None
    output_shape = None
    if output_value is not None:
        actual = np.asarray(_block(output_value))
        output_shape = list(actual.shape)
        if expected is not None and actual.shape == expected.shape:
            max_abs_error = float(np.max(np.abs(actual - expected)))

    return {
        "label": label,
        "status": "ok",
        "first_forward_ms": first_forward_ms,
        "output_shape": output_shape,
        "max_abs_error": max_abs_error,
        **_summarize(times),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    module_path = args.artifact_dir / "model_jax.py"
    weights_path = args.artifact_dir / "weights.npz"
    inputs_npz = _load_npz(args.inputs_npz)
    expected_npz = _load_npz(args.expected_npz) if args.expected_npz is not None else {}
    expected = expected_npz.get(args.output_name)

    start = time.perf_counter()
    module = _load_module(module_path)
    params = module.load_weights(str(weights_path))
    inputs = {name: jnp.asarray(value) for name, value in inputs_npz.items()}
    cold_start_ms = (time.perf_counter() - start) * 1000.0

    variants = [
        _measure(
            label="Generated JAX Metal eager",
            function=module.forward,
            params=params,
            inputs=inputs,
            output_name=args.output_name,
            expected=expected,
            warmups=args.warmups,
            runs=args.runs,
        )
    ]

    jitted = jax.jit(lambda actual_inputs: module.forward(params, actual_inputs))

    def _jit_forward(_params: Any, actual_inputs: dict[str, Any]) -> Any:
        return jitted(actual_inputs)

    variants.append(
        _measure(
            label="Generated JAX Metal jit",
            function=_jit_forward,
            params=None,
            inputs=inputs,
            output_name=args.output_name,
            expected=expected,
            warmups=args.warmups,
            runs=args.runs,
        )
    )

    primary = variants[-1]
    return {
        "status": "ok",
        "runtime": "jax-metal-compat",
        "cold_start_ms": cold_start_ms,
        "device": ",".join(str(device) for device in jax.devices()),
        "jax_default_backend": str(jax.default_backend()),
        "python": platform.python_version(),
        "versions": {
            "jax": jax.__version__,
            "jaxlib": _optional_version("jaxlib"),
            "jax-metal": _optional_version("jax-metal"),
        },
        "variants": variants,
        **{key: value for key, value in primary.items() if key not in {"label", "status"}},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure generated TNNX JAX artifacts in an isolated JAX Metal runtime."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--inputs-npz", type=Path, required=True)
    parser.add_argument("--expected-npz", type=Path, default=None)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = run(args)
    except Exception as exc:
        result = {
            "status": "failed",
            "runtime": "jax-metal-compat",
            "python": platform.python_version(),
            "message": str(exc),
        }
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
