from .infer_qwen3_5_from_transformers_jax import build_tiny_qwen3_5_config
from .infer_qwen3_5_from_transformers_jax import run_demo as run_demo_jax
from .infer_qwen3_5_from_transformers_mlx import run_demo as run_demo_mlx

run_demo = run_demo_jax

__all__ = [
    "build_tiny_qwen3_5_config",
    "run_demo",
    "run_demo_jax",
    "run_demo_mlx",
]
