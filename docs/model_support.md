# Model Support

Tiers follow `llm_context/validation.md`: Tier 0 is import/catalog, Tier 1 is unit contract,
Tier 2 is generated-code compile/import, Tier 3 is numerical parity on small practical models,
Tier 4 is named model generation with local assets, and Tier 5 is full large-model inference.

| Path | Status | Tier | Assets | Gate |
| --- | --- | ---: | --- | --- |
| MLP/Conv/Residual examples | ready | 3 | generated locally | `uv run pytest -q tests/integration/test_examples_export_cli.py` |
| NanoGPT tiny JAX | ready | 3 | generated locally | `uv run pytest -q tests/integration/test_nanogpt_tiny_example_jax.py` |
| GPT-2 tiny JAX | ready | 3 | Transformers cache or local tiny model | `uv run pytest -q tests/integration/test_gpt2_from_transformers_example_jax.py` |
| Whisper tiny synthetic MLX | ready | 3 | generated locally | `uv run pytest -q tests/integration/test_whisper_tiny_example_mlx.py` |
| Real Whisper audio MLX | ready on this host | 4 | cached `openai/whisper-tiny`, `examples/terminator.mp3`, ffmpeg | `uv run pytest -q tests/integration/test_whisper_audio_real_mlx.py tests/integration/test_whisper_audio_transpile_source.py` |
| Qwen3.5 synthetic JAX/MLX | ready | 3 | generated locally | `uv run pytest -q tests/integration/test_qwen3_5_example_jax.py tests/integration/test_qwen3_5_example_mlx.py` |
| Qwen3.5 real JAX/MLX | environment-gated | 4 | ONNX-community snapshot and tokenizer cache or allowed download | `RUN_QWEN_JAX_E2E=1` or `RUN_QWEN_MLX_E2E=1` targeted tests |
| FLUX reduced/synthetic JAX | ready | 4 | generated locally from reduced configs | `uv run pytest -q tests/integration/test_flux_jax_prompt_to_image_smoke.py` |
| FLUX checkpoint-backed JAX | environment-gated | 4 | `TNNX_FLUX_SNAPSHOT` plus `RUN_FLUX_E2E=1` | checkpoint-backed FLUX integration tests |
| Model zoo ResNet-18 JAX/MLX | ready | 4 | generated locally from torchvision | `uv run pytest -q tests/integration/test_model_zoo_generated_runtime_parity.py` |
| Model zoo BERT/Llama/YOLO lanes | mixed experimental | 3-4 | local runtime deps and model-specific loaders | `uv run python -m examples.model_zoo.transpile_smoke --list` then targeted smoke |

Do not promote an environment-gated path to ready unless the exact gate above passes in the target
environment.
