# Model Zoo And Named Paths

## Canonical Names Discovered

- NanoGPT tiny
- GPT-2 (sshleifer/tiny-gpt2 default)
- Whisper tiny source and openai/whisper-tiny audio lane
- Qwen3.5-0.8B JAX
- Qwen3.5-0.8B MLX
- FLUX.2 Klein JAX reduced/snapshot lanes
- YOLOv2
- YOLOv3-tiny
- YOLOv4-tiny
- YOLOv5s
- YOLOv8n
- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-50 (quantized)
- Small BERT variants
- Tiny ViT
- MobileNetV1/V2
- EfficientNet-lite
- Llama 3.1 8B smoke

## Tier Map

| Model path | Source files | Required assets | Backend | Validation tier | Gate |
| --- | --- | --- | --- | --- | --- |
| NanoGPT tiny | examples/model_nanogpt_tiny.py; examples/run_nanogpt_tiny_jax.py | local generated | JAX | Tier 3 | tests/integration/test_nanogpt_tiny_example_jax.py |
| GPT-2 tiny/default | examples/nanogpt/infer_gpt2_from_transformers_jax.py | HF cache/download unless local | JAX | Tier 3 synthetic/default practical | tests/integration/test_gpt2_from_transformers_example_jax.py |
| Whisper tiny toy | examples/model_whisper_tiny.py; examples/run_whisper_tiny_mlx.py | local generated | MLX | Tier 3 | tests/integration/test_whisper_tiny_example_mlx.py |
| Whisper real audio | examples/whisper_audio/* | HF whisper snapshot; ffmpeg or matching Homebrew x265 fallback; terminator.mp3 | MLX | Tier 4 passing on this host | `uv run pytest -q tests/integration/test_whisper_audio_real_mlx.py tests/integration/test_whisper_audio_transpile_source.py` |
| Qwen3.5 | examples/qwen/* | HF/ONNX-community snapshot or synthetic config | JAX/MLX | Tier 3 synthetic; Tier 4/5 env-gated real | tests/integration/test_qwen3_5_example_jax.py; test_qwen3_5_example_mlx.py |
| FLUX.2 Klein | examples/flux/* | TNNX_FLUX_SNAPSHOT for real; synthetic/tiny config always-on | JAX first | Tier 3 synthetic; Tier 4 env-gated real | tests/integration/test_flux_* |
| ResNet-18 | examples/model_zoo/* | torchvision | JAX/MLX smoke subsets | Tier 4 runtime parity in default suite | tests/integration/test_model_zoo_generated_runtime_parity.py |
| YOLO/quantized ResNet/MobileNet/Llama smoke | examples/model_zoo/* | upstream packages/checkpoints | varies | planned/experimental | model_zoo readiness only unless explicit smoke passes |

## Baseline Model Evidence

- `uv run python -m examples.model_zoo.list_models`: PASS, lists 14 source references.
- `uv run python -m examples.model_zoo.check_compatibility`: PASS, marks ResNet-18/34/50, Small BERT variants, Tiny ViT, EfficientNet-lite as ready; YOLO, quantized ResNet, MobileNet family, and Llama smoke are blocked or experimental.
- `uv run python -m examples.model_zoo.transpile_smoke --target jax --model ResNet-18`: PASS, ONNX export and JAX transpile succeeded.
- Focused named-model gate after implementation: PASS, `15 passed, 2 skipped` across Whisper real audio, Whisper tiny MLX, model-zoo runtime parity, NanoGPT, GPT-2, and Qwen synthetic JAX/MLX.
- Full Qwen real lanes are skipped unless `RUN_QWEN_JAX_E2E=1` or `RUN_QWEN_MLX_E2E=1`.
- Real FLUX checkpoint lanes are skipped unless `RUN_FLUX_E2E=1` and snapshot assets exist.
- Real Whisper audio lane passes in this environment after the loader retries the Homebrew x265 dylib mismatch. It is not skipped for a broken host `ffmpeg`.

## Rules For Future Claims

Do not call a model supported because it is listed in a catalog. A model is supported only to the strongest passing tier recorded in `validation.md`.
