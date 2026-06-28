# Baseline Evidence

## Environment

- Review base ref: `local-main-initial-review-base`
- Review base SHA: `91bf6abf1f392863d0c3cdca692d3272efdceb4b`
- OS/arch: Darwin arm64, macOS 26.3.1 observed via Python/platform and `uname`.
- Python: 3.14.6 in project env.
- Package manager: uv 0.11.21.
- Lock hashes: `uv.lock` sha256 `d03397a8aa0530dbab720333bb83f08f6d9f66e2d4ab12f3ef66e6d8045788cf`; `pyproject.toml` sha256 `2971db896f03920f7bc4ef35056abe032ebe8c213b36e6f641eb37fb7ec1b21e`.
- Backends: numpy 2.4.2; onnx 1.20.1; jax/jaxlib 0.9.0.1; torch 2.10.0; transformers 5.2.0; diffusers 0.36.0 after examples group; onnxscript 0.6.2 after examples group; MLX import available.
- Accelerators: JAX CPU device only; torch MPS available; CUDA unavailable.
- Env vars by name: RUN_MLX_E2E unset in shell but set to `1` by `pyproject.toml` pytest env; RUN_FLUX_E2E unset; RUN_QWEN_JAX_E2E unset; RUN_QWEN_MLX_E2E unset; TNNX_FLUX_SNAPSHOT unset; TNNX_WHISPER_SNAPSHOT unset; JAX_ENABLE_X64 unset.
- Network: `git ls-remote --heads origin main` succeeded but returned no `main` head. Examples group install resolved from lock/cache in this environment.

## Repository Metrics

- Tracked file count at review base: 214.
- Total counted lines via `git ls-files | xargs wc -l`: 29447.
- `src/tnnx`: 26 files / 4421 LOC.
- `tests`: 111 files / 9079 LOC.
- `examples`: 44 files / 8702 LOC.
- `docs`: 3 files / 215 LOC before context pack.
- `scripts`: 11 files / 3699 LOC.
- `.agents`: 8 files / 634 LOC.
- Dependencies: runtime 2; dev group 13; examples group 3.
- Optional-extra count: 0 project optional extras; dependency groups are `dev` and `examples`.
- Built wheel: 44,675 bytes, 30 entries, package-only.
- Built sdist: 411,066 bytes, 215 entries, includes examples, tests, scripts, docs, `.agents`, and assets.
- Large tracked artifact: `examples/terminator.mp3` about 156 KB; `examples/whisper_audio/assets/mel_filters.npz` about 8 KB.
- Ignored artifacts observed: `.venv`, `.pytest_cache`, `.ruff_cache`, `dist`, `__pycache__` trees.

## Existing Checks

| Command | Exit | Signature |
| --- | --- | --- |
| uv run ruff check . | 0 | All checks passed |
| uv run ruff format --check . | 0 | 171 files already formatted |
| uv run ty check src | 0 | All checks passed |
| uv run pytest -q | 1 | 15 failed, 206 passed, 15 skipped before examples group; FLUX missing onnxscript/diffusers and Whisper ffmpeg failure |
| uv sync --dev --group examples | 0 | Installed accelerate, diffusers, onnx-ir, onnxscript and metadata deps |
| uv run pytest -q | 1 | 2 failed, 222 passed, 12 skipped, 131 warnings after examples group; both Whisper ffmpeg failures |
| uv build | 0 | wheel and sdist built |
| clean wheel import on Python 3.14 | 0 | tnnx 0.1.0 imports and __all__ matches |
| uv run tnnx --help | 0 | CLI help renders |
| uv run tnnx transpile --help | 0 | transpile flags render |
| tiny CLI JAX transpile + py_compile | 0 | weights, metadata, graph_ir, model_jax.py written and compiles |
| uv run python -m examples.model_zoo.transpile_smoke --target jax --model ResNet-18 | 0 | ResNet-18 ONNX export + transpile succeeded |
| low-level residue grep | 1/no matches | no FPGA/HLS/RTL/C/CMake/native-backend hits |

## Baseline Failures

- BF001: default pytest fails because FLUX tests require examples deps absent from default dev sync. Reproducible. Repository-owned dependency/gating issue. Planned by T01.
- BF002: default/examples-enabled pytest fails because host `ffmpeg` aborts on missing x265 dynamic library. Reproducible. Environment blocker plus repository-owned skip policy issue. Planned by T02.

## Public Behavior Snapshot

Top-level exports: `ArtifactManifest`, `CompileConfig`, `DEFAULT_PASSES`, `ResourceBudget`, `transpile_onnx`.

Console entry point: `tnnx` with subcommand `transpile`.

Backends: `jax`, `mlx`.

Default passes: `prune`, `normalize`, `shape_prop`.

Artifacts: `weights.npz`, backend module, `compile_metadata.json`, optional `graph_ir.json`.

## Generated-Code Baseline

Tiny CLI JAX generation completed; generated `model_jax.py` parses/compiles. Snapshot and runtime parity tests cover broader JAX/MLX generated code, but generated-code import/compile gates should be explicit for both backends.

## Named-Model Baseline

See `docs/llm_context/model_zoo.md`. Canonical paths include NanoGPT, GPT-2, Whisper, Qwen3.5, FLUX.2 Klein, model-zoo ResNet/BERT/ViT/EfficientNet/YOLO/MobileNet/Llama entries. Only ResNet-18 smoke was run directly during this audit; many real paths are environment/network/snapshot gated.
