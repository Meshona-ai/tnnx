# Validation

## Baseline Status

Status: `PRE-EXISTING-FAILURES`.

Default lint/type/build gates pass. Default pytest fails because the default dev install lacks examples deps required by FLUX tests and because pytest enables an environment-dependent Whisper MLX audio path. After installing the examples group, only the two Whisper audio tests fail, and direct `ffmpeg -version` confirms the host ffmpeg binary aborts on missing `libx265.215.dylib`.

## Validation Matrix

| ID | Command | Workdir | Prerequisites | Proves | Expected | Baseline | When | Class | Coverage | Context | Failure policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V01 | uv run ruff check . | repo root | dev env | lint/import ordering | All checks passed | PASS | baseline/final | REQUIRED | all | no | fail blocks |
| V02 | uv run ruff format --check . | repo root | dev env | format | 171 files already formatted | PASS | baseline/final | REQUIRED | all | no | fail blocks |
| V03 | uv run ty check src | repo root | dev env | type check for package | All checks passed | PASS | baseline/final | REQUIRED | src | no | fail blocks |
| V04 | uv run pytest -q | repo root | default dev | default suite | pass | FAIL: 15 failed, 206 passed, 15 skipped before examples group | baseline/final | REQUIRED | all | no | pre-existing failure until T01/T02 |
| V05 | uv sync --dev --group examples && uv run pytest -q | repo root | examples deps | examples-enabled suite | pass | FAIL: 2 Whisper ffmpeg failures, 222 passed, 12 skipped | baseline/final | REQUIRED | all | no | environment-blocked until T02 |
| V06 | uv build | repo root | build backend | wheel/sdist build | wheel and sdist built | PASS | baseline/final | REQUIRED | package | no | fail blocks |
| V07 | uv run --no-project --python /opt/homebrew/opt/python@3.14/bin/python3.14 --with dist/tnnx-0.1.0-py3-none-any.whl python -c 'import tnnx' | outside checkout | built wheel | clean wheel import | 0.1.0 imports | PASS | baseline/final | REQUIRED | package | no | fail blocks |
| V08 | uv run tnnx --help && uv run tnnx transpile --help | repo root | dev env | CLI parser | help text renders | PASS | baseline/final | REQUIRED | CLI | no | fail blocks |
| V09 | uv run tnnx transpile --onnx <tmp>/tnnx_cli_smoke.onnx --target jax --out <tmp>/tnnx_cli_smoke_jax | repo root | tiny ONNX | CLI JAX generation | artifacts written and model_jax.py compiles | PASS | baseline/final | REQUIRED | JAX | no | fail blocks |
| V10 | uv run python -m examples.model_zoo.transpile_smoke --target jax --model ResNet-18 | repo root | examples deps | model-zoo smoke | ResNet-18 transpiled | PASS | baseline/final | OPTIONAL | model-zoo/JAX | no | fail blocks only model-zoo release |
| V11 | prescribed low-level residue grep excluding `.codex/plans/**`, `docs/llm_context/**`, and `llms.txt` | repo root | git tracked tree | retired FPGA/HLS/C/native product/config residue | no true hits | PASS | baseline/final | REQUIRED | repo | no | any true product/config hit blocks |
| V12 | docs/llm_context machine index parse | repo root | Python stdlib | context pack consistency | JSON/JSONL/TSV parse | TO RUN AFTER GENERATION | final | REQUIRED | context | yes | fail blocks plan commit |

## Validation Tiers

- Tier 0: imports, CLI help, config loading, catalog listing.
- Tier 1: unit/contract tests.
- Tier 2: generated code parses, compiles, imports, and has no unresolved placeholders.
- Tier 3: numerical parity on practical small models.
- Tier 4: named large-model generation using available fixtures/cache/assets.
- Tier 5: full large-model inference when already supported and practical.

## Final Gate Target

A final execution pass should run: ruff check, ruff format check, ty, default pytest, examples-enabled pytest or intentionally split examples marker, build, clean wheel import, CLI smoke, generated-code parse/import for JAX and MLX, model-zoo ResNet-18 smoke, context-pack lint, and residue grep.
