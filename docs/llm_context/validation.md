# Validation

## Current Status

Status: `DEFAULT-GATES-PASS` on `codex/tnnx-second-pass-exec`.

Default lint/type/build gates pass. Default pytest includes FLUX example dependencies through the `dev` dependency group and runs Whisper real-audio MLX tests by default. The Whisper audio loader retries the observed Homebrew `ffmpeg`/x265 dylib mismatch with an installed legacy x265 keg when dyld reports `libx265.215.dylib` missing.

## Validation Matrix

| ID | Command | Workdir | Prerequisites | Proves | Expected | Baseline | When | Class | Coverage | Context | Failure policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V01 | uv run ruff check . | repo root | dev env | lint/import ordering | All checks passed | PASS | baseline/final | REQUIRED | all | no | fail blocks |
| V02 | uv run ruff format --check . | repo root | dev env | format | 169 files already formatted | PASS | baseline/final | REQUIRED | all | no | fail blocks |
| V03 | uv run ty check src | repo root | dev env | type check for package | All checks passed | PASS | baseline/final | REQUIRED | src | no | fail blocks |
| V04 | uv run pytest -q | repo root | default dev | default suite | pass | PASS: 238 passed, 12 skipped in the final-tree `scripts/check.sh` run | baseline/final | REQUIRED | all | no | fail blocks |
| V05 | uv sync --dev --group examples && uv run pytest -q | repo root | examples deps | examples-enabled suite | pass | Superseded for default FLUX deps; examples group remains valid for explicit example installs | baseline/final | OPTIONAL | all/examples | no | fail blocks only example-release validation |
| V06 | uv build | repo root | build backend | wheel/sdist build | wheel and sdist built | PASS | baseline/final | REQUIRED | package | no | fail blocks |
| V07 | uv run --no-project --python /opt/homebrew/opt/python@3.14/bin/python3.14 --with dist/tnnx-0.1.0-py3-none-any.whl python -c 'import tnnx' | outside checkout | built wheel | clean wheel import | imports from uv package archive, not checkout | PASS | baseline/final | REQUIRED | package | no | fail blocks |
| V08 | uv run tnnx --help && uv run tnnx transpile --help | repo root | dev env | CLI parser | help text renders | PASS | baseline/final | REQUIRED | CLI | no | fail blocks |
| V09 | uv run tnnx transpile --onnx <tmp>/tnnx_cli_smoke.onnx --target jax/mlx --out <tmp>/tnnx_cli_smoke_<target> | repo root | tiny ONNX | CLI JAX and MLX generation/runtime | generated modules import and return expected Add output | PASS | baseline/final | REQUIRED | JAX/MLX | no | fail blocks |
| V10 | uv run python -m examples.model_zoo.transpile_smoke --target jax/mlx --model ResNet-18 | repo root | examples deps | model-zoo smoke | ResNet-18 transpiled for both targets | PASS | baseline/final | OPTIONAL | model-zoo/JAX/MLX | no | fail blocks only model-zoo release |
| V11 | uv run python scripts/check_residue.py | repo root | git tracked tree | retired low-level/web product/config residue | no true hits | PASS | baseline/final | REQUIRED | repo | no | any true product/config hit blocks |
| V12 | uv run python scripts/check_context_pack.py | repo root | Python stdlib | context pack consistency | indexes current and parse | PASS after index generation | final | REQUIRED | context | yes | fail blocks |
| V13 | uv run python scripts/check_docs_links.py | repo root | Python stdlib | docs links | no missing repo-relative Markdown targets | PASS | final | REQUIRED | docs | no | fail blocks |
| V14 | uv run python scripts/check_operator_docs.py | repo root | project import | operator docs | generated docs/operators.md current | PASS | final | REQUIRED | operators | no | fail blocks |
| V15 | uv run python scripts/check_generated_code.py | repo root | dev env | generated JAX/MLX compile/import/runtime | pass | PASS | final | REQUIRED | JAX/MLX | no | fail blocks |
| V16 | uv run python scripts/check_package_contents.py | repo root | uv build already run | package contents | wheel package-only; sdist excludes examples/tests/agents/plans/scripts | PASS | final | REQUIRED | package | no | fail blocks |

## Validation Tiers

- Tier 0: imports, CLI help, config loading, catalog listing.
- Tier 1: unit/contract tests.
- Tier 2: generated code parses, compiles, imports, and has no unresolved placeholders.
- Tier 3: numerical parity on practical small models.
- Tier 4: named large-model generation using available fixtures/cache/assets.
- Tier 5: full large-model inference when already supported and practical.

## Final Gate Result

Final execution ran: ruff check, ruff format check, ty, default pytest, build, package contents, clean wheel import, installed CLI help, CLI JAX and MLX smoke with runtime output checks, generated-code parse/import/runtime for JAX and MLX, ResNet-18 model-zoo JAX and MLX smoke, named-model gates available in the environment, context-pack lint, docs link check, operator-doc check, residue grep, and `git diff --check`.
