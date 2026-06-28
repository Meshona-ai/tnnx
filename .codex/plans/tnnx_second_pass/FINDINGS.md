# Findings

## F001: Default dev environment does not include examples deps required by default pytest

- severity: HIGH
- category: TEST-GAP/PACKAGING
- evidence: `uv run pytest -q` after `uv run`/dev install failed 13 FLUX tests on missing `onnxscript`/`diffusers`; `uv sync --dev --group examples` reduced this to the two Whisper ffmpeg failures.
- affected files: pyproject.toml; tests/integration/test_flux_*; examples/flux/source.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T01.
- confidence: HIGH
- linked execution task: T01
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: `uv sync --dev`; `uv run pytest -q` passes or example-only tests are explicitly gated.

## F002: Default pytest enables host-dependent Whisper MLX audio path

- severity: HIGH
- category: REPRODUCIBILITY/TEST-GAP
- evidence: `pyproject.toml` sets `RUN_MLX_E2E=1`; `uv run pytest -q` fails two Whisper tests because local `ffmpeg -version` aborts on missing `libx265.215.dylib`.
- affected files: pyproject.toml; tests/integration/test_whisper_audio_real_mlx.py; tests/integration/test_whisper_audio_transpile_source.py; examples/whisper_audio/whisper_hf_tiny_source.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T02.
- confidence: HIGH
- linked execution task: T02
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Default `uv run pytest -q` passes without working ffmpeg; opt-in Whisper gate still runs when env and ffmpeg are healthy.

## F003: No CI workflow exists

- severity: HIGH
- category: CI
- evidence: `.github/` is absent; only local `scripts/check.sh` defines lint/type/test checks.
- affected files: .github/; scripts/check.sh
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T03.
- confidence: HIGH
- linked execution task: T03
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: A CI workflow runs ruff, format, ty, unit/default tests, package build, and residue grep.

## F004: MLX emitter does not share JAX topological scheduling

- severity: HIGH
- category: BACKEND-PARITY/BUG
- evidence: JAX emits `for node in _scheduled_nodes(ir)` while MLX emits `for node in ir.nodes`; JAX has out-of-order SPLIT tests, MLX does not.
- affected files: src/tnnx/codegen/jax_codegen.py; src/tnnx/codegen/mlx_codegen.py; tests/unit/test_jax_codegen_topology_contract.py; tests/integration/test_jax_topology_runtime_parity.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T04.
- confidence: HIGH
- linked execution task: T04
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Matching MLX out-of-order graph compile/runtime parity test passes.

## F005: No durable LLM context pack existed before this audit

- severity: HIGH
- category: LLM-CONTEXT-GAP
- evidence: No pre-existing `llms.txt` or `docs/llm_context/` was present at review base.
- affected files: llms.txt; docs/llm_context/**
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T00.
- confidence: HIGH
- linked execution task: T00
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Context pack lint passes; indexes parse.

## F006: Public CompileConfig fields are metadata-only or unused

- severity: MEDIUM
- category: API-ERGONOMICS
- evidence: `deterministic`, `emit_shape_asserts`, and `opset` are serialized to metadata but not used to change ingest/codegen behavior.
- affected files: src/tnnx/config.py; src/tnnx/api.py; src/tnnx/cli.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T05.
- confidence: MEDIUM
- linked execution task: T05
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Public config docs/tests prove each field is behavioral, metadata-only, or removed.

## F007: ONNX shape inference failures are swallowed

- severity: MEDIUM
- category: BUG
- evidence: `load_onnx_to_ir` catches all exceptions from `onnx.shape_inference.infer_shapes` and continues silently.
- affected files: src/tnnx/ingest/onnx_reader.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T06.
- confidence: MEDIUM
- linked execution task: T06
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: A malformed shape-inference fixture records or raises an actionable diagnostic without breaking supported models.

## F008: Advertised prune pass is a no-op

- severity: MEDIUM
- category: ARCHITECTURE
- evidence: `prune_dead_nodes` returns `ir`; default passes include `prune`; docs list prune as a current compiler pass.
- affected files: src/tnnx/passes/prune.py; src/tnnx/config.py; docs/architecture.md
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T07.
- confidence: MEDIUM
- linked execution task: T07
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Dead-node characterization test either proves implemented pruning or docs/config stop advertising behavior.

## F009: IR validation lacks producer/topology/duplicate-output invariants

- severity: MEDIUM
- category: IR-INVARIANT
- evidence: `validate_graph` checks references and arity but not duplicate producers, topological order, graph output provenance, or cycles.
- affected files: src/tnnx/ir/schema.py; src/tnnx/ir/types.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T08.
- confidence: MEDIUM
- linked execution task: T08
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: IR invariant tests fail before fix and pass after fix; JAX/MLX codegen stays deterministic.

## F010: Docs reference missing demo runbook

- severity: MEDIUM
- category: DOC-GAP
- evidence: `docs/README.md` lists `demo-runbook.md`, but no such file exists.
- affected files: docs/README.md
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T09.
- confidence: MEDIUM
- linked execution task: T09
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Docs link check has no missing repo-relative Markdown targets.

## F011: README supported-op table is stale

- severity: MEDIUM
- category: DOC-GAP
- evidence: README lists a small v0 table, while schema/codegen cover 69 semantic ops.
- affected files: README.md; src/tnnx/ir/schema.py; src/tnnx/codegen/*.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T10.
- confidence: MEDIUM
- linked execution task: T10
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Generated/operator-derived table matches schema, ingest map, JAX, MLX, and tests.

## F012: Model support claims mix ready, experimental, planned, and environment-gated paths

- severity: MEDIUM
- category: DOC-GAP/DEMO
- evidence: Model-zoo readiness prints planned/experimental/ready; Qwen real path may download; FLUX real path needs `TNNX_FLUX_SNAPSHOT`; Whisper needs HF cache and ffmpeg.
- affected files: examples/README.md; examples/model_zoo/*; examples/qwen/README.md; examples/flux/*; examples/whisper_audio/*
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T11.
- confidence: MEDIUM
- linked execution task: T11
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Every named model has a tiered gate and docs mark unvalidated or environment-blocked paths explicitly.

## F013: sdist ships examples, research scripts, tests, agent files, and audio asset

- severity: MEDIUM
- category: PACKAGING
- evidence: `uv build` wheel has 30 package entries, but sdist has 215 entries including `.agents`, `scripts/research`, `examples/terminator.mp3`, and tests.
- affected files: pyproject.toml; examples/**; scripts/research/**; .agents/**; tests/**
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T12.
- confidence: MEDIUM
- linked execution task: T12
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: sdist/wheel contents are intentional and documented; unnecessary files excluded.

## F014: Research scripts are large, unowned, and not in default validation

- severity: MEDIUM
- category: DEAD-CODE/VERIFY
- evidence: `scripts/research` is 11 files / 3699 LOC; docs do not define which scripts are retained product gates.
- affected files: scripts/research/**
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T13.
- confidence: MEDIUM
- linked execution task: T13
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Each research script is kept with an owner/gate, moved to docs, or deleted after reachability check.

## F015: JAX and MLX codegen duplicate helper parsing and many ONNX helper templates

- severity: MEDIUM
- category: LOC-REDUCTION
- evidence: `jax_codegen.py` is 1334 LOC and `mlx_codegen.py` is 1229 LOC with repeated attr parsers, slot handling, and operator-helper structure.
- affected files: src/tnnx/codegen/jax_codegen.py; src/tnnx/codegen/mlx_codegen.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T14.
- confidence: MEDIUM
- linked execution task: T14
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Snapshots and runtime parity pass after one narrow shared-helper extraction; no backend behavior drift.

## F016: Python requirement is pinned to 3.14 only

- severity: MEDIUM
- category: PACKAGING
- evidence: Clean wheel smoke without `--python 3.14` failed because uv chose Python 3.11; source mostly uses syntax available before 3.14.
- affected files: pyproject.toml; src/**/*.py; tests/unit/test_environment_contract.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T15.
- confidence: MEDIUM
- linked execution task: T15
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Decision recorded: either tested wider Python range or docs/CI clearly require 3.14.

## F017: External data, downloads, .env token, and subprocess boundaries need explicit validation

- severity: MEDIUM
- category: SECURITY
- evidence: ONNX load uses default external-data behavior; FLUX reads `HUGGING_FACE_TOKEN` from `.env`; Whisper shells out to `ffmpeg`; Qwen may download unless local-only.
- affected files: src/tnnx/ingest/onnx_reader.py; examples/flux/runtime_env.py; examples/whisper_audio/whisper_hf_tiny_source.py; examples/qwen/*.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T16.
- confidence: MEDIUM
- linked execution task: T16
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Security characterization tests cover external data paths, token handling, download opt-in/offline behavior, and ffmpeg failure messages.

## F018: Small public helpers appear unused outside their own tests

- severity: LOW
- category: VERIFY/LOC-REDUCTION
- evidence: `TranspilerError`, `stable_dict`, `coerce_attr`, and `NameGenerator/order_nodes_for_emission` have no product callers; some are exported from package submodules.
- affected files: src/tnnx/utils/**; src/tnnx/ir/schema.py; src/tnnx/codegen/common.py; tests/unit/test_name_stability.py; tests/unit/test_topological_emit_order.py
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T17.
- confidence: MEDIUM
- linked execution task: T17
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Public-use grep, docs, imports, and tests prove each helper is retained or removed safely.

## F019: Low-level residue search is currently clean

- severity: LOW
- category: RESIDUE
- evidence: Prescribed path, terminology, and backend-claim greps returned no FPGA/HLS/RTL/C/CMake/native-backend hits.
- affected files: repo-wide
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T18.
- confidence: HIGH
- linked execution task: T18
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Residue grep is added as a reusable validation gate.

## F020: No context-pack maintenance lint existed

- severity: LOW
- category: DOC-GAP
- evidence: No pre-existing page index, log, code index, symbol index, or graph edge file existed.
- affected files: docs/llm_context/**
- retained behavior at risk: ONNX ingest, IR, backend codegen, CLI/API, examples, or validation gates as applicable.
- recommended action: execute T19.
- confidence: MEDIUM
- linked execution task: T19
- context-pack pages to update: `known_limits.md`, `validation.md`, `index.md`, and topic-specific pages.
- validation gate: Context-pack lint lists no orphan pages and machine indexes parse.
