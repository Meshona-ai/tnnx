# Second-Pass Review

## Executive Assessment

`tnnx` has a compact core and strong small-model/backend parity coverage, but the repository is not ready for serious ML engineers without cleanup. The biggest issues are validation reliability, model support claim clarity, one backend parity gap, stale docs, and unowned research/example bulk.

## Correctness Findings

- F004: MLX lacks JAX topological scheduling.
- F006: public config fields look behavioral but are metadata-only or unused.
- F007: ONNX shape inference failures are swallowed.
- F008: prune pass is advertised but no-op.
- F009: IR validation lacks several invariants.
- F017: external data/download/subprocess trust boundaries need tests.

## Architecture Findings

The direct architecture is a strength. Do not add a plugin framework. The useful architecture work is narrower: scheduler sharing or parity, IR invariants, pruning truthfulness, and backend helper shrink after tests exist.

## Backend Parity Findings

JAX and MLX semantic dispatch both cover 69 schema ops. JAX has topology scheduling and tests; MLX does not. FLUX docs explicitly say MLX work is future work, so do not claim FLUX MLX support yet.

## Test And CI Findings

- F001 and F002 make default pytest unreliable across clean environments.
- F003: no CI exists.
- Skip reasons are mostly explicit, but marker/env policy is not centralized.

## Packaging And Dependency Findings

Runtime dependencies are small: numpy and onnx. Dev/examples dependencies are heavy and need clearer grouping. Python 3.14-only may be intentional but is a public adoption risk. The wheel is lean; the sdist is broad.

## Docs And Demo Readiness Findings

Docs are useful but stale in key places: missing demo runbook, stale operator table, and support-tier ambiguity. Real-model docs should distinguish always-on synthetic gates from asset/network/env-gated real gates.

## LLM Context-Pack Findings

No prior context pack existed. This commit creates one with Markdown pages and machine indexes.

## Dead Code And LOC Findings

Research scripts, historical FLUX notes, unused public helpers, and duplicated codegen helper patterns are all shrink candidates. None should be deleted from Ponytail output alone; each needs reachability and retained-behavior gates.

## Low-Level Residue Findings

Prescribed residue grep returned no FPGA/HLS/RTL/C/C++/CMake/native-backend/web API hits. Keep the grep as a guard.

## Ponytail Audit Findings With Independent Evaluation

- `delete: scripts/research/*`. Replacement: nothing or docs/benchmarks only after reachability review. [scripts/research/**]
- `yagni: src/tnnx/utils/* exports unused base error and stable_dict`. Replacement: delete if public-use grep stays empty. [src/tnnx/utils/**]
- `yagni: src/tnnx/codegen/common.py exposes NameGenerator/order_nodes_for_emission unused by backends`. Replacement: delete or replace with real shared scheduler after MLX parity. [src/tnnx/codegen/common.py]
- `shrink: duplicated attr/slot parsing in JAX and MLX emitters`. Replacement: one small shared helper after snapshot coverage. [src/tnnx/codegen/*_codegen.py]
- `delete: historical FLUX handoff prose`. Replacement: compact current-state docs and context log. [examples/flux/jax_backend_notes.md; examples/flux/submodule_plan.md]
- `shrink: sdist ships examples/research/tests/agents by default`. Replacement: explicit package-content policy. [pyproject.toml]
- `yagni: metadata-only CompileConfig knobs`. Replacement: implement or remove/document metadata-only. [src/tnnx/config.py]

Independent evaluation: all Ponytail items are hypotheses. The plan requires reachability checks, characterization tests, and context updates before any deletion/shrink.

Net estimate: -15 to -35 files, -2500 to -6500 LOC, -0 to -3 optional/dev dependencies possible after verification.

## Risk Assessment

Highest risk is making the suite green by hiding real model regressions. Second highest is deleting example/research code that is used as informal demo evidence. The plan orders characterization and validation before deletion.
