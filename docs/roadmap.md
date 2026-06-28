# Roadmap

This document captures the next layer of work that makes `tnnx` more than a
portable transpiler.

## Current Baseline

Today the project already has:

- deterministic ONNX ingest
- a custom IR
- schema validation
- prune, normalize, and shape propagation passes
- code generation for JAX and MLX
- example-backed proof on model families beyond toy layers

That is a strong compiler skeleton. The next step is to make the optimization
layer much deeper.

## Optimization Priorities

### 1. Graph simplification

- constant folding
- static precomputation of shape-only or compile-time subgraphs
- duplicate elimination and common subexpression elimination
- more aggressive dead code pruning

### 2. Inference quality

- richer shape inference across more operators
- stronger dtype propagation
- layout and broadcasting reasoning as first-class compiler information
- safer handling of dynamic and partially known dimensions

### 3. Backend-aware lowering

- cost-model guided lowering choices
- backend-specific canonicalization
- fusion where it improves artifact quality or runtime efficiency
- memory planning and buffer reuse for constrained targets

### 4. Hardware expansion

The larger thesis is not limited to the current JAX and MLX output.

The long-term goal is a target contract that lets new backends plug into the
pipeline cleanly:

- TPU-oriented lowering
- NPU and edge-accelerator lowering
- broader backend coverage
- future custom accelerator backends

The honest wording for now is:

> `tnnx` is building toward zero-effort deployment across heterogeneous
> accelerators.

That is a roadmap claim, not a statement that every backend already exists.

## Verification Priorities

Optimization is only useful if semantics stay trustworthy.

That means continuing to invest in:

- parity checks against source models
- snapshot stability for generated artifacts
- backend-to-backend comparisons where possible
- deterministic graph hashing and artifact identity

## Demo Closing Soundbite

Use this line when you want to connect the current implementation to the
ambition:

> I have already built the compiler boundary. The next step is to make it the
> optimization and lowering layer that lets new AI hardware run real models
> without bespoke rewrite work.
