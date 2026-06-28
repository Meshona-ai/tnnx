# Model Zoo Smoke Transpile Examples

This folder keeps the model-zoo examples in one place instead of scattering one script per model.

Use the shared smoke runner:

```bash
uv run python -m examples.model_zoo.transpile_smoke --list
uv run python -m examples.model_zoo.transpile_smoke \
  --target jax \
  --model ResNet-18 \
  --output-dir /tmp/examples/model-zoo
uv run python -m examples.model_zoo.transpile_smoke --target mlx --model "Small BERT variants"
```

Behavior:
- `ready`: intended to export and transpile with the shared local wrapper.
- `experimental`: implemented, but not treated as a stable no-surprises path yet.
- `planned`: intentionally listed but not auto-exported from this workspace.

Artifacts go under the `--output-dir` you pass (default: `examples/out/model_zoo/`).
