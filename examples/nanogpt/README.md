# NanoGPT Examples

This folder contains two separate examples:

1. `train_nanogpt_tiny.py`
   This is the toy example. It trains a tiny custom checkpoint, caches that checkpoint,
   transpiles the tiny model to JAX, and runs generated JAX inference.

   ```bash
   uv run python -m examples.nanogpt.train_nanogpt_tiny \
     --prompt "hello jax" \
     --max-new-tokens 32 \
     --output-dir /tmp/examples/nanogpt-demo
   ```

   It writes the toy checkpoint as `nanogpt_tiny_train_checkpoint.pt`
   inside the output directory.

2. `infer_gpt2_from_transformers_jax.py`
   This is the pretrained example. It loads a GPT-2 checkpoint directly with
   `GPT2LMHeadModel.from_pretrained(...)`, exports it to ONNX, transpiles it to JAX,
   and runs greedy generation with the generated JAX module.

   ```bash
   uv run python -m examples.nanogpt.infer_gpt2_from_transformers_jax \
     --model-id sshleifer/tiny-gpt2 \
     --prompt "Hello from tnnx" \
     --output-dir /tmp/examples/gpt2-jax
   ```

   The default model id is `sshleifer/tiny-gpt2` so export and transpile stay practical.
   Pass `--model-id gpt2` if you want the full base GPT-2 checkpoint.
   Add `--local-files-only` if you want to use only the local Hugging Face cache.
