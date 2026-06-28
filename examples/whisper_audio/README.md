# Whisper Tiny Through The Transpiler

This example uses the locally cached Hugging Face weights for `openai/whisper-tiny`,
implements the model from source in PyTorch-style code, exports it to ONNX, transpiles
that ONNX graph into generated MLX code, and then runs the generated MLX module for
speech-to-text.

Input audio defaults to `examples/terminator.mp3`.

## Run

```bash
uv run python -m examples.whisper_audio.transpile_and_transcribe \
  --audio examples/terminator.mp3 \
  --decode-tokens 64 \
  --max-new-tokens 48 \
  --output-dir /tmp/examples/whisper-audio
```

## Outputs

- `<output-dir>/openai_whisper_tiny_64.onnx`
- `<output-dir>/generated_mlx_64/model_mlx.py`
- `<output-dir>/generated_mlx_64/weights.npz`
- `<output-dir>/terminator.transcript.txt`
- `<output-dir>/terminator.transcript.json`
