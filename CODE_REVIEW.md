# Code Review

## Tests
- `python -m compileall .` (pass)
- `pytest` (skipped: numpy/pandas not available in the offline container)

## Findings and fixes
1. **Gradio compatibility**
   - The Chatbot uses default tuple-based history (no `type` or other 6.1.8-only arguments) and the streaming helper now yields `(user, assistant)` tuples for older 6.1.0 runtimes.
2. **Streaming state updates**
   - The generator still yields interim tokens but now rewrites the last tuple each step to keep history consistent before persisting assistant memory.
3. **Model loading and device control**
   - The LLM pipeline lazy-loads on first use, keeps an explicit CPU device and `torch.float32` dtype, and caches the generator to trim startup overhead.
4. **LLM-free testing**
   - When `DISABLE_LLM=true`, `respond_with_model` returns a stubbed reply so automated tests can run without downloading models.
