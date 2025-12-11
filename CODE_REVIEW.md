# Code Review

## Tests
- `python -m compileall .` (pass)

## Findings and fixes
1. **Gradio compatibility**
   - `show_copy_button` on `Chatbot` is unsupported in the runtime version and blocked startup. The parameter was removed and custom CSS is now passed via `Blocks(css=...)` to stay backward compatible with older Gradio builds.
2. **Streaming state updates**
   - The generator previously yielded tokens without committing the final turn to session history, so memory would reset on the next request. The streaming helper now emits a final update with persisted history.
3. **Model loading and device control**
   - The LLM pipeline now lazy-loads on first use, pins `device_map="cpu"` with an explicit `torch_dtype`, and keeps the generator cached to trim startup overhead and avoid unexpected device selection.
