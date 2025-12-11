# Code Review

## Tests
- `python -m compileall .` (pass)

## Findings and fixes
1. **Gradio compatibility**
   - The Chatbot component now uses `type="messages"` and the streaming helper emits `{"role", "content"}` dictionaries to satisfy the runtime’s expected format.
2. **Streaming state updates**
   - The generator still yields interim tokens but now copies prior UI history safely and appends turns in the new message format before persisting assistant memory.
3. **Model loading and device control**
   - The LLM pipeline lazy-loads on first use, keeps an explicit CPU device and `torch.float32` dtype, and caches the generator to trim startup overhead.
