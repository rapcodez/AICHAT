# Code Review

## Tests
- `python -m compileall .` (pass)

## Findings
1. **Model loading is eager and unbounded**
   - `build_model()` loads `AutoTokenizer` and `AutoModelForCausalLM` at import time (`generator = build_model()`), which increases app startup time and memory usage on Spaces CPU. Consider lazy-loading inside the request handler or caching via `gradio.load` with smaller quantization to keep boot times predictable. Relevant code: `build_model` definition and global `generator` initialization.
2. **LLM device placement is implicit**
   - The text-generation pipeline is created without `device_map` or `torch_dtype` hints, so it defaults to CPU and full precision. Explicitly setting `device_map="cpu"`/`torch_dtype=torch.float16` (if supported) or using `pipeline(..., model=model.to("cpu"))` can avoid unexpected GPU expectations and reduce memory footprint.
3. **Pipeline sampling parameters are fixed**
   - The generator uses `do_sample=True`, `temperature=0.3`, `top_p=0.9` for all requests. Exposing these as configurable (or switching to deterministic decoding for business answers) would yield more consistent responses and easier debugging.
