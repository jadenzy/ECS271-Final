# Selector packing optimization

All 1,242 V2 test and 589 V1 comparison questions reproduced their saved selected message IDs, prompt SHA-256, prompt token counts, and memory token counts.

The original HTTP packing loop was additionally compared with the optimized loop on six V2, six V1, and three LoCoMo questions. All selections, prompt text, counts, and skip counts matched.

| Metric | Original | Optimized |
| --- | ---: | ---: |
| Mean packing seconds, 15-question HTTP comparison | 2.8106 | 0.2220 |
| Mean packing HTTP requests | 530.7 | 2.0 |

The optimized path uses the pinned native GGUF vocabulary, the GGUF Jinja template, cached formatted source blocks, and exact token-count caching at Qwen35 pre-tokenizer boundaries. Every final prompt is checked against the server before scoring. The prepared full and bare prompts are reused by scoring. No model training, answer-accuracy tuning, or full evaluation rerun was performed.

Timing measures packing only, excluding initial vocabulary/model loading and Qwen inference. Per-question measurements and complete equivalence results are in verification.json.

LightGBM training, tuning, and prediction now explicitly use CPU with four threads; Qwen remains configured for GPU inference. Explicit CPU prediction reproduced the existing scores bit for bit on 1,000 real feature rows. All 131 repository tests pass; see tests.log.
