# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens)

Status: **complete**. Scored 1/1 selected PersonaMem-v2 32K questions from 1 test personas.
Selection: **compact**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **100.00%** (1/1).
- 95% confidence interval, clustered by persona: not estimable from a single persona.
- Mean correct-option NLL (four-label distribution): **0.1242**.
- Average runtime per question: **2.222 seconds**.
- Average inference request time per question: **1.116 seconds**.
- Average GPU memory per question (device-wide): **3,468.0 MiB** (1/1 questions measured).
- Average Qwen model server RAM per question (working set): **3,087.3 MiB** (1/1 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **2,419 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| stereotypical_pref | 1 | 100.00% | 0.1242 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1 | 1 | 2.222 | 3468.000 | 3087.330 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
