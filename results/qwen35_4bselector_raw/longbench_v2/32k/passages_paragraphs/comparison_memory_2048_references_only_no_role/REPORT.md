# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens)

Status: **complete**. Scored 111/111 selected LongBench-v2 32K questions from 111 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **39.64%** (44/111).
- 95% confidence interval, clustered by persona: **30.63%–48.65%**.
- Mean correct-option NLL (four-label distribution): **1.8766**.
- Average runtime per question: **0.378 seconds**.
- Average inference request time per question: **0.323 seconds**.
- Average GPU memory per question (device-wide): **4,058.4 MiB** (111/111 questions measured).
- Average Qwen model server RAM per question (working set): **9,933.0 MiB** (111/111 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **2,748 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| Code Repository Understanding | 2 | 50.00% | 0.7171 |
| Long In-context Learning | 7 | 42.86% | 1.8526 |
| Long Structured Data Understanding | 1 | 0.00% | 6.0082 |
| Long-dialogue History Understanding | 8 | 12.50% | 1.6706 |
| Multi-Document QA | 32 | 37.50% | 2.0676 |
| Single-Document QA | 61 | 44.26% | 1.7765 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–111 | 111 | 0.378 | 4058.360 | 9932.987 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
