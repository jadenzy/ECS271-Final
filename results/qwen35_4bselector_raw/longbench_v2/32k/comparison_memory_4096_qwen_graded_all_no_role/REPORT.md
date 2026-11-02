# Frozen Qwen baseline: LightGBM selector (4,096 Qwen memory tokens)

Status: **complete**. Scored 111/111 selected LongBench-v2 32K questions from 111 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **39.64%** (44/111).
- 95% confidence interval, clustered by persona: **30.63%–48.65%**.
- Mean correct-option NLL (four-label distribution): **1.9058**.
- Average runtime per question: **0.665 seconds**.
- Average inference request time per question: **0.590 seconds**.
- Average GPU memory per question (device-wide): **4,908.9 MiB** (111/111 questions measured).
- Average Qwen model server RAM per question (working set): **10,147.7 MiB** (111/111 questions measured).
- Allocated context: **5,096 tokens**; longest complete prompt: **4,789 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| Code Repository Understanding | 2 | 50.00% | 0.6440 |
| Long In-context Learning | 7 | 28.57% | 1.9096 |
| Long Structured Data Understanding | 1 | 0.00% | 7.5627 |
| Long-dialogue History Understanding | 8 | 12.50% | 1.6224 |
| Multi-Document QA | 32 | 37.50% | 2.0792 |
| Single-Document QA | 61 | 45.90% | 1.8002 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–111 | 111 | 0.665 | 4908.946 | 10147.658 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 4,096 Qwen memory tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
