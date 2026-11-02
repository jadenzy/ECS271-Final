# Frozen Qwen baseline: full visible history

Status: **complete**. Scored 111/111 selected LongBench-v2 32K questions from 111 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **45.05%** (50/111).
- 95% confidence interval, clustered by persona: **36.04%–54.05%**.
- Mean correct-option NLL (four-label distribution): **2.0455**.
- Average runtime per question: **2.749 seconds**.
- Average inference request time per question: **2.714 seconds**.
- Average GPU memory per question (device-wide): **5,525.0 MiB** (111/111 questions measured).
- Average Qwen model server RAM per question (working set): **10,569.9 MiB** (111/111 questions measured).
- Allocated context: **34,560 tokens**; longest complete prompt: **34,440 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| Code Repository Understanding | 2 | 50.00% | 0.5550 |
| Long In-context Learning | 7 | 42.86% | 1.3935 |
| Long Structured Data Understanding | 1 | 0.00% | 8.8521 |
| Long-dialogue History Understanding | 8 | 50.00% | 1.6232 |
| Multi-Document QA | 32 | 40.62% | 1.9696 |
| Single-Document QA | 61 | 47.54% | 2.1528 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–111 | 111 | 2.749 | 5524.978 | 10569.910 |

The frozen Qwen3.5-4B-Q4_K_M model receives a complete chronological transcript of all visible source messages, with role labels and all repeated system persona descriptions, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. History is never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
