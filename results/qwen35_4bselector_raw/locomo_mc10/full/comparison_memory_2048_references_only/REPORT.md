# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens)

Status: **complete**. Scored 1,986/1,986 selected LoCoMo-MC10 FULL questions from 10 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **60.42%** (1200/1986).
- 95% confidence interval, clustered by conversation: **57.86%–63.13%**.
- Mean correct-option NLL (ten-label distribution): **1.3642**.
- Average runtime per question: **0.446 seconds**.
- Average inference request time per question: **0.323 seconds**.
- Average GPU memory per question (device-wide): **4,600.1 MiB** (1986/1986 questions measured).
- Average Qwen model server RAM per question (working set): **8,638.5 MiB** (1986/1986 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **2,653 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| adversarial | 446 | 38.57% | 2.3923 |
| multi_hop | 321 | 35.83% | 2.1603 |
| open_domain | 841 | 80.62% | 0.6157 |
| single_hop | 282 | 67.02% | 0.9762 |
| temporal_reasoning | 96 | 47.92% | 1.6223 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1986 | 1986 | 0.446 | 4600.100 | 8638.505 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 10 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
