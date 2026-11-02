# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens, stopping 1.5 below the top score after 1,000 tokens)

Status: **complete**. Scored 1,986/1,986 selected LoCoMo-MC10 FULL questions from 10 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **66.26%** (1316/1986).
- 95% confidence interval, clustered by conversation: **63.43%–69.30%**.
- Mean correct-option NLL (ten-label distribution): **1.1559**.
- Average runtime per question: **0.245 seconds**.
- Average inference request time per question: **0.233 seconds**.
- Average GPU memory per question (device-wide): **4,247.6 MiB** (1986/1986 questions measured).
- Average Qwen model server RAM per question (working set): **11,260.1 MiB** (1986/1986 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **2,515 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| adversarial | 446 | 38.34% | 2.2363 |
| multi_hop | 321 | 45.17% | 1.8393 |
| open_domain | 841 | 87.04% | 0.4380 |
| single_hop | 282 | 74.11% | 0.7680 |
| temporal_reasoning | 96 | 61.46% | 1.2789 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1986 | 1986 | 0.245 | 4247.641 | 11260.064 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, stopping 1.5 below the top score after 1,000 tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 10 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
