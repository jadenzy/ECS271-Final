# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens, stopping 1.5 below the top score after 1,000 tokens)

Status: **complete**. Scored 1,242/1,242 selected PersonaMem-v2 32K questions from 50 test personas.
Selection: **compact**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **39.86%** (495/1242).
- 95% confidence interval, clustered by persona: **36.97%–42.77%**.
- Mean correct-option NLL (four-label distribution): **1.6980**.
- Average runtime per question: **0.282 seconds**.
- Average inference request time per question: **0.266 seconds**.
- Average GPU memory per question (device-wide): **4,510.8 MiB** (1242/1242 questions measured).
- Average Qwen model server RAM per question (working set): **11,192.4 MiB** (1242/1242 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **2,830 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 226 | 45.13% | 1.4953 |
| ask_to_forget | 253 | 30.04% | 1.9785 |
| health_and_medical_conditions | 137 | 48.91% | 1.5112 |
| neutral_preferences | 213 | 41.31% | 1.5376 |
| sensitive_info | 123 | 39.02% | 2.0536 |
| stereotypical_pref | 142 | 42.96% | 1.3934 |
| therapy_background | 148 | 35.81% | 1.9286 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1242 | 1242 | 0.282 | 4510.783 | 11192.370 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, stopping 1.5 below the top score after 1,000 tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
