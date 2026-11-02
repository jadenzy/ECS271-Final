# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens)

Status: **complete**. Scored 1,242/1,242 selected PersonaMem-v2 32K questions from 50 test personas.
Selection: **compact**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **39.37%** (489/1242).
- 95% confidence interval, clustered by persona: **36.63%–42.01%**.
- Mean correct-option NLL (four-label distribution): **1.6842**.
- Average runtime per question: **0.437 seconds**.
- Average inference request time per question: **0.337 seconds**.
- Average GPU memory per question (device-wide): **5,061.5 MiB** (1242/1242 questions measured).
- Average Qwen model server RAM per question (working set): **10,031.7 MiB** (1242/1242 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **3,046 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 226 | 45.58% | 1.4308 |
| ask_to_forget | 253 | 28.85% | 2.0074 |
| health_and_medical_conditions | 137 | 45.26% | 1.5643 |
| neutral_preferences | 213 | 39.91% | 1.5311 |
| sensitive_info | 123 | 39.02% | 1.9952 |
| stereotypical_pref | 142 | 44.37% | 1.4351 |
| therapy_background | 148 | 37.16% | 1.8308 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1242 | 1242 | 0.437 | 5061.462 | 10031.719 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
