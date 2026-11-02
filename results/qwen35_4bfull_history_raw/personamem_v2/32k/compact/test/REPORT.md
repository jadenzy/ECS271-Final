# Frozen Qwen baseline: full visible history

Status: **complete**. Scored 1,242/1,242 selected PersonaMem-v2 32K questions from 50 test personas.
Selection: **compact**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **37.36%** (464/1242).
- 95% confidence interval, clustered by persona: **34.26%–40.34%**.
- Mean correct-option NLL (four-label distribution): **1.8304**.
- Average runtime per question: **0.410 seconds**.
- Average inference request time per question: **0.349 seconds**.
- Average GPU memory per question (device-wide): **5,099.9 MiB** (1242/1242 questions measured).
- Average Qwen model server RAM per question (working set): **10,049.5 MiB** (1242/1242 questions measured).
- Allocated context: **41,216 tokens**; longest complete prompt: **37,894 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 226 | 39.82% | 1.6426 |
| ask_to_forget | 253 | 23.72% | 2.1384 |
| health_and_medical_conditions | 137 | 37.96% | 1.7741 |
| neutral_preferences | 213 | 41.31% | 1.6713 |
| sensitive_info | 123 | 39.02% | 2.0977 |
| stereotypical_pref | 142 | 47.18% | 1.4330 |
| therapy_background | 148 | 39.86% | 2.0306 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1242 | 1242 | 0.410 | 5099.940 | 10049.530 |

The frozen Qwen3.5-4B-Q4_K_M model receives a complete chronological transcript of all visible source messages, with role labels and all repeated system persona descriptions, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. History is never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
