# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens)

Status: **complete**. Scored 1,242/1,242 selected PersonaMem-v2 32K questions from 50 test personas.
Selection: **compact**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **37.84%** (470/1242).
- 95% confidence interval, clustered by persona: **35.00%–40.56%**.
- Mean correct-option NLL (four-label distribution): **1.7243**.
- Average runtime per question: **4.433 seconds**.
- Average inference request time per question: **1.920 seconds**.
- Average GPU memory per question (device-wide): **3,447.4 MiB** (1242/1242 questions measured).
- Average Qwen model server RAM per question (working set): **7,441.4 MiB** (1242/1242 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **3,041 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 226 | 42.48% | 1.5071 |
| ask_to_forget | 253 | 31.23% | 1.9763 |
| health_and_medical_conditions | 137 | 40.88% | 1.6477 |
| neutral_preferences | 213 | 39.44% | 1.5904 |
| sensitive_info | 123 | 38.21% | 1.9885 |
| stereotypical_pref | 142 | 40.14% | 1.4639 |
| therapy_background | 148 | 34.46% | 1.9194 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1242 | 1242 | 4.433 | 3447.398 | 7441.400 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
