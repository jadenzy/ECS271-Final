# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens, stopping 1.5 below the top score after 1,000 tokens)

Status: **complete**. Scored 1,242/1,242 selected PersonaMem-v2 32K questions from 50 test personas.
Selection: **compact**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **40.18%** (499/1242).
- 95% confidence interval, clustered by persona: **37.43%–42.89%**.
- Mean correct-option NLL (four-label distribution): **1.6801**.
- Average runtime per question: **0.291 seconds**.
- Average inference request time per question: **0.275 seconds**.
- Average GPU memory per question (device-wide): **4,548.5 MiB** (1242/1242 questions measured).
- Average Qwen model server RAM per question (working set): **11,197.8 MiB** (1242/1242 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **2,948 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 226 | 46.90% | 1.4491 |
| ask_to_forget | 253 | 32.02% | 1.9302 |
| health_and_medical_conditions | 137 | 42.34% | 1.6008 |
| neutral_preferences | 213 | 43.19% | 1.4835 |
| sensitive_info | 123 | 38.21% | 2.0887 |
| stereotypical_pref | 142 | 42.25% | 1.3850 |
| therapy_background | 148 | 37.16% | 1.9054 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1242 | 1242 | 0.291 | 4548.501 | 11197.800 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, stopping 1.5 below the top score after 1,000 tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
