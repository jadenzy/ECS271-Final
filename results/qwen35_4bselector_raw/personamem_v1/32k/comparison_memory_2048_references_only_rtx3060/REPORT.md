# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens)

Status: **complete**. Scored 589/589 selected PersonaMem-v1 32K questions from 20 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **60.61%** (357/589).
- 95% confidence interval, clustered by persona: **56.42%–65.07%**.
- Mean correct-option NLL (four-label distribution): **1.1354**.
- Average runtime per question: **1.764 seconds**.
- Average inference request time per question: **1.642 seconds**.
- Average GPU memory per question (device-wide): **3,279.3 MiB** (589/589 questions measured).
- Average Qwen model server RAM per question (working set): **8,600.2 MiB** (589/589 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **2,789 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| generalizing_to_new_scenarios | 57 | 61.40% | 0.9294 |
| provide_preference_aligned_recommendations | 55 | 78.18% | 0.5623 |
| recall_user_shared_facts | 129 | 58.91% | 1.1585 |
| recalling_facts_mentioned_by_the_user | 17 | 52.94% | 1.0144 |
| recalling_the_reasons_behind_previous_updates | 99 | 83.84% | 0.4598 |
| suggest_new_ideas | 93 | 29.03% | 2.2711 |
| track_full_preference_evolution | 139 | 60.43% | 1.1612 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–589 | 589 | 1.764 | 3279.289 | 8600.223 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
