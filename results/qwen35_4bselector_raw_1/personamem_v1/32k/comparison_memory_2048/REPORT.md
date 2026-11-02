# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens)

Status: **complete**. Scored 589/589 selected PersonaMem-v1 32K questions from 20 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **59.25%** (349/589).
- 95% confidence interval, clustered by persona: **55.73%–63.40%**.
- Mean correct-option NLL (four-label distribution): **1.1846**.
- Average runtime per question: **3.691 seconds**.
- Average inference request time per question: **1.773 seconds**.
- Average GPU memory per question (device-wide): **3,472.2 MiB** (589/589 questions measured).
- Average Qwen model server RAM per question (working set): **7,757.1 MiB** (589/589 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **2,755 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| generalizing_to_new_scenarios | 57 | 61.40% | 0.9365 |
| provide_preference_aligned_recommendations | 55 | 80.00% | 0.6661 |
| recall_user_shared_facts | 129 | 55.81% | 1.1754 |
| recalling_facts_mentioned_by_the_user | 17 | 58.82% | 1.2925 |
| recalling_the_reasons_behind_previous_updates | 99 | 80.81% | 0.5069 |
| suggest_new_ideas | 93 | 22.58% | 2.3543 |
| track_full_preference_evolution | 139 | 62.59% | 1.1868 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–589 | 589 | 3.691 | 3472.225 | 7757.148 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
