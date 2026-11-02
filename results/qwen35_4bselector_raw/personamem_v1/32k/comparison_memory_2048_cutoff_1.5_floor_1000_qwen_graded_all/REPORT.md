# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens, stopping 1.5 below the top score after 1,000 tokens)

Status: **complete**. Scored 589/589 selected PersonaMem-v1 32K questions from 20 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **59.08%** (348/589).
- 95% confidence interval, clustered by persona: **55.87%–62.44%**.
- Mean correct-option NLL (four-label distribution): **1.1374**.
- Average runtime per question: **0.276 seconds**.
- Average inference request time per question: **0.260 seconds**.
- Average GPU memory per question (device-wide): **4,625.2 MiB** (589/589 questions measured).
- Average Qwen model server RAM per question (working set): **10,959.1 MiB** (589/589 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **2,747 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| generalizing_to_new_scenarios | 57 | 63.16% | 0.8624 |
| provide_preference_aligned_recommendations | 55 | 80.00% | 0.6762 |
| recall_user_shared_facts | 129 | 52.71% | 1.1932 |
| recalling_facts_mentioned_by_the_user | 17 | 64.71% | 0.7251 |
| recalling_the_reasons_behind_previous_updates | 99 | 78.79% | 0.5199 |
| suggest_new_ideas | 93 | 25.81% | 2.1566 |
| track_full_preference_evolution | 139 | 62.59% | 1.1891 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–589 | 589 | 0.276 | 4625.243 | 10959.106 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, stopping 1.5 below the top score after 1,000 tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
