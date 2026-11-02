# Frozen Qwen baseline: LightGBM selector (2,048 Qwen memory tokens, stopping 1.5 below the top score after 1,000 tokens)

Status: **complete**. Scored 589/589 selected PersonaMem-v1 32K questions from 20 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **60.78%** (358/589).
- 95% confidence interval, clustered by persona: **56.78%–64.77%**.
- Mean correct-option NLL (four-label distribution): **1.1474**.
- Average runtime per question: **0.302 seconds**.
- Average inference request time per question: **0.284 seconds**.
- Average GPU memory per question (device-wide): **4,544.4 MiB** (589/589 questions measured).
- Average Qwen model server RAM per question (working set): **10,970.9 MiB** (589/589 questions measured).
- Allocated context: **3,048 tokens**; longest complete prompt: **2,789 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| generalizing_to_new_scenarios | 57 | 63.16% | 0.9072 |
| provide_preference_aligned_recommendations | 55 | 76.36% | 0.7091 |
| recall_user_shared_facts | 129 | 56.59% | 1.1939 |
| recalling_facts_mentioned_by_the_user | 17 | 58.82% | 0.8726 |
| recalling_the_reasons_behind_previous_updates | 99 | 83.84% | 0.4376 |
| suggest_new_ideas | 93 | 29.03% | 2.2142 |
| track_full_preference_evolution | 139 | 62.59% | 1.2015 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–589 | 589 | 0.302 | 4544.424 | 10970.926 |

The frozen Qwen3.5-4B-Q4_K_M model receives complete LightGBM-selected source messages within 2,048 Qwen memory tokens, stopping 1.5 below the top score after 1,000 tokens, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
