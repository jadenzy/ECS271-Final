# Frozen Qwen baseline: full visible history

Status: **complete**. Scored 589/589 selected PersonaMem-v1 32K questions from 20 comparison personas.

- Accuracy: **62.14%** (366/589).
- 95% confidence interval, clustered by persona: **58.05%–66.37%**.
- Mean correct-option NLL (four-label distribution): **1.1325**.
- Average runtime per question: **4.542 seconds**.
- Average inference request time per question: **4.428 seconds**.
- Average GPU memory per question (device-wide): **4,448.4 MiB** (589/589 questions measured).
- Average llama-server RAM per question (working set): **4,837.9 MiB** (589/589 questions measured).
- Allocated context: **33,792 tokens**; longest complete prompt: **33,134 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the llama-server working set only. Runtime includes prompt formatting, template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| generalizing_to_new_scenarios | 57 | 70.18% | 0.6875 |
| provide_preference_aligned_recommendations | 55 | 67.27% | 0.9629 |
| recall_user_shared_facts | 129 | 55.81% | 1.3122 |
| recalling_facts_mentioned_by_the_user | 17 | 64.71% | 1.0230 |
| recalling_the_reasons_behind_previous_updates | 99 | 90.91% | 0.2501 |
| suggest_new_ideas | 93 | 11.83% | 2.7891 |
| track_full_preference_evolution | 139 | 75.54% | 0.7490 |

Overall resource averages combine cache modes. Segment averages:

| Saved prediction range | Prefix cache | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1–262 | on | 262 | 3.689 | 4498.504 | 5218.348 |
| 263–295 | off | 33 | 12.790 | 4477.315 | 3786.350 |
| 296–589 | on | 294 | 4.375 | 4400.602 | 4616.846 |

The frozen Qwen3.5-4B Q4_K_M model receives a complete chronological transcript of all visible source messages, with role labels and all repeated system persona descriptions, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. History is never truncated. The answer is the argmax of the four answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
