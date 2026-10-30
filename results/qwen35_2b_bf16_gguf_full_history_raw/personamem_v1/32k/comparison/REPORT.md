# Frozen Qwen baseline: full visible history

Status: **complete**. Scored 589/589 selected PersonaMem-v1 32K questions from 20 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **50.08%** (295/589).
- 95% confidence interval, clustered by persona: **45.40%Ã¢â‚¬â€œ54.77%**.
- Mean correct-option NLL (four-label distribution): **1.2357**.
- Average runtime per question: **4.490 seconds**.
- Average inference request time per question: **4.407 seconds**.
- Average GPU memory per question (device-wide): **4,772.1 MiB** (589/589 questions measured).
- Average Qwen model server RAM per question (working set): **5,650.1 MiB** (589/589 questions measured).
- Allocated context: **33,280 tokens**; longest complete prompt: **33,134 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| generalizing_to_new_scenarios | 57 | 42.11% | 1.1719 |
| provide_preference_aligned_recommendations | 55 | 43.64% | 1.2685 |
| recall_user_shared_facts | 129 | 55.81% | 1.0430 |
| recalling_facts_mentioned_by_the_user | 17 | 58.82% | 1.1174 |
| recalling_the_reasons_behind_previous_updates | 99 | 86.87% | 0.4801 |
| suggest_new_ideas | 93 | 15.05% | 2.3677 |
| track_full_preference_evolution | 139 | 46.76% | 1.2228 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1Ã¢â‚¬â€œ589 | 589 | 4.490 | 4772.097 | 5650.074 |

The frozen Qwen3.5-2B-BF16-GGUF model receives a complete chronological transcript of all visible source messages, with role labels and all repeated system persona descriptions, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. History is never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
