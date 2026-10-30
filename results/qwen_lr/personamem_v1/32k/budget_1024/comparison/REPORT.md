# Frozen Qwen baseline: LR-selected memories, 1,024-token budget

Status: **complete**. Scored 589/589 selected PersonaMem-v1 32K questions from 20 comparison personas.
Mean selected memories: **10.33** of **135.26** available; mean added memory tokens: **1019.54**.
Mean LR ranking and packing time: **0.031 seconds**; included in per-question runtime. Offline feature generation and input loading are excluded.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **61.80%** (364/589).
- 95% confidence interval, clustered by persona: **58.14%–65.81%**.
- Mean correct-option NLL (four-label distribution): **1.0781**.
- Average runtime per question: **0.744 seconds**.
- Average inference request time per question: **0.713 seconds**.
- Average GPU memory per question (device-wide): **3,809.9 MiB** (589/589 questions measured).
- Average llama-server RAM per question (working set): **3,188.5 MiB** (589/589 questions measured).
- Allocated context: **4,096 tokens**; longest complete prompt: **1,779 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the llama-server working set only. Runtime includes prompt formatting, template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| generalizing_to_new_scenarios | 57 | 75.44% | 0.5666 |
| provide_preference_aligned_recommendations | 55 | 72.73% | 0.7060 |
| recall_user_shared_facts | 129 | 55.81% | 1.1538 |
| recalling_facts_mentioned_by_the_user | 17 | 58.82% | 1.1958 |
| recalling_the_reasons_behind_previous_updates | 99 | 83.84% | 0.3710 |
| suggest_new_ideas | 93 | 26.88% | 2.3237 |
| track_full_preference_evolution | 139 | 65.47% | 1.0206 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–589 | 589 | 0.744 | 3809.881 | 3188.500 |

The frozen Qwen3.5-4B Q4_K_M model receives whole LR-selected conversation snippets, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected snippets are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
