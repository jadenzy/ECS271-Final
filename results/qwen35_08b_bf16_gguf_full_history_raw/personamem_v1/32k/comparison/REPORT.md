# Frozen Qwen baseline: full visible history

Status: **interrupted**. Scored 70/589 selected PersonaMem-v1 32K questions from 3 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **45.71%** (32/70).
- 95% confidence interval, clustered by persona: **35.29%Ã¢â‚¬â€œ66.67%**.
- Mean correct-option NLL (four-label distribution): **1.1895**.
- Average runtime per question: **1.667 seconds**.
- Average inference request time per question: **1.577 seconds**.
- Average GPU memory per question (device-wide): **2,652.4 MiB** (70/70 questions measured).
- Average Qwen model server RAM per question (working set): **2,610.9 MiB** (70/70 questions measured).
- Allocated context: **33,280 tokens**; longest complete prompt: **31,729 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| generalizing_to_new_scenarios | 11 | 18.18% | 1.3862 |
| provide_preference_aligned_recommendations | 8 | 37.50% | 1.2132 |
| recall_user_shared_facts | 12 | 66.67% | 0.9473 |
| recalling_facts_mentioned_by_the_user | 1 | 100.00% | 0.1697 |
| recalling_the_reasons_behind_previous_updates | 14 | 64.29% | 0.8595 |
| suggest_new_ideas | 11 | 18.18% | 1.7871 |
| track_full_preference_evolution | 13 | 53.85% | 1.1603 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1Ã¢â‚¬â€œ70 | 70 | 1.667 | 2652.377 | 2610.886 |

The frozen Qwen3.5-0.8B-BF16-GGUF model receives a complete chronological transcript of all visible source messages, with role labels and all repeated system persona descriptions, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. History is never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
