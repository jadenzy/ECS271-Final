# Frozen Qwen baseline: LR-selected memories, 1,024-token budget

Status: **interrupted**. Scored 75/1,986 selected LoCoMo-MC10 FULL questions from 1 comparison personas.
Mean selected memories: **11.49** of **419.00** available; mean added memory tokens: **998.93**.
Mean LR ranking and packing time: **0.030 seconds**; included in per-question runtime. Offline feature generation and input loading are excluded.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **68.00%** (51/75).
- 95% confidence interval, clustered by conversation: not estimable from a single persona.
- Mean correct-option NLL (ten-label distribution): **1.4020**.
- Average runtime per question: **0.504 seconds**.
- Average inference request time per question: **0.474 seconds**.
- Average GPU memory per question (device-wide): **3,759.0 MiB** (75/75 questions measured).
- Average llama-server RAM per question (working set): **3,153.8 MiB** (75/75 questions measured).
- Allocated context: **4,096 tokens**; longest complete prompt: **1,329 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the llama-server working set only. Runtime includes prompt formatting, template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| adversarial | 14 | 50.00% | 2.4220 |
| multi_hop | 5 | 40.00% | 3.7858 |
| open_domain | 52 | 76.92% | 0.7999 |
| single_hop | 3 | 33.33% | 3.2764 |
| temporal_reasoning | 1 | 100.00% | 0.8884 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–75 | 75 | 0.504 | 3758.973 | 3153.828 |

The frozen Qwen3.5-4B Q4_K_M model receives whole LR-selected conversation snippets, in chronological order, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected snippets are never truncated. The answer is the argmax of the 10 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
