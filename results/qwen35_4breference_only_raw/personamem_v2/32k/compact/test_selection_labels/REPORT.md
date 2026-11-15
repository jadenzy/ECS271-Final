# Frozen Qwen baseline: original references only (label 2, oracle)

Status: **complete**. Scored 1,119/1,119 selected PersonaMem-v2 32K questions from 50 test personas.
Selection: **compact**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **54.42%** (609/1119).
- 95% confidence interval, clustered by persona: **51.09%–57.60%**.
- Mean correct-option NLL (four-label distribution): **1.1938**.
- Average runtime per question: **0.199 seconds**.
- Average inference request time per question: **0.190 seconds**.
- Average GPU memory per question (device-wide): **6,792.3 MiB** (1119/1119 questions measured).
- Average Qwen model server RAM per question (working set): **11,273.0 MiB** (1119/1119 questions measured).
- Allocated context: **41,216 tokens**; longest complete prompt: **2,138 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 226 | 61.06% | 1.0317 |
| ask_to_forget | 253 | 48.62% | 1.3042 |
| health_and_medical_conditions | 137 | 55.47% | 1.1345 |
| neutral_preferences | 213 | 61.50% | 1.0235 |
| stereotypical_pref | 142 | 59.86% | 1.0632 |
| therapy_background | 148 | 37.84% | 1.6777 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1119 | 1119 | 0.199 | 6792.267 | 11272.972 |

Reference messages absent from visible histories: 7 across 7 scored questions. These remain original label 2 annotations; no label 1 messages are added.

The frozen Qwen3.5-4B-Q4_K_M model receives only original dataset reference conversation snippets (label 2), in original snippet order under the selector heading and source-position labels, followed by the question and all options; Qwen-generated label 1 messages are excluded. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
