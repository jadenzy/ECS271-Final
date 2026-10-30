# Frozen Qwen baseline: original references only (label 2, oracle)

Status: **complete**. Scored 1,242/1,242 selected PersonaMem-v2 32K questions from 50 test personas.
Selection: **compact**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **52.25%** (649/1242).
- 95% confidence interval, clustered by persona: **48.91%–55.27%**.
- Mean correct-option NLL (four-label distribution): **1.2810**.
- Average runtime per question: **0.706 seconds**.
- Average inference request time per question: **0.692 seconds**.
- Average GPU memory per question (device-wide): **4,664.4 MiB** (1242/1242 questions measured).
- Average Qwen model server RAM per question (working set): **7,878.8 MiB** (1242/1242 questions measured).
- Allocated context: **41,216 tokens**; longest complete prompt: **2,134 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 226 | 60.18% | 1.0144 |
| ask_to_forget | 253 | 47.43% | 1.3219 |
| health_and_medical_conditions | 137 | 56.20% | 1.0811 |
| neutral_preferences | 213 | 61.50% | 0.9932 |
| sensitive_info | 123 | 35.77% | 2.3038 |
| stereotypical_pref | 142 | 58.45% | 1.0766 |
| therapy_background | 148 | 39.19% | 1.5633 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1242 | 1242 | 0.706 | 4664.390 | 7878.778 |

Reference messages absent from visible histories: 320 across 130 scored questions. These remain original label 2 annotations; no label 1 messages are added.

The frozen Qwen3.5-4B-Q4_K_M model receives only original dataset reference conversation snippets (label 2), in original snippet order, followed by the question and all options; Qwen-generated label 1 messages are excluded. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. Selected messages are never truncated. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
