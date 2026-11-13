# Frozen Qwen baseline: no memory (question/options only)

Status: **complete**. Scored 1,119/1,119 selected PersonaMem-v2 32K questions from 50 test personas.
Selection: **compact**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **29.40%** (329/1119).
- 95% confidence interval, clustered by persona: **26.38%–32.22%**.
- Mean correct-option NLL (four-label distribution): **1.9900**.
- Average runtime per question: **0.426 seconds**.
- Average inference request time per question: **0.420 seconds**.
- Average GPU memory per question (device-wide): **3,515.2 MiB** (1119/1119 questions measured).
- Average Qwen model server RAM per question (working set): **8,065.0 MiB** (1119/1119 questions measured).
- Allocated context: **4,096 tokens**; longest complete prompt: **1,636 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 226 | 33.19% | 1.8926 |
| ask_to_forget | 253 | 18.58% | 2.3536 |
| health_and_medical_conditions | 137 | 38.69% | 1.6636 |
| neutral_preferences | 213 | 33.80% | 1.8849 |
| stereotypical_pref | 142 | 33.10% | 1.8151 |
| therapy_background | 148 | 23.65% | 2.1384 |

Resource averages by evaluator invocation:

| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 1–1119 | 1119 | 0.426 | 3515.158 | 8065.037 |

The frozen Qwen3.5-4B-Q4_K_M model receives only the shared evaluation instructions, question, and all options; no conversation or persona profile. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. No conversation history is loaded or passed to Qwen. The answer is the argmax of the 4 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
