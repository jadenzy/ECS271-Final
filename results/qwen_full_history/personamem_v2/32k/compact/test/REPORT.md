# Frozen Qwen baseline: full visible history

Status: **complete**. Scored 1,242/1,242 selected PersonaMem-v2 32K questions from 50 test personas.
Selection: **compact**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **35.83%** (445/1242).
- 95% confidence interval, clustered by persona: **33.04%–38.56%**.
- Mean correct-option NLL (four-label distribution): **1.9466**.
- Average runtime per question: **2.047 seconds**.
- Average inference request time per question: **1.933 seconds**.
- Average GPU memory per question (device-wide): **4,818.2 MiB** (1242/1242 questions measured).
- Average llama-server RAM per question (working set): **7,104.7 MiB** (1242/1242 questions measured).
- Allocated context: **41,984 tokens**; longest complete prompt: **37,894 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the llama-server working set only. Runtime includes prompt formatting, template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 226 | 33.63% | 1.9517 |
| ask_to_forget | 253 | 25.30% | 2.1306 |
| health_and_medical_conditions | 137 | 38.69% | 1.8922 |
| neutral_preferences | 213 | 37.09% | 1.8508 |
| sensitive_info | 123 | 32.52% | 2.2878 |
| stereotypical_pref | 142 | 50.00% | 1.4551 |
| therapy_background | 148 | 41.89% | 2.0005 |

The frozen Qwen3.5-4B Q4_K_M model receives a complete chronological transcript of all visible source messages, with role labels and all repeated system persona descriptions, followed by the question and all options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. History is never truncated. The answer is the argmax of the four answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
