# Frozen Qwen baseline: full visible history

Status: **complete**. Scored 1,986/1,986 selected LoCoMo-MC10 FULL questions from 10 comparison personas.
Selection: **full**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.

- Accuracy: **68.63%** (1363/1986).
- 95% confidence interval, clustered by conversation: **66.23%–70.92%**.
- Mean correct-option NLL (ten-label distribution): **1.2948**.
- Average runtime per question: **0.826 seconds**.
- Average inference request time per question: **0.752 seconds**.
- Average GPU memory per question (device-wide): **5,321.3 MiB** (1986/1986 questions measured).
- Average llama-server RAM per question (working set): **3,139.4 MiB** (1986/1986 questions measured).
- Allocated context: **32,768 tokens**; longest complete prompt: **32,174 tokens**.

Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the llama-server working set only. Runtime includes prompt formatting, template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.

| Question type | Questions | Accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| adversarial | 446 | 20.63% | 3.6853 |
| multi_hop | 321 | 46.73% | 1.8763 |
| open_domain | 841 | 96.08% | 0.1314 |
| single_hop | 282 | 90.43% | 0.3108 |
| temporal_reasoning | 96 | 60.42% | 1.3269 |

The frozen Qwen3.5-4B Q4_K_M model receives a complete chronological transcript of both speakers with session timestamps, followed by the question and all ten options. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. History is never truncated. The answer is the argmax of the 10 answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.

Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.
