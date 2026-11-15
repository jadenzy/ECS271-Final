# Fine-tuned 57-feature ranker: transfer evaluation

## Completed results

All **3,694 questions** and **8,818 answer-order evaluations** completed.
These are Qwen answer accuracies, using the saved fine-tuned ranker to select
whole-message memory within 2,048 native tokens.

| Dataset | Orders | New fine-tuned 57 | Original reference 55 | Difference (paired group 95%) |
| --- | ---: | ---: | ---: | --- |
| V2 test | 4 | **48.44%** | 42.18% | **+6.26 (+3.83 to +8.71)** |
| V1 | 4 | 65.37% | **68.25%** | −2.89 (−5.89 to 0.00) |
| LoCoMo | 1 | **67.07%** | 60.42% | **+6.65 (+4.46 to +9.23)** |
| Equal-dataset macro | Mixed, as requested | **60.29%** | 56.95% | **+3.34 (+1.85 to +4.86)** |
| Pooled questions | Mixed, as requested | **61.15%** | 56.15% | +5.01 |

The new pipeline improves on the reference-55 baseline on V2 and LoCoMo, while
V1 declines. Its LoCoMo result is close to the Qwen-graded 55-feature ranker
(67.98%, difference −0.91 points, interval −2.32 to +0.55), and lower than full
history (68.83%, difference −1.76, interval −3.30 to −0.59). Macro accuracy is
also higher than the Qwen-graded ranker (58.24%; gain +2.05, interval +0.60 to
+3.51) and full history (57.65%; gain +2.64, interval +0.91 to +4.21). The full
baseline table is in `REPORT.md`; paired counts and intervals are in
`comparison.json`. Small differences from separately recomputed dataset-only
intervals reflect different bootstrap draws from the seeded stream.

V2 single-pass accuracy is **42.09%** and V1 single-pass accuracy **59.76%**.
Post-inference V2 retrieval diagnostics give **95.53% reference Hit@10** and
**80.93% NDCG@10**. The actual packed Qwen memory recalls **94.91%** of visible
references and includes every visible reference on **91.42%** of questions.
Retrieval quality and final answer accuracy are distinct measurements.

Qwen evaluation took **17.5 minutes for V2**, **9.0 minutes for V1**, and
**16.5 minutes for LoCoMo**, after feature preparation. Three new transfer tests
pass. The full suite passed 187 of 188 tests, with the same existing
`test_qwen_judge.JudgeTests.test_deterministic_scoring_errors_are_not_retried`
failure. `verification.json` confirms complete unique cohorts, correct gold
labels, answer-order reconstruction, frozen artifact/code hashes, native token
budgets, selected IDs within visible pools, and zero message truncations.

## Fixed protocol

Frozen model from the persona-disjoint A/B experiment: the encoder was fine-tuned
on 9,000 V2 A questions, and the learned-similarity adapter and 57-feature
LightGBM were fitted on 9,192 B questions. The ranker includes learned
question/message similarity and generic-message correction. There is no new
training, hyperparameter search or redundancy-aware selection in this run.

| Dataset | Cohort | Questions | Answer orders |
| --- | --- | ---: | ---: |
| PersonaMem V2 | Active test | 1,119 | Four cyclic orders |
| PersonaMem V1 | Comparison | 589 | Four cyclic orders |
| LoCoMo-MC10 | Comparison | 1,986 | One pass, ten answer options |

The final V1/V2 answer maximizes the sum of each answer text's log-probability
across the four cyclic option orders. Order 0 also supplies a single-pass result.
LoCoMo uses one pass as requested. The cross-dataset macro/pooled figures combine
those requested policies; they are not a four-order LoCoMo measurement.

## Preserved evaluation setup

- Same saved EmbeddingGemma 2 text architecture, 768 dimensions, input prefixes,
  masked mean pooling and L2 normalization as the original encoder. Weights
  come from the saved A-fine-tuned checkpoint.
- Fresh message, query and chunk embeddings with that checkpoint. The original
  55 feature definitions and SUBTLEX alpha 0.5 are unchanged. The two extra
  features apply the saved B adapter and frozen B question bank without fitting.
- Same frozen Qwen3.5-4B Q4_K_M model, dataset-specific prompt, native scoring,
  single server slot, and 2,048-token selector memory budget as the saved runs.
- Greedy score-order packing of whole messages, skipping messages that do not
  fit; chronological display. V1 cutoffs (including negative cutoffs) and LoCoMo
  sessions/timestamps are preserved. Source system/persona profiles are excluded.
- Evaluation labels/answers never enter encoder or ranker feature computation.
  Answer options are used by Qwen and native prompt-budget accounting. Correct
  labels are accessed for grading after scoring.

Baseline comparisons use saved original reference-trained 55-feature, Qwen-graded
55-feature and full-history runs on exactly the same active question IDs. V1/V2
baselines use four orders and LoCoMo baselines one pass. Earlier rankers used the
full training cohort, while the new pipeline uses A for the encoder and B for
the ranker, so this does not isolate one component's effect. Paired 95% intervals
resample personas/conversations within each dataset, 5,000 draws; cross-dataset
intervals preserve that separation.

## Reproduce or resume

```powershell
.venv/Scripts/python.exe -u -B scripts/evaluate_embedding_ab_transfer.py --stage features
.venv/Scripts/python.exe -u -B scripts/evaluate_embedding_ab_transfer.py --stage score
.venv/Scripts/python.exe -u -B scripts/evaluate_embedding_ab_transfer.py --stage summarize
```

`--datasets v2`, `--datasets v1` or `--datasets locomo` restricts a stage. The
default processes all three. Feature encoding resumes from completion masks.
Scoring resumes only when the model, source, feature, scoring-code and prompt
protocol hashes agree. Local Qwen server access is required.

The live terminal follows `run.log`. `frozen_model.json` identifies the common
saved components and settings. Each dataset directory holds `protocol.json`,
`predictions.jsonl`, `report.json`, `runtime.json` and its native server log.
`REPORT.md` and `comparison.json` contain the completed three-dataset comparison.
Generated feature/embedding/score arrays remain under ignored
`data/ranker55/embedding_ab/transfer/`; frozen weights remain under ignored
`models/embedding_ab/`.
