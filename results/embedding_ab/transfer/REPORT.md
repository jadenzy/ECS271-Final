# Frozen fine-tuned ranker transfer evaluation

V1 and V2 use four cyclic answer orders; LoCoMo uses one pass. Selector memory budget: 2,048 tokens. MMR is not applied.

| Dataset | Questions | Orders | Fine-tuned 57 | Reference 55 | Graded 55 | Full history |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 | 1119 | 4 | 48.44% | 42.18% | 41.73% | 36.37% |
| v1 | 589 | 4 | 65.37% | 68.25% | 65.03% | 67.74% |
| locomo | 1986 | 1 | 67.07% | 60.42% | 67.98% | 68.83% |
| macro | | mixed | 60.29% | 56.95% | 58.24% | 57.65% |
| pooled | | mixed | 61.15% | 56.15% | 59.56% | 58.83% |

Intervals and paired gained/lost question counts are in comparison.json. Baselines are the saved original runs, on identical active question IDs.
The new encoder was fine-tuned on A and the ranker/adapter on B; older 55-feature baselines used the full original training cohort. This comparison does not isolate component effects.
