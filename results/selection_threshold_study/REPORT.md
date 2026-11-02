# Message-selection threshold study

Can a ranker-score threshold replace the fixed 2,048-token memory budget? This small test looks at the default ranker's scores at ranks 1–20 on every dataset, then scores ten selection rules with Qwen on 50 sampled questions per dataset. The best idea, budget plus a relative score cutoff, was then run on every evaluation question with both rankers ([full test sets](#full-test-sets-budget-plus-cutoff-hybrid)): same overall accuracy with 22–31% fewer memory tokens.

## Setup

- **Ranker:** `models/subtlex_mix/alpha_0.5/ranker55.txt` (references only, with role), through `evaluate_qwen.py`'s own selector ranking, packing and prompt format.
- **Questions:** 50 per dataset, sampled with seed 271. PersonaMem V2 uses **validation**, not test. V1, LoCoMo-MC10 and LongBench v2 (≤32k tokens) only have their evaluation sets, so their samples come from those; treat them as exploratory.
- **Rules**, each applied to the ranker's order:
  - `budget 2,048 tokens (current)`: the evaluator's packing (greedy by score, skip messages that do not fit).
  - `top k` for k = 3, 5, 10, 20.
  - `score ≥ t` for t = 1, 0.5, 0: messages above an absolute score, at least 1 and at most 20.
  - `within d of top` for d = 1, 1.5: messages scoring no more than d below the best one, at most 20.
- **Metrics:** Qwen accuracy and mean correct-answer NLL (lower is better; less noisy than accuracy on 50 questions). Memory tokens are the native Qwen prompt increment.
- Identical selections are scored once. Per-question results: `per_question.jsonl`; summaries: `report.json`; paired comparisons: `paired_vs_budget.json`.

## 1. Ranker scores at the top ranks (all questions)

Scores are on a similar scale on every dataset. Median score by rank:

| Dataset | Questions | Median pool | Rank 1 | Rank 3 | Rank 5 | Rank 10 | Rank 15 | Rank 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PersonaMem V2 val | 1,002 | 232 | +1.76 | +0.98 | +0.59 | +0.02 | −0.33 | −0.59 |
| PersonaMem V1 | 589 | 161 | +2.30 | +1.54 | +1.21 | +0.73 | +0.50 | +0.34 |
| LoCoMo-MC10 | 1,986 | 663 | +1.69 | +1.17 | +0.90 | +0.53 | +0.32 | +0.16 |
| LongBench v2 | 111 | 90 | +1.57 | +0.96 | +0.69 | +0.33 | +0.12 | −0.04 |

The median drop from the top score is about 1.0–1.1 by rank 5 and 1.1–1.7 by rank 10 (smallest on LoCoMo, which has the largest pools).

On V2 validation, the annotated references concentrate at the top (base rate about 1.3% of messages):

| Rank | 1 | 2 | 3 | 4 | 5 | 6–10 | 11–20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Share that are references | 34% | 32% | 23% | 21% | 11% | 5–8% | 3–5% |

| Score band (top-20 messages) | < −1 | −1 to 0 | 0 to 1 | 1 to 2 | 2 to 3 | ≥ 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Share that are references | 2% | 4% | 8% | 23% | 46% | 79% |

So by annotation, scores above about 1 mark likely evidence and the precision falls sharply after rank 4.

## 2. Qwen accuracy by rule (50 questions per dataset)

| Rule | V2 val | V1 | LoCoMo | LongBench | **Macro accuracy** | Macro NLL | Mean memory tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **budget 2,048 tokens (current)** | 40% | **72%** | **66%** | 34% | **53.0%** | **1.552** | 2,023 |
| top 3 | 44% | 66% | 44% | **42%** | 49.0% | 1.680 | 495 |
| top 5 | 42% | 68% | 54% | 38% | 50.5% | 1.708 | 807 |
| top 10 | 38% | 68% | 62% | 30% | 49.5% | 1.619 | 1,586 |
| top 20 | 36% | 64% | 60% | 34% | 48.5% | 1.655 | 3,117 |
| score ≥ 1 | 46% | 66% | 40% | 40% | 48.0% | 1.688 | 625 |
| score ≥ 0.5 | 42% | 66% | 48% | 32% | 47.0% | 1.763 | 1,371 |
| score ≥ 0 | 38% | 64% | 58% | 30% | 47.5% | 1.728 | 2,466 |
| within 1 of top | 46% | 60% | 56% | 36% | 49.5% | 1.703 | 1,143 |
| **within 1.5 of top** | **48%** | 66% | 60% | 38% | **53.0%** | 1.596 | 1,858 |

Messages selected per question (mean):

| Rule | V2 val | V1 | LoCoMo | LongBench |
| --- | ---: | ---: | ---: | ---: |
| budget 2,048 tokens | 15.4 | 13.1 | 31.8 | 8.8 |
| score ≥ 1 | 3.0 | 6.8 | 4.0 | 2.6 |
| score ≥ 0 | 10.5 | 19.3 | 17.5 | 16.7 |
| within 1.5 of top | 9.3 | 8.8 | 16.0 | 13.9 |

On V2, more messages find more references but answer worse: the share of questions with at least one reference selected rises from 44% (`score ≥ 1`) to 70% (budget) and 72% (`top 20`), while accuracy falls from 46% to 40% and 36%.

## 3. Paired against the current budget

Rule minus budget on the same questions (5,000 bootstrap draws over questions; macro resamples each dataset separately):

| Rule | V2 val | V1 | LoCoMo | LongBench | Macro accuracy | Macro NLL |
| --- | --- | --- | --- | --- | --- | --- |
| within 1.5 of top | **+8** (+2 to +16) | −6 (−14 to +2) | −6 (−16 to +2) | +4 (0 to +10) | +0.0 (−3.5 to +4.0) | +0.044 (−0.065 to +0.159) |
| score ≥ 1 | +6 (−4 to +16) | −6 (−14 to 0) | **−26** (−40 to −14) | +6 (−6 to +18) | −5.0 (−10.5 to 0) | +0.136 (−0.054 to +0.343) |
| top 3 | +4 (−4 to +12) | −6 (−14 to 0) | **−22** (−38 to −8) | +8 (−4 to +20) | −4.0 (−9.5 to +1.5) | +0.128 (−0.066 to +0.329) |
| top 10 | −2 (−8 to +4) | −4 (−10 to 0) | −4 (−14 to +6) | −4 (−10 to 0) | −3.5 (−7.0 to 0) | +0.067 (−0.028 to +0.166) |
| top 20 | −4 (−10 to 0) | **−8** (−16 to −2) | −6 (−16 to +2) | 0 (−6 to +6) | **−4.5** (−8.0 to −1.0) | **+0.103** (+0.005 to +0.206) |

Accuracy changes are percentage points.

## Findings

1. **No single score cutoff works on every dataset.** The scores share a scale, but the best amount of memory differs. LoCoMo turns are short (about 64 tokens each under the budget) and it needs about 30 of them: keeping only the top 3 or scores ≥ 1 costs 22–26 points. V1 also does best with the full budget, although its messages are about as long as V2's (about 150 tokens). PersonaMem V2 and LongBench (passages about 225 tokens) do best with a few high-scoring messages. Message size explains LoCoMo; the V1/V2 difference is about the questions, not the messages.
2. **On PersonaMem V2 and LongBench, less is more.** `score ≥ 1` (about 3 messages, 470–620 tokens) and `top 3` beat the 2,048-token budget by 4–8 points there while using 70–75% fewer tokens. More memory retrieves more references but distracts the 4B model.
3. **The relative cutoff is the best threshold rule overall.** `within 1.5 of top` ties the budget on macro accuracy (53.0%) with about 8% fewer tokens, and is significantly better on V2 (+8). It loses about 6 points on V1 and LoCoMo (not significant) and has a slightly worse macro NLL.
4. **More than the budget does not help.** `top 20` and `score ≥ 0` use more tokens on average and are worse; `top 20` is significantly worse on macro accuracy and NLL.
5. **The current budget remains the most robust single rule:** best macro NLL (1.552) and tied best macro accuracy.

## Suggestions

- Keep the 2,048-token budget as the default; no single threshold beats it across datasets.
- If fewer tokens matter, `within 1.5 of top` is the threshold to carry forward: equal macro accuracy, fewer tokens, and the best V2 result.
- Untested next step: combine the two, packing by the token budget but stopping at messages more than about 1.5 below the top score, with a token floor (for example 1,000 tokens) so short-message datasets such as LoCoMo still get enough turns.
- Confirm any choice on held-out dev splits with more questions. With 50 questions per dataset, most per-dataset differences under about 8 points are within noise.

## Full test sets: budget plus cutoff (hybrid)

The suggested hybrid was then run on every evaluation question (V2 test 1,242, V1 589, LoCoMo 1,986, LongBench 111) with both rankers: `evaluate_qwen.py --mode selector --score-cutoff 1.5 --floor-tokens 1000`. It packs by the 2,048-token budget in score order as before, but once 1,000 memory tokens are packed it stops at the first message scoring more than 1.5 below the question's top score. Outputs are the `*_memory_2048_cutoff_1.5_floor_1000_*` folders next to the budget-only runs; the comparison is in `hybrid_test_sets.json`. Intervals: 5,000 paired bootstrap draws resampling each dataset's personas, conversations or questions.

**References-only ranker**

| Dataset | Budget only | Hybrid | Hybrid − budget (95%) | Memory tokens | Messages |
| --- | ---: | ---: | --- | --- | --- |
| PersonaMem V2 test | 39.37% | 40.18% | +0.81 (−0.62 to +2.21) | 2,040 → 1,463 | 15.3 → 9.8 |
| PersonaMem V1 | 60.95% | 60.78% | −0.17 (−2.06 to +1.94) | 2,024 → 1,526 | 13.3 → 9.6 |
| LoCoMo-MC10 | 60.42% | 60.02% | −0.40 (−1.22 to +0.50) | 2,041 → 1,535 | 32.0 → 23.2 |
| LongBench v2 | 36.04% | 35.14% | −0.90 (−5.41 to +2.70) | 1,984 → 1,814 | 9.1 → 8.1 |
| **Macro** | 49.20% | 49.03% | −0.17 (−1.39 to +1.01) | 2,022 → 1,584 | |
| **Pooled** | 53.16% | 53.16% | +0.00 (−0.71 to +0.71) | 2,037 → 1,519 | |

**Qwen-graded ranker**

| Dataset | Budget only | Hybrid | Hybrid − budget (95%) | Memory tokens | Messages |
| --- | ---: | ---: | --- | --- | --- |
| PersonaMem V2 test | 37.28% | 39.86% | **+2.58** (+0.88 to +4.23) | 2,040 → 1,390 | 15.0 → 8.9 |
| PersonaMem V1 | 62.65% | 59.08% | **−3.57** (−5.20 to −2.01) | 2,026 → 1,346 | 13.1 → 8.2 |
| LoCoMo-MC10 | 67.98% | 66.26% | **−1.71** (−2.61 to −0.61) | 2,041 → 1,222 | 32.9 → 18.5 |
| LongBench v2 | 37.84% | 40.54% | +2.70 (0.00 to +6.31) | 1,985 → 1,596 | 9.0 → 6.8 |
| **Macro** | 51.44% | 51.44% | +0.00 (−0.92 to +1.03) | 2,023 → 1,388 | |
| **Pooled** | 56.62% | 56.11% | −0.51 (−1.32 to +0.33) | 2,037 → 1,304 | |

Correct-answer NLL changes are all within ±0.05 (in `hybrid_test_sets.json`).

- **Same overall accuracy with fewer tokens.** On both averages and for both rankers the hybrid is within about half a point of the budget, with intervals spanning zero, while using 22% fewer memory tokens (references only) or 31% fewer (Qwen grades).
- **The floor works.** On LoCoMo the hybrid keeps about 20 short turns and loses at most 1.7 points, against 22–26 points for the pure cutoffs in the sample study.
- **With the references-only ranker nothing changes beyond noise** on any dataset.
- **With the Qwen-graded ranker the effect splits by dataset:** cutting its weaker topical matches helps where few strong messages are enough (V2 +2.6, LongBench +2.7) and hurts where more context is needed (V1 −3.6, LoCoMo −1.7).
- **The 50-question sample overstated the gain:** `within 1.5 of top` was +8 on 50 V2 validation questions; on the full V2 test set the hybrid is +0.8 (references only) and +2.6 (Qwen grades).

**Recommendation:** the hybrid is a token saving, not an accuracy gain. Use it where memory tokens cost something (about a quarter to a third fewer, no measurable accuracy loss overall); keep the plain budget as the default otherwise. The cutoff and floor were chosen from the sample study, which drew 50 questions from each of the V1, LoCoMo and LongBench evaluation sets, so those results are not fully held out.

## Limits

- 50 questions per dataset; intervals resample questions and ignore persona or conversation clustering, so they are somewhat too narrow.
- V1, LoCoMo and LongBench samples come from their evaluation sets.
- One ranker only. The no-role ranker selects differently on LongBench (43.24% vs 36.04% at 2,048 tokens) and would need its own scores and cutoffs.
