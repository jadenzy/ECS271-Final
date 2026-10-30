# Review of the saved LR and full-history results

Reviewed all six completed evaluations and the current 800-persona LR training report. This review reads saved predictions, features, labels and model coefficients; it does not train a model or run Qwen.

## Main conclusion

The current selector largely preserves PersonaMem accuracy while reducing prompt tokens by about 95–96%. LoCoMo has a meaningful accuracy gap, especially on answerable questions. The evidence favors improving memory representation, supervision and ranking features/objectives over adding more personas or searching additional C values.

These are diagnoses and proposed experiments. No proposed accuracy improvement has yet been demonstrated by a new Qwen evaluation.

## Completed evaluations

| Dataset | Questions | Full-history correct / accuracy | LR correct / accuracy | LR minus full | Full prompt tokens, mean | LR prompt tokens, mean | Token reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PersonaMem V1 | 589 | 366 / 62.14% | 364 / 61.80% | -0.34 pp | 27,708 | 1,440 | 94.80% |
| PersonaMem V2 test | 1,242 | 445 / 35.83% | 435 / 35.02% | -0.81 pp | 35,933 | 1,381 | 96.16% |
| LoCoMo-MC10 | 1,986 | 1,363 / 68.63% | 1,251 / 62.99% | -5.64 pp | 28,067 | 1,205 | 95.71% |

V1/V2 have four answer choices; LoCoMo has ten. Compare modes within a dataset, rather than interpreting accuracy differences between datasets as a ranking comparison.

Every prediction file has the expected complete count with unique example IDs. Paired modes have identical question ID sets and correct labels. All six protocols identify the same Qwen GGUF hash. All three LR protocols record the current selector hash `b10261b4d96729b5317d71165d7fe01cf14e054f08d4c8f8695a953bba779155`, which matches the 800-persona model.

Paired question outcomes, using the same questions:

| Dataset | LR fixes a full-history error | LR loses a full-history success | Paired difference 95% cluster-bootstrap interval |
| --- | ---: | ---: | ---: |
| V1 | 49 | 51 | -5.00 to +3.73 pp |
| V2 | 100 | 110 | -2.81 to +1.16 pp |
| LoCoMo | 262 | 374 | -7.86 to -3.30 pp |

Intervals use 10,000 paired bootstrap resamples of personas/conversations, seed 271, with question-weighted accuracy in each resample. LoCoMo has only ten conversation clusters. These intervals describe saved-run variability; they do not remove the protocol differences noted below.

## LoCoMo's aggregate score hides a larger answerable-question loss

| LoCoMo group | Questions | Full history | LR | Difference |
| --- | ---: | ---: | ---: | ---: |
| Answerable: all non-adversarial types | 1,540 | 1,271 / 82.53% | 982 / 63.77% | -18.77 pp |
| Adversarial / “Not answerable” | 446 | 92 / 20.63% | 269 / 60.31% | +39.69 pp |
| Open-domain category | 841 | 96.08% | 73.72% | -22.35 pp |
| Single-hop | 282 | 90.43% | 65.96% | -24.47 pp |
| Multi-hop | 321 | 46.73% | 39.56% | -7.17 pp |
| Temporal reasoning | 96 | 60.42% | 51.04% | -9.38 pp |

The special “Not answerable” option occurs in the 446 adversarial examples. Its answer letter varies; the result is not explained by a fixed correct letter. LR selects that option 269 times. The gain could reflect reduced distraction or different abstention behavior; a question/options-only baseline would help distinguish these from improved retrieval. No such baseline is saved here.

Examples where full history succeeds and LR fails include `locomo:conv-26_q104`, asking which book Caroline recommended to Melanie, and `locomo:conv-26_q106`, asking what Melanie's new shoes were for. Specific entities, events and surrounding turns matter for these questions.

Report answerable and adversarial performance separately in future comparisons.

## The classifier accuracy is not the retrieval objective

The current model has two inputs, BM25 score and MiniLM cosine, and three classifier parameters. It trained on 2,674,713 pairs from 800 personas; validation has 131,074 pairs from 40 personas.

- Validation positives: 2,069 / 131,074 = 1.58%.
- Validation classification accuracy at 0.5: 98.4207%.
- Always predicting negative: 98.4215% accuracy.
- Only one validation pair has predicted probability at least 0.5; it is a negative label.
- Validation average precision: 0.05948; ROC AUC: 0.6860.
- Train / validation log loss: 0.07469 / 0.07646.

This does not imply that the ranking is constant or useless: inference sorts probabilities and does not apply a 0.5 threshold. It does mean that classification accuracy gives a misleading impression of selector quality. Lowering a classification threshold would not change the current ordering or budget packing.

The standardized weights are BM25 **-0.05024** and MiniLM **+0.75421**. Up to an additive constant, the ranking score in raw feature units is approximately:

`6.394 * minilm_cosine - 0.01044 * bm25_score`

Thus the learned model mainly follows MiniLM, with a small negative lexical adjustment. A negative BM25 coefficient is not automatically a bug; evaluate feature interactions and ablations before changing its sign.

The C sweep from 0.01 to 10 changes validation log loss from 0.0764596 to 0.0764644. Additional C search is a low priority.

## Read-only validation retrieval comparison

Computed from the current saved validation feature/label rows, without fitting anything. Ranking metrics below average the 910 questions with at least one reference-positive candidate; 92 questions have no positive reference labels and are excluded from these macro ranking metrics. Ties use embedding index.

| Ranking score | Mean per-question AP | Hit@5 | Hit@10 | Recall@10 | Mean top-10 overlap with current LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current 800-persona LR | 22.71% | 36.26% | 46.48% | 33.67% | 100% |
| MiniLM cosine only | 23.17% | 36.70% | 46.70% | 34.07% | 95.81% |
| BM25 only | 14.65% | 28.13% | 39.01% | 22.80% | 27.50% |
| Archived 100-persona LR coefficients | 22.70% | 36.15% | 46.48% | 33.62% | 99.76% |

The archived-model row applies its saved scaler and coefficients to the same current validation scalar features. It is not an old-vs-new end-to-end Qwen experiment. The small MiniLM advantage is descriptive, not an established significant improvement.

These labels measure exact reference overlap, not complete semantic relevance or answer usefulness. Still, the analysis shows no clear learned ranking advantage over MiniLM alone and almost no ranking change after expanding from 100 to 800 training personas.

## Memory representation and label limitations

V2 test reference-overlap diagnostics, used only for post-evaluation analysis:

- 1,119 of 1,242 questions have mapped reference spans and a retrievable reference-positive candidate.
- Of those, the packed LR context contains at least one reference-positive memory for 626 questions: **55.94%**.
- Mean recall of reference-positive memory indexes is **41.10%**.
- Qwen accuracy is 35.14% when this overlap proxy is hit and 34.89% when it is missed. This coarse proxy alone is not a strong indicator of answer success; it does not establish that reference labels are useless or that retrieval has no effect.
- Available memory chunks contain 476,576 of 476,794 user-reference characters: **99.95%**.
- Available chunks contain only 113,765 of 1,543,450 assistant-reference characters: **7.37%**.
- Packed chunks contain 172,194 user-reference characters and 48,843 assistant-reference characters, respectively **36.12%** and **3.16%** of their reference totals.

These character counts union intersecting source spans within each reference span, then sum across questions; shared evidence can therefore count for more than one question. References are incomplete weak annotations, and character coverage is not semantic evidence recall.

The current PersonaMem memory units center on user text, with at most 24 wordpieces of preceding assistant context for short replies. Full history includes complete assistant turns and source system profiles. Some evidence is consequently absent from the selector's candidate representation, and no ranking weights can recover excluded text.

The label reports also contain 2,057 training and 92 validation questions with no positive memories. Under the current exact-reference rule, unmapped references can make every pair negative. Review and repair these cases or create reviewed semantic labels before treating these zeros as evidence of irrelevance.

## Recommended experiments, in priority order

1. **Establish controls and choose models by the actual objective.** Compare MiniLM-only, BM25-only and LR with the same candidate units, order and 1,024-token packing on validation. Track reference-based Hit/Recall under the actual token budget and Qwen answer accuracy. Keep the current LR as the baseline. A lower global log loss is not sufficient evidence of better memory selection.

2. **Improve memory units and supervision.** Trial a top-ranked user turn plus its adjacent assistant turn, preserving speaker and time information and the same total token budget. For LoCoMo, both speakers are already candidates; focus on covering complete events and neighboring turns. Review unmatched references and high-scoring false negatives in V2 training/validation, with a small stratified pilot of semantic relevance labels. Do not use test answer labels as training features. An oracle-context experiment on reviewed validation questions can separate retrieval failure from Qwen/prompt failure.

3. **Add inexpensive scalar features while keeping LR lightweight.** Candidates include per-question BM25/cosine percentile or rank, `log1p(BM25)`, BM25–cosine interaction, exact entity/speaker match, relative turn position, chunk length, and visible update/forget/negation indicators. Recency must be conditional on the question: historical and preference-evolution questions can need old and new evidence together. Feature choices and weights require validation, not assumed positive signs.

4. **Train for within-question ranking.** Compare query-balanced pointwise weighting with a pairwise logistic objective on positive-minus-negative feature vectors from the same question. Prefer reviewed hard negatives near the top of the current ranker; incomplete weak labels can otherwise make hard-negative training harmful. Score each memory once at inference. Select using ranking/answer metrics at the frozen budget. The probabilistic pairwise ranking objective in the [RankNet paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf) provides a standard reference; using a linear scoring function here is a proposed adaptation, not a claim that a neural ranker has been trained.

5. **Test 1,536 and 2,048 memory tokens on validation.** Current 1,024-token packing selects about 10.3 / 14.7 / 12.0 memories for V1 / V2 / LoCoMo. LoCoMo offers about 601 candidates per question. A larger or confidence-dependent budget, adjacent-turn bundles, or a penalty for redundant selections may recover evidence, at a resource cost. Measure that cost. Existing exact prompt/context checks must remain; larger budgets are hypotheses, not measured improvements.

6. **Use class weighting as a controlled secondary experiment.** `--class-weight balanced` is already available; scikit-learn weights classes inversely to their frequencies. This can change the learned ranking and calibration but is not a guaranteed fix, and shifting only the intercept cannot improve ordering. See the [official LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). Choose it by validation ranking/answer metrics, not by getting more probabilities above 0.5.

Use the 40-persona validation set to choose changes. Keep current V2 test, V1 and LoCoMo comparisons for final reporting; this examination of their errors is exploratory. A LoCoMo-specific tuning set would need an explicit new evaluation design.

## Resource results and comparability

| Dataset | Full / LR elapsed seconds per question | Full / LR mean GPU MiB | Full / LR mean correct-label NLL |
| --- | ---: | ---: | ---: |
| V1 | 4.542 / 0.744 | 4,448 / 3,810 | 1.133 / 1.078 |
| V2 | 2.047 / 0.852 | 4,818 / 3,834 | 1.947 / 1.847 |
| LoCoMo | 0.826 / 0.774 | 5,321 / 3,904 | 1.295 / 1.371 |

These are descriptive saved-run averages. V1 full history combines cached and uncached runtime segments. Full history and LR use different context allocations and recorded launcher/template options, including explicit thinking-template handling in the LR runs. PersonaMem full history includes source system profiles that retrieval excludes. Standardize applicable answer/template settings and document source-input differences before making a causal claim about a small accuracy delta.

GPU measurements are device-wide and include desktop/other processes; server RAM excludes Python. Retrieval features are precomputed, so saved selection timings exclude feature-generation cost for a genuinely new question. Warm full-history prefix reuse also explains why LoCoMo's much shorter LR prompts produce only a modest elapsed-time difference. Optimized LR packing is already around 25–30 ms per question; further packing-speed work is not the main accuracy bottleneck.

The 800-persona training set is the explicitly approved custom split containing 41 reassigned benchmark-source personas. Current held-out validation/test persona membership remains disjoint; describe results as the prepared custom subset rather than an untouched full official V2 benchmark protocol.

## Local sources

- [800-persona training report](C:/Users/jaden/Code/ECS271/models/lr_selector/training_report.json)
- [Current validation pair predictions](C:/Users/jaden/Code/ECS271/models/lr_selector/validation_predictions.jsonl)
- [Archived 100-persona training report](C:/Users/jaden/Code/ECS271/models/lr_selector_100_train_20_val/training_report.json)
- [V1 full-history summary](C:/Users/jaden/Code/ECS271/results/qwen_full_history/personamem_v1/32k/comparison/summary.json)
- [V1 LR summary](C:/Users/jaden/Code/ECS271/results/qwen_lr/personamem_v1/32k/budget_1024/comparison/summary.json)
- [V2 full-history summary](C:/Users/jaden/Code/ECS271/results/qwen_full_history/personamem_v2/32k/compact/test/summary.json)
- [V2 LR summary](C:/Users/jaden/Code/ECS271/results/qwen_lr/personamem_v2/32k/compact/budget_1024/test/summary.json)
- [LoCoMo full-history summary](C:/Users/jaden/Code/ECS271/results/qwen_full_history/locomo_mc10/full/comparison/summary.json)
- [LoCoMo LR summary](C:/Users/jaden/Code/ECS271/results/qwen_lr/locomo_mc10/full/budget_1024/comparison/summary.json)
- [V2 preparation/source policy](C:/Users/jaden/Code/ECS271/data/processed/v2/preparation_report.json)
- [Memory ranking and packing](C:/Users/jaden/Code/ECS271/scripts/lr_selection.py)
- [Reference-overlap label generation](C:/Users/jaden/Code/ECS271/scripts/create_relevance_labels.py)
