# Consolidated BM25 and LightGBM selector report

**Selected training architecture: 55-feature LightGBM LambdaRank, including relative conversation position.**

Current source: `scripts/ranker55.py` contains the training architecture. `scripts/build_ranker_features.py` builds the external bundle under `data/ranker55/`, and newly trained weights go to `models/ranker55.txt`. The earlier embedded model, statistics and payload were removed at the user's request. Recorded accuracy numbers below describe that earlier checkpoint and do not measure a newly trained model.

The user explicitly selected the relative-position model after the position experiment. The distance-from-latest feature is excluded. This decision supersedes older archived statements that the 54-feature model remains selected; historical results and prior gate outcomes are preserved as recorded.

| Model | Development Hit@10 (5,107 questions) | Historical Hit@10 (1,815 questions) | Historical NDCG@10 |
|---|---:|---:|---:|
| Original 54 features | 68.93% | 70.19% | 0.4105 |
| Selected 55: relative position | **69.77%** | **71.57%** | **0.4226** |
| Distance only, discarded | 69.92% | 71.18% | 0.4222 |
| Both position features, discarded | 69.75% | 71.35% | 0.4228 |

The selected model adds +1.38 historical Hit@10 points, paired persona-bootstrap95 [+0.34, +2.42], rescuing 58 questions and losing 33. The historical cohort was previously inspected. Development improves by +0.84 points, below the original >1-point adoption threshold. Selection is an explicit user decision rather than a claim that the old gate passed. No fresh confirmation or validation/test evaluation was performed during consolidation. The 80% goal has not been reached.

## Retained files and usage

- `REPORT.md`: this consolidated report, all 15 original Markdown documents, essential metrics, provenance and packaging verification.
- `scripts/ranker55.py`: training source, ordered 55-feature schema, feature construction, fixed LightGBM parameters and training functions; moved out of the experiment folder. `scripts/build_ranker_features.py` supplies its feature NPZ, labels, corpus statistics and embeddings.

The shared `.venv` and `models/embeddinggemma-2` directory remain outside the cleaned experiment folder. NumPy and LightGBM are sufficient for training from prepared features. Constructing dense features additionally requires externally supplied pinned embeddings.

```powershell
.venv/Scripts/python.exe -B scripts/ranker55.py --describe
.venv/Scripts/python.exe -B scripts/build_ranker_features.py
.venv/Scripts/python.exe -B scripts/ranker55.py
```

The external NPZ must contain `X` (candidate rows × 55), `y` (binary reference labels), `offsets` (contiguous question boundaries starting at zero and ending at the row count), and `feature_names` (exact ordered schema). The output model is created when training runs; an existing model output is overwritten atomically after successful training. A failed fit or save preserves the previous output. Output paths cannot replace the training input or this script. Programmatic `feature_matrix(question, messages, token_lengths, statistics, embed)` builds features from externally supplied statistics/embeddings. `train_ranker` fits arrays directly. Keep questions together and separate personas when evaluating folds.

## The 55-feature architecture

LightGBM LambdaRank uses 300 trees, maximum depth 5, up to 31 leaves/tree, learning rate 0.05, minimum 100 samples/leaf, L2 regularization 5, max_bin 63 and binary gains [0,1]. Fixed training seed is 1271, deterministic column-wise training, four CPU threads and truncation level 13. The model was fitted on 5,107 non-forget training questions from 280 personas, with 1,167,923 candidate rows and 14,443 exact-reference positives.

Feature groups: 28 base features (whole cosine, metadata/role/length, whole BM25, lexical chunks, question-frequency weights and RRF), 20 directional neighbor features (±1/±2 score/rank/availability), 6 dense chunk features, and 1 relative-position feature. Neighbor scores remain question-to-neighbor measurements; no direct message-to-message semantic similarity was added.

Relative position = (current original source index - earliest visible source index) / (latest visible source index - earliest visible source index). Singleton histories use zero. Source gaps are preserved, and future/invisible messages never enter the calculation. This measures conversation order, not elapsed wall-clock time.

Whole-message BM25 uses the question-visible corpus, b=0 and k1=2. Chunk BM25 uses frozen global parent-message DF, b=0 and k1=1.5, conversational stopwords, 48-word windows with stride 24 for messages over 64 lexical tokens, and max/top-two mean aggregation. Question weights are clipped inverse question frequency in [0.1,1]. Equal RRF uses c=60, no lexical vote for zero, and mean reciprocal votes for positive lexical ties. Final model score ties use original source order.

Dense cosine uses the pinned 768-dimensional EmbeddingGemma text backbone, the original search/memory prefixes, attention-mask mean pooling including prefixes, then L2 normalization. Inputs over 1,200 complete tokens fail rather than truncate. Dense chunks use source spans with NFKC/casefold normalization while preserving punctuation. Original messages remain the retrieval/output unit.

Ordered inputs:

| # | Feature |
|---:|---|
| 1 | `cosine` |
| 2 | `cosine_rank_fraction` |
| 3 | `cosine_zscore` |
| 4 | `cosine_gap` |
| 5 | `role_assistant` |
| 6 | `log_lexical_length` |
| 7 | `log_content_tokens` |
| 8 | `bm25_b0` |
| 9 | `bm25_rank_fraction` |
| 10 | `bm25_zscore` |
| 11 | `bm25_gap` |
| 12 | `winning_fusion` |
| 13 | `chunk_max` |
| 14 | `chunk_rank_fraction` |
| 15 | `chunk_zscore` |
| 16 | `chunk_gap` |
| 17 | `chunk_top2mean` |
| 18 | `log_chunk_count` |
| 19 | `qweight_bm25` |
| 20 | `qweight_bm25_rank_fraction` |
| 21 | `qweight_bm25_zscore` |
| 22 | `qweight_bm25_gap` |
| 23 | `qweight_fusion` |
| 24 | `qweight_chunk_max` |
| 25 | `qweight_chunk_rank_fraction` |
| 26 | `qweight_chunk_zscore` |
| 27 | `qweight_chunk_gap` |
| 28 | `qweight_chunk_top2mean` |
| 29 | `neighbor_minus2_cosine` |
| 30 | `neighbor_minus2_cosine_rank` |
| 31 | `neighbor_minus2_weighted_bm25` |
| 32 | `neighbor_minus2_weighted_bm25_rank` |
| 33 | `neighbor_minus2_available` |
| 34 | `neighbor_minus1_cosine` |
| 35 | `neighbor_minus1_cosine_rank` |
| 36 | `neighbor_minus1_weighted_bm25` |
| 37 | `neighbor_minus1_weighted_bm25_rank` |
| 38 | `neighbor_minus1_available` |
| 39 | `neighbor_plus1_cosine` |
| 40 | `neighbor_plus1_cosine_rank` |
| 41 | `neighbor_plus1_weighted_bm25` |
| 42 | `neighbor_plus1_weighted_bm25_rank` |
| 43 | `neighbor_plus1_available` |
| 44 | `neighbor_plus2_cosine` |
| 45 | `neighbor_plus2_cosine_rank` |
| 46 | `neighbor_plus2_weighted_bm25` |
| 47 | `neighbor_plus2_weighted_bm25_rank` |
| 48 | `neighbor_plus2_available` |
| 49 | `dense_chunk_max` |
| 50 | `dense_chunk_rank` |
| 51 | `dense_chunk_zscore` |
| 52 | `dense_chunk_gap` |
| 53 | `dense_chunk_top2mean` |
| 54 | `dense_chunk_minus_whole` |
| 55 | `relative_conversation_position` |

## Provenance and cleanup scope

- Selected model SHA-256: `15a3852fcd09bc3bf7d070ccc13f49571e4209af82e9198d57fcf9c7f012fe93`.
- Original frozen statistics SHA-256: `c66863f7b187fc99725a01f81998c52961552cb2379e301b69dc90906127499f`.
- Embedded asset payload SHA-256: `7cb00c80ea1901c6e4d4ee7d86f8173d048f5048d15da83632358d900b8e9a50`.
- Encoder: `google/embeddinggemma-2` revision `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- Raw datasets, split IDs, main scripts/configuration, the shared environment and primary neural model assets are outside the cleanup scope.
- Old experiment scripts, alternative model weights, cached embeddings, arrays, logs and individual reports are removed after standalone reconstruction succeeds. Removed historical artifact references are rendered as plain provenance text; links between retained report sections point within this file.

## Position experiment: exact tables

| method | features | questions | hit10 | ndcg10 | tail_positives | tail_rescued | tail_rescue10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline54 | 54 | 5107 | 0.6892500489524183 | 0.38789715050992907 | 6769 | 1410 | 0.20830255576894666 |
| distance55 | 55 | 5107 | 0.6992363422753084 | 0.40215423384115406 | 6769 | 1470 | 0.2171664943123061 |
| relative55 | 55 | 5107 | 0.6976698648913257 | 0.40138625475182005 | 6769 | 1478 | 0.21834835278475404 |
| both56 | 56 | 5107 | 0.6974740552183278 | 0.40276042105175497 | 6769 | 1493 | 0.2205643374205939 |
| random_exact | 0 | 5107 | 0.11949813633340779 | 0.02740076753968231 | 6769 | 295.29696391461704 | 0.043624902336329895 |

| method | reference | delta_hit_pp | delta_hit_ci_low_pp | delta_hit_ci_high_pp | delta_ndcg | passes_gate | rescued | lost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distance55 | baseline54 | 0.9986293322890151 | 0.36935518904803505 | 1.6132207374125116 | 0.014257083331224914 | False | 168 | 117 |
| relative55 | baseline54 | 0.8419815938907381 | 0.2177055473469617 | 1.4509354715540281 | 0.013489104241890907 | False | 163 | 120 |
| both56 | baseline54 | 0.8224006265909536 | 0.15915653189226567 | 1.4694426436574075 | 0.014863270541825818 | False | 164 | 122 |
| both56 | distance55 | -0.17622870569806148 | -0.7219547419127588 | 0.356437408446727 | 0.0006061872106008992 | False | 92 | 101 |
| both56 | relative55 | -0.019580967299784608 | -0.5301498243884035 | 0.4974172565525158 | 0.0013741662999349354 | False | 91 | 92 |

| method | features | questions | hit10 | ndcg10 | tail_positives | tail_rescued | tail_rescue10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline54 | 54 | 1815 | 0.7019283746556474 | 0.4105444208134589 | 2298 | 506 | 0.22019147084421237 |
| distance55 | 55 | 1815 | 0.7118457300275483 | 0.42217706380239617 | 2298 | 522 | 0.22715404699738903 |
| relative55 | 55 | 1815 | 0.715702479338843 | 0.42261629162659137 | 2298 | 537 | 0.23368146214099217 |
| both56 | 56 | 1815 | 0.7134986225895317 | 0.42281308907479115 | 2298 | 547 | 0.2380330722367276 |
| random_exact | 0 | 1815 | 0.1177151740057798 | 0.027181221764318 | 2298 | 100.20012688338565 | 0.043603188373971126 |

| method | reference | delta_hit_pp | delta_hit_ci_low_pp | delta_hit_ci_high_pp | delta_ndcg | passes_gate | rescued | lost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distance55 | baseline54 | 0.9917355371900827 | 0.055741360089186176 | 1.9230769230769231 | 0.011632642988937252 | False | 52 | 34 |
| relative55 | baseline54 | 1.3774104683195594 | 0.34048844746417845 | 2.4225534141846228 | 0.012071870813132395 | True | 58 | 33 |
| both56 | baseline54 | 1.1570247933884297 | 0.16180793856712422 | 2.186106717684881 | 0.012268668261332216 | True | 55 | 34 |

### Original packaging and experiment checks

```json
{
  "frozen_inputs_code_and_protocol_unchanged": true,
  "models_unchanged": true,
  "original_model_profile_and_source_data_unchanged": true,
  "persona_disjoint_oof": true,
  "secondary_personas_disjoint_from_fit": true,
  "secondary_previously_inspected": true,
  "validation_test_evaluated": false,
  "GPU_used": false
}
```

```json
{
  "question_id": "v2:train:00062",
  "messages": 223,
  "new_features_exact": true,
  "saved_model_scores_max_abs_error": {
    "baseline54": 0.0,
    "distance55": 0.0,
    "relative55": 0.0,
    "both56": 0.0
  },
  "answer_reference_and_future_metadata_cannot_change_position_features": true,
  "preexisting_protected_changes_recorded": 1
}
```

```text
test_absolute_distance_differs_at_same_relative_position (test_study.PositionTests.test_absolute_distance_differs_at_same_relative_position) ... ok
test_invalid_visibility_or_order (test_study.PositionTests.test_invalid_visibility_or_order) ... ok
test_regular_and_gapped_source_indices (test_study.PositionTests.test_regular_and_gapped_source_indices) ... ok
test_single_message (test_study.PositionTests.test_single_message) ... ok
test_translation_invariance (test_study.PositionTests.test_translation_invariance) ... ok
test_visible_history_only_and_metadata_independence (test_study.PositionTests.test_visible_history_only_and_metadata_independence) ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.275s

OK

```

## Archived report index

1. [confirmation/FINDINGS.md](#archived-report-01) — original SHA-256 `9c86fd9fcebdb384cead935f46e86dbe0ba1c1ca2bb66230bf3eb168b6496e7d`
2. [context_refinement/FINDINGS.md](#archived-report-02) — original SHA-256 `7138f31e476a75c1ba46a41b44a864aef2524766b89987608672119af3a3c70d`
3. [FINDINGS.md](#archived-report-03) — original SHA-256 `f1a6b169f7c671b8ddec901a8fc67ce39ea4132ffd28987ddc220d0c4fee0f9f`
4. [IMPROVED_PLAN.md](#archived-report-04) — original SHA-256 `324ada7b7fb613fa5351b8c0eb228ba84d8aefe85b3501471fe5e76b0795e18d`
5. [next_stage/FINDINGS.md](#archived-report-05) — original SHA-256 `cef46b2d7a296d5fc016dd8606d514a5b6b09ea33faac72160c811fe4f54bd69`
6. [next_stage/models/ms-marco-MiniLM-L6-v2/README.md](#archived-report-06) — original SHA-256 `7c0ec39941d0d1d766ccde671f592971d82fe45b7eea550a80d1ea864ebc3baa`
7. [position_features/FINDINGS.md](#archived-report-07) — original SHA-256 `cbc00a994f08961722f0a1a245db5ec683a3c3005a16890b7017f6d41d83fecb`
8. [question_weight_role/FINDINGS.md](#archived-report-08) — original SHA-256 `8ff6454cef44830b48f6d13d9e6ccbe91f3fa9a9c7be1f8b742e0af1311cd217`
9. [ranker54_failure_audit/FINDINGS.md](#archived-report-09) — original SHA-256 `74a52a72c22fd5f50255a1c640f2c6d448ce9075342b1d6af89401263188fe9d`
10. [ranker54_failure_audit/PROPOSAL_REVIEW.md](#archived-report-10) — original SHA-256 `e1d600fc258b1f9540e8a7d0ab961d8f3ae601ee74c568a27d7b6e725b07c21c`
11. [README.md](#archived-report-11) — original SHA-256 `cf5b525190405f4d0cadc020d45d63fa01246469c71d6b6e10f282c0bf676cf1`
12. [relevance_methods/FINDINGS.md](#archived-report-12) — original SHA-256 `04df1ab793c6989f41165fc2b263080bcfdda17a2fec6ca7ad5524e5df43a60e`
13. [target80_review/IMPROVED_PLAN.md](#archived-report-13) — original SHA-256 `af89175315d828fe4d412db16b9529184b8319453a2888faaeb11cae890591df`
14. [target80_run/FINDINGS.md](#archived-report-14) — original SHA-256 `712fe3e140f4195d5ae0c0d320c4aa9f1460fca9bc3728e58f4944e2224ea590`
15. [target80_run/SUPPLEMENT.md](#archived-report-15) — original SHA-256 `f407488d4f4612efa2e7a493d29b205117547133951d2bd54e5b0884eaddbd5f`

---

<a id="archived-report-01"></a>

## Archived source: confirmation/FINDINGS.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Complete winning fusion: confirmation and ranker results

**The winning fusion cleared the frozen confirmation gate.** It achieved **43.62% Hit@10**, compared with **35.02%** for cosine on 1,882 fresh non-forget questions: **+8.61 percentage points**, paired 95% interval **[+6.60, +10.61]**.

**Chunk features cleared both retention gates and merit retention in the tested ranker.** Its Hit@10 is **60.63%**, +1.22 points versus the same ranker without chunks and +17.00 points versus the complete working fusion.

## All three requested actions

1. **Use the complete winning method:** working_baseline.json (historical artifact: `../working_baseline.json`) selects complete-message EmbeddingGemma 2 cosine plus local-message BM25 (`b=0`, `k1=2`), combined by equal RRF with constant 60. Zero lexical scores contribute zero; tied positive lexical ranks share the mean reciprocal contribution. Chronological order resolves final ties. winning_combination.py (historical artifact: `../winning_combination.py`) implements this method for reusable inference and is used by this experiment.
2. **Fresh-persona confirmation:** 1,882 non-forget questions from 100 training personas disjoint from the original 80. Settings were frozen before confirmation metrics. Global message DF used 142,832 messages from 620 other training personas, excluding both fitting and confirmation groups. Validation/test were unused.
3. **Small matched rankers:** trained on 1,450 original-pilot questions. All three models use the same fixed 150-tree LightGBM LambdaRank recipe. No early stopping or setting selection used confirmation metrics. Fourfold persona-grouped OOF training diagnostics were also computed with the same fixed recipe.

## Fresh confirmation results

| Method | Hit@10 | NDCG@10 | Delta Hit vs cosine (pp) | Delta Hit vs working fusion (pp) | Tail rescue@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cosine alone | 35.02% | 0.1272 | +0.00 | -8.61 | 0.00% |
| Default BM25 alone | 28.59% | 0.0922 | -6.43 | -15.04 | 6.69% |
| BM25 b=0/k1=2 alone | 25.93% | 0.0811 | -9.09 | -17.69 | 7.41% |
| Chunk-max BM25 alone | 35.81% | 0.1274 | +0.80 | -7.81 | 10.43% |
| Original default RRF | 35.87% | 0.1215 | +0.85 | -7.76 | 1.05% |
| Complete winning fusion (working baseline) | 43.62% | 0.1676 | +8.61 | +0.00 | 3.22% |
| Global + stoplist + b=0 whole-message fusion | 44.58% | 0.1762 | +9.56 | +0.96 | 3.46% |
| Matching chunk-max fusion | 44.53% | 0.1694 | +9.51 | +0.90 | 3.83% |
| Cosine + metadata ranker | 58.34% | 0.2499 | +23.33 | +14.72 | 2.62% |
| Winning fusion feature ranker | 59.40% | 0.2581 | +24.39 | +15.78 | 4.11% |
| Winning fusion + chunk feature ranker | 60.63% | 0.2678 | +25.61 | +17.00 | 5.64% |
| Exact random baseline | 12.12% | 0.0276 | -22.89 | -31.50 | 4.40% |

Most of the large ranker gain is already present in the cosine + metadata control (58.34% Hit@10). The base lexical group adds +1.06 points to that control, with a confidence interval crossing zero. The additional chunk feature group adds +1.22 points to the matched base ranker and narrowly passes the predeclared gate. The full improvement over raw cosine should not be attributed to BM25 or chunks alone.

Chunk fusion and its matched whole-message control perform similarly in Hit@10, with lower NDCG for chunk fusion. The demonstrated incremental chunk benefit is in the learned feature combination.

Hit/NDCG average questions equally; NDCG uses binary reference labels. Tail rescue counts positive occurrences whose cosine rank exceeds 50, not entire failed questions. Random Hit and NDCG account for each question's actual candidate and positive counts.

The original 42.00% was measured on the pilot sample. Compare methods on the same fresh sample above; a different absolute percentage across samples is not a measured improvement or regression.

## Frozen decision gates

Each gate requires strictly more than +1.0 percentage point Hit@10, no mean NDCG@10 regression, and a positive lower paired 95% persona-bootstrap Hit bound. Bootstrap uses 10,000 resamples. The primary hypothesis is complete winning fusion vs cosine; chunk retention requires both chunk comparisons below to pass. Other comparisons remain diagnostic.

| Comparison | Delta Hit (pp) | 95% paired interval (pp) | Delta NDCG | Gate |
| --- | ---: | ---: | ---: | --- |
| Complete winning fusion (working baseline) vs Cosine alone | +8.61 | [+6.60, +10.61] | +0.0405 | PASS |
| Winning fusion feature ranker vs Complete winning fusion (working baseline) | +15.78 | [+13.66, +17.85] | +0.0905 | PASS |
| Winning fusion + chunk feature ranker vs Complete winning fusion (working baseline) | +17.00 | [+14.91, +19.07] | +0.1002 | PASS |
| Winning fusion + chunk feature ranker vs Winning fusion feature ranker | +1.22 | [+0.06, +2.43] | +0.0097 | PASS |
| Winning fusion feature ranker vs Cosine + metadata ranker | +1.06 | [-0.38, +2.53] | +0.0082 | FAIL |
| Matching chunk-max fusion vs Global + stoplist + b=0 whole-message fusion | -0.05 | [-1.16, +1.04] | -0.0068 | FAIL |

The matched whole-message control shares chunk scoring's global parent-message IDF, stoplist, `b=0` and `k1=1.5`. It isolates the chunk view from those other changes. The chunk ranker ablation includes max/top-two scores, associated ranks/statistics, and chunk count as one feature group.

## Rescued and displaced cosine successes

| Method | Cosine misses rescued | Cosine hits lost | Net questions | Tail positives retrieved |
| --- | ---: | ---: | ---: | ---: |
| Complete winning fusion (working baseline) | 286 | 124 | +162 | 80/2483 |
| Matching chunk-max fusion | 319 | 140 | +179 | 95/2483 |
| Cosine + metadata ranker | 501 | 62 | +439 | 65/2483 |
| Winning fusion feature ranker | 513 | 54 | +459 | 102/2483 |
| Winning fusion + chunk feature ranker | 539 | 57 | +482 | 140/2483 |

## Original training-persona OOF diagnostics

These are descriptive checks from the original development personas. They did not select hyperparameters, iterations or the confirmation sample.

| Method | Original OOF Hit@10 | Fresh confirmation Hit@10 |
| --- | ---: | ---: |
| Complete winning fusion (working baseline) | 42.00% | 43.62% |
| Cosine + metadata ranker | 57.59% | 58.34% |
| Winning fusion feature ranker | 57.93% | 59.40% |
| Winning fusion + chunk feature ranker | 59.24% | 60.63% |

## Confirmation by question type

| Type | Questions | Cosine Hit@10 | Working fusion Hit@10 | Base ranker Hit@10 | Chunk ranker Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| anti_stereotypical_pref | 475 | 42.11% | 49.47% | 64.63% | 65.89% |
| health_and_medical_conditions | 286 | 32.52% | 46.50% | 59.44% | 63.29% |
| neutral_preferences | 459 | 30.72% | 35.95% | 53.81% | 54.03% |
| stereotypical_pref | 301 | 38.54% | 44.85% | 61.13% | 61.46% |
| therapy_background | 361 | 30.19% | 42.38% | 58.17% | 59.28% |

Gold question type only filters/stratifies the evaluation. It is not an inference feature. Metadata consists of role and original lexical/subword lengths; the subword tokenizer is EmbeddingGemma and these lengths are not Qwen packing costs.

## Saved deliverables and verification

- Working baseline configuration and reusable complete-message inference implementation are one directory above this report.
- `winning_fusion_top10.jsonl` contains actual top-ten outputs from the complete winning combination for every fresh question. `base_ranker_top10.jsonl` and `chunk_ranker_top10.jsonl` contain the learned outputs.
- `models/` contains all three fitted models. `training_report.json` stores feature order, parameters, model hashes and descriptive importance. `train_features.npz` / `confirmation_features.npz` cache aligned scores, features, labels and question offsets.
- `metrics.csv`, `comparisons.csv`, `by_question_type.csv` and `per_question_metrics.jsonl` preserve the results with denominators and paired comparisons.
- All 1,450 original winning rankings were reproduced exactly. Eight confirmation tests cover recipe parity, zero scores, label independence, source ordering, full-question group selection, feature isolation, paired bootstrap gates and tail metrics. Source data, protected scripts/configs/splits and original pilot inputs passed hash verification.
- This is training-persona evidence. No downstream answer-accuracy result or final validation/test claim is made. Further tuning cannot reuse these confirmation results as untouched evaluation.

Reproduce from the project root (encoding can run alongside training-feature creation):

```powershell
.venv/Scripts/python.exe -B bm25_test/run_confirmation.py prepare
.venv/Scripts/python.exe -B bm25_test/run_confirmation.py encode
.venv/Scripts/python.exe -B bm25_test/run_confirmation.py features_train
.venv/Scripts/python.exe -B bm25_test/run_confirmation.py train
.venv/Scripts/python.exe -B bm25_test/run_confirmation.py features_confirmation
.venv/Scripts/python.exe -B bm25_test/run_confirmation.py evaluate
.venv/Scripts/python.exe -B bm25_test/make_confirmation_report.py
```

Use the complete winning method on a question-visible JSONL history:

```powershell
.venv/Scripts/python.exe -B bm25_test/winning_combination.py --messages PATH_TO_MESSAGES.jsonl --question "Your question" --output bm25_test/retrieved.json
```

The history schema is one object per original message with `message_id`, `message_index`, `role` and `text`, in source order. The inference command computes cosine with the pinned local encoder, computes local BM25, and emits the complete fused top ten.


---

<a id="archived-report-02"></a>

## Archived source: context_refinement/FINDINGS.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Five-stage ranker refinement

Decision: **keep_existing_54_feature_model**. All five steps were addressed through CPU ablations, a bounded existing-encoder sentence pilot, a separate evidence audit, and learned/gated use of cached expansions.

Development uses the same 5,107 non-forget questions and 280 personas as relevance_methods. Shared frequencies exclude all 380 previously inspected and 100 reserved personas. The historical 1,815-question audit was not used for tuning. These development percentages are not directly comparable with historical 70.19% confirmation Hit@10.

## 1. Channel strength and disagreement

Six additional features describe the best lexical/dense ranks, rank disagreement and dominance. Zero-score lexical matches cannot receive a confident rank through chronological ties.

## 2. Final first-pass neighbor scores

The baseline already has neighboring cosine/BM25 features. This experiment adds first-pass ranker predictions, including four self-score summaries and ten previous/next score/rank/availability/role features. The self-only control distinguishes general stacking from an increment attributable to neighbors. Twelve inner first-pass models exclude both their predicted personas and the outer evaluation personas. Held outer questions reuse the identical frozen 54-feature baseline OOF predictions.

| Development method | Features | Hit@10 | NDCG@10 | Tail rescue |
|---|---:|---:|---:|---:|
| baseline54 | 54 | 68.59% | 0.3860 | 20.65% |
| channel60 | 60 | 68.44% | 0.3848 | 20.36% |
| self58 | 58 | 69.06% | 0.3894 | 22.65% |
| neighbor68 | 68 | 68.32% | 0.4334 | 29.59% |
| channel_self64 | 64 | 68.61% | 0.3875 | 22.12% |
| channel_neighbor74 | 74 | 68.34% | 0.4347 | 29.72% |

| Comparison | Hit change pp | Persona 95% CI | NDCG change | Gate |
|---|---:|---|---:|---|
| channel60 vs baseline54 | -0.16 | [-0.67, +0.35] | -0.0012 | FAIL |
| self58 vs baseline54 | +0.47 | [-0.22, +1.17] | +0.0035 | FAIL |
| neighbor68 vs baseline54 | -0.27 | [-1.06, +0.51] | +0.0474 | FAIL |
| channel_self64 vs baseline54 | +0.02 | [-0.61, +0.65] | +0.0015 | FAIL |
| channel_neighbor74 vs baseline54 | -0.25 | [-1.03, +0.52] | +0.0488 | FAIL |
| neighbor68 vs self58 | -0.74 | [-1.60, +0.14] | +0.0439 | FAIL |
| channel_neighbor74 vs channel_self64 | -0.27 | [-1.10, +0.56] | +0.0473 | FAIL |
| channel_self64 vs self58 | -0.45 | [-1.02, +0.14] | -0.0020 | FAIL |
| channel_neighbor74 vs neighbor68 | +0.02 | [-0.47, +0.50] | +0.0013 | FAIL |

The neighbor model rescues 197 questions but loses 211 relative to the baseline. It returns 926 additional reference messages on questions both models already hit. This helps explain higher NDCG and tail rescue without improved question coverage; it does not establish a causal mechanism or downstream answer benefit. The LambdaRank objective rewards ranked reference gains, which need not align perfectly with question-level Hit@10. Coverage diagnostics (historical artifact: `reference_coverage_diagnostics.json`).

CPU candidate after group pruning: **baseline54**. The gate remains >1 absolute Hit point, a positive paired-persona bootstrap lower 95% bound, and no NDCG regression.

## 3. Sentence evidence pilot

The existing pinned EmbeddingGemma encoded 20,154 unique exact-source spans from 3,616 messages across 200 development questions' full candidate pools. Sentences beginning with a pronoun/deictic cue receive their preceding sentence as context. Spans over96 words are split into64-word windows, stride32. No original candidate message is truncated or removed.

Encoding took 77.5s; projected full-development encoding is 0.44h (count-based estimate, conservatively ignoring cross-message text deduplication). Peak allocated CUDA memory was 901MiB. Maximum input length was 122 model tokens; truncated inputs: zero.

Sentence maximum, top-two mean and log span count enter a fixed regularized residual scorer. The first-pass baseline query-zscore coefficient is fixed at 1; three bounded coefficients are fitted to other outer-fold pilot questions only. This is a bounded residual pilot, not a full-data augmented LightGBM ablation. New spans preserve original casing; the older long-message chunk view used NFKC/casefold, so this is a concrete representation comparison rather than a pure sentence-boundary ablation. The sentence recipe, count controls and coefficients are recorded in pilot_protocol.json.

## 4. Separate evidence-label audit

Twenty seeded questions include two hits and two misses per question type. The best-ranked reference and highest-ranked unlisted message were reviewed in full for each, giving 40 messages. Original reference membership and ranks were visible: this was a single agent review, not blind human annotation.

The reviewer assigned grade2 textual relevance to 15/20 references and 13/20 unlisted candidates. 6/10 sampled misses had a grade2 unlisted candidate. These are descriptive counts from a deliberately balanced, small sample—not estimates of dataset label error or downstream answer correctness.

Attribution/usefulness uncertainty was flagged in 29/40 messages. Categories: generic_topical: 10, fiction_or_uncertain: 3, quoted_self_report: 12, third_party: 5, explicit_user: 6, implied_interest: 3, wrapper: 1.

Examples include digital-game preferences described by a friend rather than the user, useful assistant rewrites of commute biographies, and a highly relevant unlisted description of writing difficulties. All support quotes are exact source substrings. No original labels were overwritten, no audit judgment entered training, and no Qwen annotator was run. Further supervision would require independent review and a separate evaluation of answer usefulness.

## 5. Learned and gated expansion pilot

The prior 200 query-only local-Qwen expansions were reused. No new generation or hypothetical persona facts were added. Expanded BM25 uses the newly isolated background IDF and original-query retention. A bounded residual weight is learned from other persona folds rather than applied with fixed RRF.

The miss-probability gate is fitted to the outer training personas using their inner-OOF first-pass predictions. Its inference inputs are normalized score gaps, top-ten weighted query coverage and lexical/dense rank overlaps. It never uses gold-reference overlap. Expansion applies at estimated miss probability >=0.5.

Gate activation: 5.0% of all development questions, 3.5% of pilot questions. Brier score 0.2039, compared with 0.2154 for a constant-prevalence diagnostic. These probabilities are model estimates; calibration is not guaranteed.

| 200-question pilot | Hit@10 | NDCG@10 | Tail rescue |
|---|---:|---:|---:|
| baseline | 68.50% | 0.4116 | 22.05% |
| sentence | 67.00% | 0.3750 | 21.65% |
| expansion | 68.50% | 0.4037 | 20.87% |
| expansion_gated | 68.50% | 0.4109 | 22.05% |

| Pilot comparison | Hit change pp | Persona 95% CI | NDCG change | Gate |
|---|---:|---|---:|---|
| sentence vs baseline | -1.50 | [-6.28, +3.05] | -0.0366 | False |
| expansion vs baseline | +0.00 | [-3.00, +3.27] | -0.0079 | False |
| expansion_gated vs baseline | +0.00 | [+0.00, +0.00] | -0.0007 | False |
| expansion_gated vs expansion | +0.00 | [-3.27, +3.00] | +0.0072 | False |

Pilot scale decisions: {'sentence': False, 'expansion': False, 'expansion_gated': False}. Gate and residual pilot judgments are development evidence only. A positive pilot would require full-development integration before confirmation; a failed pilot does not rule out every sentence or expansion recipe. The pilot has only 200 questions from 16 personas and broad confidence intervals; the gated arm activates on seven pilot questions, so it is especially limited. Fresh generation and HyDE were deferred because the tested cached integration did not clear the quality gate. Projected ungated new generation is 2.79 hours; the activated subset is cheaper, but had no observed Hit gain.

## Final decision and verification

No candidate cleared the development gates. The reserved 1,785-question confirmation was not labeled, encoded or evaluated. The prior 54-feature model with historical 70.19% Hit@10 remains selected.

All 6 new checks pass, including exact reconstruction of the 14 second-pass features and invariance to injected answer/reference annotations. Twelve inner fits and 20 outer candidate fits ran on CPU; sentence inference used the existing encoder with about 901 MiB peak allocated CUDA memory.

All new code, features, source-offset caches, predictions and audit judgments are in this folder. Main scripts, raw data, split IDs and previous studies were preserved. Validation/test were not used. No ColBERT/SPLADE models were downloaded. No quota replacement or automatic neighbor insertion was adopted.

CPU comparisons (historical artifact: `cpu_comparisons.csv`), pilot comparisons (historical artifact: `pilot_comparisons.csv`), evidence judgments (historical artifact: `evidence_judgments.jsonl`), gate (historical artifact: `gate_report.json`), selection (historical artifact: `selection.json`), verification (historical artifact: `verification.json`), tests (historical artifact: `tests.log`).


---

<a id="archived-report-03"></a>

## Archived source: FINDINGS.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# BM25 findings and revised plan

**The strongest screening signals are disabling BM25 length normalization and using chunk scores.** Cosine alone reaches **33.93%** non-forget Hit@10; existing RRF reaches **33.72%**; RRF with `b=0, k1=2` reaches **42.00% (+8.07 points)**; chunk-max RRF reaches **41.93% (+8.00 points)**. These results justify a bounded confirmation of Phase 1.

**The difficult cosine tail remains unresolved.** The two leading fusion settings recover only **3.29% / 3.34%** of >50-rank positive occurrences, below the **4.33%** exact random per-positive inclusion baseline. The overall Hit improvements do not establish a hard-tail solution.

This is a fresh, exploratory **single-message, training-only pilot**. Existing source scripts and historical results were left unchanged. All new artifacts are in this folder.

Sample: **1,835 questions / 80 training personas**, including **1,450 non-forget questions**, with 18,510 source message occurrences. Candidate pools span 165–286 messages (mean 232.3). Fresh complete-message EmbeddingGemma 2 vectors were computed; historical paired-memory cosine scores were not reused.

The adoption threshold was frozen in `protocol.json` before seeing results: **strictly greater than +1.0 percentage point non-forget Hit@10**, no NDCG regression, and a positive lower paired 95% persona-bootstrap bound on a later independent training-persona confirmation. The intervals below are pointwise screening intervals, not corrected for selecting among many configurations; no configuration is adopted from this run.

Global message and question statistics exclude the evaluated persona fold. All other active training personas provide fitting statistics. References only define evaluation positives. Validation/test questions and labels were not used to fit or select settings.

## Retrieval results

All metrics in this table exclude `ask_to_forget`. Tail rescue is positive-occurrence retrieval at ten conditional on that positive having cosine rank >50; it is different from rescuing an entire failed question. NDCG uses binary gains and all exact-reference positives in the ideal ranking.

| Method | Hit@10 | NDCG@10 | Delta Hit vs cosine (pp) | 95% delta interval (pp) | Tail rescue@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cosine alone | 33.93% | 0.1201 | +0.00 | [+0.00, +0.00] | 0.00% |
| BM25 default | 28.14% | 0.0862 | -5.79 | [-8.21, -3.42] | 6.31% |
| BM25 global IDF | 29.10% | 0.0945 | -4.83 | [-7.45, -2.27] | 6.26% |
| BM25 global IDF + question weights | 34.83% | 0.1319 | +0.90 | [-1.73, +3.52] | 8.47% |
| BM25 global IDF + stoplist | 32.28% | 0.1194 | -1.66 | [-4.27, +0.86] | 7.44% |
| BM25 global + stoplist + question weights | 35.10% | 0.1383 | +1.17 | [-1.57, +3.94] | 8.93% |
| BM25 chunk max | 34.07% | 0.1248 | +0.14 | [-2.59, +2.86] | 8.88% |
| BM25 chunk top-two mean | 34.14% | 0.1233 | +0.21 | [-2.52, +2.95] | 8.98% |
| BM25 b=0, k1=1.2 | 26.55% | 0.0827 | -7.38 | [-10.52, -4.14] | 7.80% |
| BM25 b=0, k1=2 | 25.38% | 0.0784 | -8.55 | [-11.85, -5.10] | 7.75% |
| BM25 previous + self | 30.55% | 0.1197 | -3.38 | [-6.15, -0.56] | 9.86% |
| BM25 self + next | 29.86% | 0.1405 | -4.07 | [-7.15, -1.06] | 10.37% |
| BM25 three-message view | 28.41% | 0.1335 | -5.52 | [-8.73, -2.26] | 10.37% |
| BM25 cosine feedback | 34.34% | 0.1361 | +0.41 | [-1.87, +2.73] | 7.29% |
| Existing equal RRF | 33.72% | 0.1166 | -0.21 | [-2.48, +1.98] | 0.87% |
| Tie-neutral, zero-aware RRF | 33.72% | 0.1166 | -0.21 | [-2.48, +1.98] | 0.87% |
| RRF + global IDF | 34.90% | 0.1230 | +0.97 | [-1.17, +3.24] | 1.03% |
| RRF + global IDF + question weights | 38.34% | 0.1472 | +4.41 | [+2.06, +6.58] | 1.54% |
| RRF + global IDF + stoplist | 37.86% | 0.1395 | +3.93 | [+1.75, +6.15] | 1.90% |
| RRF + global + stoplist + question weights | 40.55% | 0.1531 | +6.62 | [+4.25, +8.82] | 2.31% |
| RRF + chunk max | 41.93% | 0.1615 | +8.00 | [+5.60, +10.29] | 3.34% |
| RRF + chunk top-two mean | 41.38% | 0.1577 | +7.45 | [+5.06, +9.85] | 3.29% |
| RRF + BM25 b=0, k1=1.2 | 41.79% | 0.1602 | +7.86 | [+5.63, +10.03] | 3.08% |
| RRF + BM25 b=0, k1=2 | 42.00% | 0.1631 | +8.07 | [+5.80, +10.31] | 3.29% |
| RRF + previous + self | 39.17% | 0.1578 | +5.24 | [+2.97, +7.58] | 2.36% |
| RRF + self + next | 38.76% | 0.1715 | +4.83 | [+2.38, +7.23] | 1.80% |
| RRF + three-message view | 37.17% | 0.1686 | +3.24 | [+0.92, +5.60] | 1.33% |
| RRF + cosine feedback | 36.83% | 0.1426 | +2.90 | [+0.89, +4.77] | 0.67% |
| BM25, cosine as tie breaker | 28.14% | 0.0862 | -5.79 | [-8.21, -3.42] | 6.31% |
| BM25, expected random ties | 28.14% | 0.0862 | -5.79 | [-8.21, -3.42] | 6.31% |
| Exact random baseline | 11.99% | 0.0273 | -21.94 | [-24.66, -18.99] | 4.33% |

The full parameter grid and all rows are in metrics.csv (historical artifact: `metrics.csv`). Positive-rank median/p90 and Recall/Hit/full-reference coverage at 5/10/20/30/50 for all three fresh baselines are in baseline_rank_diagnostics.csv (historical artifact: `baseline_rank_diagnostics.csv`). These are standalone scoring/fusion comparisons, **not learned feature-group ranker ablations**. The scorer retains the current per-history average length when testing global IDF. Chunks/windows use parent-message global IDF, stop filtering and `b=0`; they are explicitly different auxiliary views, not one-variable ablations of the default.

## Net rescues, not just tail hits

| Method | Cosine misses rescued | Cosine hits lost | Net questions |
| --- | ---: | ---: | ---: |
| Existing equal RRF | 149 | 152 | -3 |
| Tie-neutral, zero-aware RRF | 149 | 152 | -3 |
| RRF + BM25 b=0, k1=2 | 207 | 90 | +117 |
| RRF + global + stoplist + question weights | 215 | 119 | +96 |
| RRF + chunk max | 228 | 112 | +116 |
| RRF + three-message view | 175 | 128 | +47 |
| RRF + cosine feedback | 149 | 107 | +42 |

## Exact-zero diagnostics

Default BM25 assigned exactly zero to **284/4,144 non-forget positive occurrences (6.85%)**. Across all types it was 315/5,682 (5.54%). Global IDF cannot create a match where no query term occurs.

Length uses the local EmbeddingGemma tokenizer on original message content, without prompt or special tokens; these counts are not Qwen packing costs. Buckets do not overlap.

| Non-forget message length | Positive occurrences | Exactly zero | Zero share |
| --- | ---: | ---: | ---: |
| <=64 | 1355 | 180 | 13.28% |
| 65-300 | 2401 | 104 | 4.33% |
| >300 | 388 | 0 | 0.00% |

| Question type | Positive occurrences | Exactly zero | Zero share |
| --- | ---: | ---: | ---: |
| anti_stereotypical_pref | 962 | 66 | 6.86% |
| ask_to_forget | 1538 | 31 | 2.02% |
| health_and_medical_conditions | 746 | 78 | 10.46% |
| neutral_preferences | 1079 | 57 | 5.28% |
| stereotypical_pref | 625 | 37 | 5.92% |
| therapy_background | 732 | 46 | 6.28% |

A broader mismatch diagnostic: **2153/4144 (51.95%)** non-forget positives share no non-stop content terms with their question. In the >50 cosine tail this is **1195/1948 (61.34%)**. Only **332/4144 (8.01%)** have a shared content term appearing in <=1% of fitting messages. The stoplist is a fixed heuristic, so these are diagnostics rather than a validated semantic-mismatch label.

Full length × type and role breakdowns are in positive_zero_by_group.csv (historical artifact: `positive_zero_by_group.csv`); per-question denominators and all-positive-zero flags are in question_zero_diagnostics.csv (historical artifact: `question_zero_diagnostics.csv`).

## Ties and random baselines

Current BM25, cosine and existing RRF use descending score followed by chronological source-message order. **7/1835** questions had a BM25 tie at the rank-ten cutoff. **17/1835** had all their positive BM25 scores equal to zero (including forget questions). The table includes chronological, cosine-secondary and exact expected random-tie BM25 results. Tie-neutral RRF averages reciprocal contributions within equal positive scores and gives zero lexical contribution to zero scores.

The exact random baseline averages each question's actual candidate and positive counts: per-positive inclusion is `10/N`; question Hit is `1 - C(N-P,10)/C(N,10)`. Using `10/230` for Hit ignores multiple positives and actual pool sizes. Expected random NDCG is also computed per question.

## Question-type results

| Type | Questions | Cosine Hit@10 | Default BM25 Hit@10 | Existing RRF Hit@10 | PRF RRF Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| anti_stereotypical_pref | 343 | 43.15% | 27.99% | 37.90% | 44.31% |
| ask_to_forget | 385 | 99.48% | 85.19% | 98.18% | 98.70% |
| health_and_medical_conditions | 257 | 35.02% | 29.18% | 36.58% | 37.74% |
| neutral_preferences | 373 | 27.35% | 22.52% | 26.27% | 27.88% |
| stereotypical_pref | 214 | 42.99% | 29.91% | 41.12% | 47.20% |
| therapy_background | 263 | 22.81% | 33.84% | 30.04% | 30.42% |

## Recommended plan changes

The measured priorities are:

- **Confirm `b=0` and chunk-max fusion first** on unused training personas. The `b=0` grid setting gains 117 net questions (207 rescued, 90 lost), while chunk max gains 116 (228 rescued, 112 lost). The 0.07-point gap between them is too small to choose a winner. `k1=1.2` vs `2.0` at `b=0` differs by just 0.21 point; this pilot gives little reason to spend more search on `k1`.
- **Question weights and stop filtering deserve a combined feature group.** Global IDF alone adds only 0.97 point in fusion; adding question weights and stop filtering reaches +6.62 points. Standalone score changes cannot replace incremental fusion/ranker evaluation.
- **Ties are currently a secondary issue at ten.** Alternate tie handling leaves default non-forget Hit@10 and NDCG@10 unchanged in this sample. Retain safe handling without spending the next experiment on tie policies.
- **Do not promote this PRF recipe.** It reaches 36.83% fused Hit versus 37.86% for its directly comparable global-IDF + stoplist control without expansion: a 1.03-point loss. Its fused tail rescue also falls from 1.90% to 0.67%. A different bounded expansion recipe or query-only model expansion remains a separate hypothesis.
- **Keep a dedicated tail experiment.** 61.34% of tail positive occurrences have no shared non-stop content terms. Even the improved fusions remain below random occurrence-level tail rescue. Chunk cosine and grounded extraction are useful competing hypotheses; no semantic-tail gain is demonstrated yet.

Protocol changes to preserve:

1. Optimize incremental retrieval over cosine, not raw BM25 score. Keep default BM25 and a cosine+metadata ranker control so gains from role/length are not attributed to lexical features.
2. Use training-only global IDF and question frequencies, with the evaluated personas excluded. Add the actual default to the parameter grid. Protect negations and constraints in stop/stem variants.
3. Audit zero ties and generic-only matches. Context/chunk/neighbor views remain attached to an original message, with independent labels and strict visibility/boundary checks.
4. Use both add-one and leave-one-group-out ranker ablations, then confirm locked winners on unused training personas. Do not apply the Phase 1 stopping rule to standalone BM25 or one RRF setting.
5. Keep PRF as a small controlled experiment, then compare compact query-only LLM expansion, chunk cosine and grounded fact extraction. The local 2B model is absent; the configured generator is 4B. SPLADE remains last.

The detailed replacement plan, exact feature definitions, stopping rules and primary references are in [IMPROVED_PLAN.md](#archived-report-04).

## Scope and verification

- Completed: fresh single-message cosine/default BM25 baseline; global-IDF, clipped question-weight, stoplist, parameter-grid, chunk, cross-message-window and bounded PRF scoring; RRF comparisons; exact random/tie diagnostics; non-forget metrics and zero-score breakdowns.
- Pending: stemming selection, neighbor-feature controls, learned ranker/group ablations, independent training-persona confirmation, LLM query expansion, extracted facts, SPLADE and chunk cosine. No full Phase 1 or adoption claim is made.
- Default BM25 scores agree with the existing implementation on all 1835 questions. Eight hand-calculated metric/tie/global-IDF checks pass. Input and protected source/config/split hashes are unchanged; see verification.json (historical artifact: `verification.json`).
- Score arrays are cached in `score_cache.npz` with column names, question offsets and separate labels. Row order follows `questions.jsonl` and each question's `memory_ids`, enabling later ablations without rescoring. Original global/query frequency tables, actual PRF terms, source hashes and embedding metadata are retained.

Reproduce from the project root:

```powershell
.venv/Scripts/python.exe -B bm25_test/run_pilot.py prepare
.venv/Scripts/python.exe -B bm25_test/run_pilot.py encode
.venv/Scripts/python.exe -B bm25_test/run_pilot.py evaluate
.venv/Scripts/python.exe -B bm25_test/make_report.py
.venv/Scripts/python.exe -B -m unittest discover -s bm25_test -p test_pilot.py
```


---

<a id="archived-report-04"></a>

## Archived source: IMPROVED_PLAN.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Revised BM25 experiment plan

The goal is better non-forget retrieval at ten complete messages. Increasing the numerical BM25 score is not a success criterion. Keep the current source scripts unchanged while developing and measuring the experiment here.

## Priority update from the completed screening pilot

Fresh results from 1,450 non-forget questions are in [FINDINGS.md](#archived-report-03). Cosine Hit@10 is 33.93%; existing RRF is 33.72%. RRF with `b=0, k1=2` reaches 42.00%, and chunk-max RRF reaches 41.93%. Freeze these as confirmation candidates; do not select between their 0.07-point difference. Both clear the predeclared practical screening threshold, but independent confirmation and combined ranker ablations remain pending.

Move `b=0` and chunk features to the front of the next bounded experiment. Global IDF alone adds 0.97 point in fusion, while global IDF + question weights + stop filtering adds 6.62 points. Tie handling did not affect default non-forget top-ten metrics in this sample. The tested PRF recipe loses 1.03 points against its matching non-expanded fusion control, so do not promote that recipe.

Keep hard-tail recovery as a separate objective: the leading fusions recover only 3.29–3.34% of positive occurrences with cosine rank >50, versus a 4.33% exact random inclusion baseline. These results justify confirming useful lexical fusion features while keeping chunk cosine and grounded extraction in the experimental queue. They do not establish success on vocabulary mismatch or semantic tail retrieval.

## What needs correcting first

- Current `scripts/compute_bm25.py` already uses single messages, positive IDF, unique query terms, `k1=1.5`, and `b=0.75`. Its DF and average length are question-local. Equal scores favor earlier source messages.
- Historical V2 diagnostics describe the deleted combined-message representation. Their 2,144,257 candidates and tail results cannot serve as the new message-level baseline. The current active training corpus has 184,159 distinct source messages for 18,532 questions across 800 personas; question pools range from 151 to 334 messages.
- Fresh V2 single-message BM25 scores, cosine scores and labels were absent at inspection. Regenerate all three from the same active IDs, preserving exact role/content labels and excluded questions.
- `diagnose_retrieval.py` reports Hit/Recall/full coverage, includes forget questions, and does not compute NDCG. Use non-forget Hit@10 as primary, binary NDCG@10 as secondary, plus separately reported forget metrics and question-type macro metrics.
- The proposed local 2B expansion model is absent. The configured local generator is Qwen3.5-4B Q4_K_M. Do not silently substitute it or download a replacement; choose and freeze the actual model when that experiment begins.

## Phase 0: Freeze a meaningful protocol

1. Use 1–2k non-forget training questions from a seeded persona sample. Keep every question's full visible message pool. Reserve separate training personas for confirmation; validation/test are untouched during development. Never split candidate rows randomly.
2. Declare a **strictly greater than 1.0 percentage point** non-forget Hit@10 improvement as the practical adoption threshold before inspecting results. Require no NDCG@10 regression and a positive lower 95% paired persona-bootstrap bound on independent training-persona confirmation. The pilot only screens candidates. A 1–2k pilot may be underpowered for a one-point effect; report intervals rather than interpreting a non-significant result as equivalence.
3. Fit global message statistics and question DF on fitting training personas. Exclude the evaluated fold's personas. At final training, fit once on all active training personas. No validation/test questions, answers, references or question types enter scoring features. Global statistics over held-out documents are a separate transductive protocol, not the default.
4. Evaluate cosine, BM25, existing RRF, zero-aware/tie-neutral RRF, and a small weighted normalized-score fusion selected on inner training folds. Freeze the same fusion setting for comparisons; changes in score scale must not masquerade as retrieval improvements. Never sum raw cosine and unnormalized BM25.
5. Record exact-zero positive occurrence rates and question-level rates where all positives are zero. Stratify by length, role, type, and length × type, with denominators. Define buckets without overlap: `<=64`, `65–300`, `>300`. Name the tokenizer. Keep lexical word counts and generator packing costs distinct.
6. Measure chronological tie ranks, cosine-secondary tie ranks, and expected metrics under uniformly random tied permutations. Zero-score candidates must not acquire lexical evidence solely through chronological rank in fusion. Also record ties at the top-ten boundary. Gold labels are used only to calculate diagnostics.
7. Separate two tail measures: (a) fraction of positive occurrences with cosine rank >50 recovered at ten; (b) fraction of cosine Hit@10 failures rescued at ten. Count displaced cosine successes too. Incremental Hit is `(rescued questions - lost questions) / eligible questions`. A positive >50 is not necessarily in a question that cosine failed.
8. Use exact per-question random baselines. With `N` candidates, `P` positives, `k'=min(k,N)`: expected Recall@k and per-positive rescue are `k'/N`; question Hit@k is `1 - C(N-P,k')/C(N,k')`; expected NDCG uses expected relevance `P/N` at each discounted rank and the question's ideal DCG. `k/230` is only a rough single-positive approximation.

## Phase 1: One cached lexical feature pass

Cache tokenized source messages, query tokens, term counts, postings, DF, lengths and visibility. Use a cache key containing corpus/source hashes, training persona IDs, tokenizer/stemmer/stoplist identities, `b`, `k1`, query weights and window recipe. Score each configuration once and reuse its arrays for all fusion and ranker ablations. Do not build a full Cartesian product of every option.

| Group | Concrete experiment | Controls and interpretation |
| --- | --- | --- |
| Global statistics | Local IDF vs training-global IDF; initially keep the same local average length | IDF only changes weights of existing matches. It cannot change lexical-zero coverage. Test global average length separately rather than confounding the comparison. |
| Query weights | Clipped inverse question DF with a positive floor; retain original terms | Fit question DF inside the fitting fold. Do not assume frequent domain terms or words such as “fun” are useless. Compare with no weighting. |
| Tokenization | Existing Unicode normalization; normalization + a small conversational stoplist; each with a pinned English stemmer | Preserve negators, constraints, numbers, names and medical terms. Define positions/chunks before stopword removal. Select two recipes only on inner training folds. Stemming can create matches, but can also merge different words. |
| Message parameters | Existing default plus `b in {0,.4,.75,1}` × `k1 in {1.2,2}` | The default `k1=1.5` is **not** in the supplied grid, so include it explicitly. Select a replacement only if its fused/ranked incremental gain over the default exceeds the gate on confirmation. Otherwise preserve the default; at most two predeclared `b` variants become features. |
| Chunks | For >64 lexical words, 48-word windows, stride 24, `b=0`; max and mean of best two | Keep a single whole chunk for shorter messages; one chunk's top-two mean equals that chunk. Include number of chunks and original length to expose the max-over-many advantage. Return one score per original message. Shortlisted chunk scores never replace whole-message labels. |
| Context windows | Previous+self, self+next, previous+self+next | Only visible source neighbors in the same history/session; do not cross omitted source turns. Score and label the center message. Keep direction-specific scores and center's own score. Concatenated text is an auxiliary view, not a new positive unit. |
| Neighbors | Directional scores/ranks at -2,-1,+1,+2 plus max/mean and availability masks | Boundary values are missing with masks, not a misleading rank of zero. Compare to role/length/position-matched shuffled-neighbor controls. Preserve sessions where known; V2 has no guaranteed session markers. |
| Match evidence | Unique matched query terms, unique-term coverage, matched global-IDF sum, IDF-weighted coverage, rare-match flag | Define rarity before results, e.g. training message DF/N <=1%. Compute original-query and expanded-query coverage separately. Distinguish a rare shared word from many generic words. |
| Metadata | Role, lexical length, named subword length; history position as a separately controlled feature | Length and role belong in both the cosine+metadata control and the lexical model so their benefit is not credited to BM25. |

Keep the scoring unit explicit when estimating global DF for chunks/windows. The pilot uses **parent-message DF** for these views to isolate added context, with `b=0`. A canonical chunk/window corpus has different DF and is a separate setting; do not call the two methods identical. Do not leak the identity of a positive neighbor into a feature.

## Phase 2: Test mismatch hypotheses cheaply

1. **PRF first:** retain the original query and add a separately weighted expansion from the top three cosine messages. Start with 8–12 novel terms, exclude generic/numeric-only terms, cap total expansion mass, and use training-global IDF × saturated TF × retrieval weight. Compare original, expansion-only, and combined BM25. Save expansion terms and latency. Top-three feedback is least reliable when those cosine hits are unrelated; stratify by cosine confidence and measure false rescues/lost successes. Do not use gold answers or reference snippets.
2. **Query-only LLM expansion next:** freeze the actual available model, generation settings, prompt and JSON schema. Ask for synonyms, paraphrases, related object names and abbreviations; no invented personal facts. Compare a compact 4–8-term expansion with 10–15 terms at the same expansion mass, cache by model/prompt/question hash, and keep it a second query channel. This is a small pilot, not an assumption that vocabulary mismatch is solved.
3. **Facts index after extraction quality is measured:** extract from messages only, preserve exact support quotes, roles and speaker/subject attribution, then attach fact-query scores to their original parent message. Missing/failed extractions get explicit masks. Evaluate extraction coverage, grounding and total latency before scaling. Exact-reference labels do not certify every unannotated message as irrelevant.
4. **Chunk cosine as a competing investment:** compare on the same locked questions, token accounting and parent-message candidates. This may address dilution in long assistant replies more directly than a lexical parameter grid.
5. **SPLADE last:** only if cheaper experiments fail and hardware/latency allow it. Require a measured incremental benefit over the best frozen combination; dense-model or sparse-model reputation is not evidence on this corpus.

## Phase 3: Grouped ranker ablations and confirmation

Use `LGBMRanker(objective='lambdarank', label_gain=[0,1])`, with contiguous rows grouped by question and persona-disjoint fitting/evaluation. Fix a small tree budget/hyperparameter recipe before group ablation; early stopping and feature selection use inner training personas. Report the fixed-iteration out-of-fold comparison first, then a locked configuration on unused training personas. Do not choose features or stopping iterations on validation. Validation becomes a single frozen audit, and test remains final.

Compare raw cosine; cosine ranker + metadata; that same ranker + default BM25; + global/query-weight/tokenization group; + chunks; + context windows; + neighbors; + match evidence; full Phase 1; and full Phase 1 with each group removed. Add PRF, LLM and facts one group at a time later. Keep candidate pools and packing identical. Gold question type is an evaluation stratum only.

For each row report non-forget Hit@10, NDCG@10, delta vs raw cosine, delta vs the matching ranker control, paired confidence interval, >50 positive tail rescue, cosine-failure question rescue, lost cosine successes, throughput, peak RAM/VRAM and cache size. Include exact random baselines and type-macro results. Retain groups only when their removal causes a practical, confirmed loss; several individually weak groups may work together, so perform both add-one and leave-one-out comparisons.

## Stopping rule and order

Finish Phase 0, then use the cheap Phase 1 pass to screen candidates. A weak standalone BM25 score or default RRF row does **not** prove all lexical features are useless to a ranker. Apply the Phase 1 gate after testing the combined feature group against both cosine and cosine+metadata ranker controls. If its confirmed gain is <=1 point, stop lexical tuning and spend the next bounded pilot on PRF, chunk cosine or grounded extraction. Avoid expanding the grid because the pilot winner narrowly missed the gate.

The plan-review pilot in this folder deliberately does not train the ranker, run stemming, generate LLM expansions, extract facts or run SPLADE. Its results are screening evidence, not completed Phase 1/Phase 3 ablations or an adoption decision.

## Primary references

- [Stanford IR book: inverse document frequency](https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html) explains collection DF as term discrimination.
- [Stanford IR book: pseudo relevance feedback](https://nlp.stanford.edu/IR-book/html/htmledition/pseudo-relevance-feedback-1.html) describes the risk of query drift.
- [scikit-learn NDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html) documents tie averaging; our operational table retains a declared deterministic ranking.
- [LightGBM ranker API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html) documents question groups and ranking evaluation.


---

<a id="archived-report-05"></a>

## Archived source: next_stage/FINDINGS.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Further Hit@10 experiments

**The selected 54-feature ranker achieves 70.19% Hit@10 and 0.4105 NDCG@10.** On the same confirmation questions, the 28-feature control with 1,450 fitting questions achieves 61.16%: **+9.04 points**, paired 95% interval **[+7.13, +10.96]**. Against the 28-feature control with the same 5,107 fitting questions and larger trees, the gain is **+6.17 points**, interval **[+4.29, +8.08]**. Cosine-tail rescue rises from 7.05% (162/2,298) to 22.02% (506/2,298) against that matched control.

The retained model has 300 trees, maximum depth 5 and up to 31 leaves per tree. Its inputs are the original 28 features (including role and question-frequency weights), 20 directional neighbor features and six dense chunk-cosine features. More data alone adds 0.99 points and does not clear the frozen gate; larger trees add 1.87 points on the full data and do clear it. Neighbor and dense evidence provide the larger additional improvement.

The selected experimental method is **final_candidate**, retaining role and question-frequency weighting. Confirmation uses 1,815 fresh non-forget questions from 100 personas; development uses 5,107 previously inspected questions from 280 personas. All model/group/capacity and semantic-channel choices were frozen before confirmation metrics.

| Method | Training questions | Hit@10 | NDCG@10 | Cosine-tail rescue |
|---|---:|---:|---:|---:|
| cosine | — | 34.82% | 0.1261 | 0.00% |
| base1450 | 1450 | 61.16% | 0.2839 | 5.92% |
| base3000 | 3003 | 61.38% | 0.2923 | 5.87% |
| base_full | 5107 | 62.15% | 0.2957 | 6.35% |
| base_full_capacity | 5107 | 64.02% | 0.3050 | 7.05% |
| selected | 5107 | 70.19% | 0.4105 | 22.02% |
| final_candidate | 5107 | 70.19% | 0.4105 | 22.02% |
| random_exact | — | 11.77% | 0.0272 | 4.36% |

| Comparison | Hit@10 change (pp) | Paired persona 95% interval | NDCG change | Gate |
|---|---:|---|---:|---|
| base3000 vs base1450 | +0.22 | [-0.73, +1.16] | +0.0085 | FAIL |
| base_full vs base1450 | +0.99 | [-0.27, +2.21] | +0.0119 | FAIL |
| base_full_capacity vs base_full | +1.87 | [+0.88, +2.89] | +0.0093 | PASS |
| base_full_capacity vs base1450 | +2.87 | [+1.40, +4.29] | +0.0211 | PASS |
| selected vs base_full_capacity | +6.17 | [+4.29, +8.08] | +0.1055 | PASS |
| final_candidate vs base_full_capacity | +6.17 | [+4.29, +8.08] | +0.1055 | PASS |
| final_candidate vs base_full | +8.04 | [+6.20, +9.88] | +0.1148 | PASS |
| final_candidate vs base1450 | +9.04 | [+7.13, +10.96] | +0.1267 | PASS |

## Development decisions

Selected groups: **neighbor, dense**. Selected capacity: **larger**. Selected schema has 54 features. The learning curve uses persona subsets; OOF fits are smaller than full-data fits and their actual mean counts are recorded. Development group screening and leave-one-out pruning use the same >1-point gain, positive lower bootstrap interval, and no NDCG regression gate. The primary fresh comparison uses a 28-feature control with the same full training data and preselected tree capacity. Separate original-capacity full-data and 1,450-question controls distinguish data scale, tree capacity and new features.

| Development model | Features | Mean OOF fitting questions | Hit@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| base28 | 28 | 3830 | 60.66% | 0.2811 |
| curve_small | 28 | 1056 | 59.39% | 0.2685 |
| curve_medium | 28 | 2245 | 60.33% | 0.2763 |
| base28_larger | 28 | 3830 | 62.17% | 0.2873 |
| add_neighbor | 48 | 3830 | 64.99% | 0.3635 |
| add_match | 33 | 3830 | 61.74% | 0.2862 |
| add_shuffled | 48 | 3830 | 63.81% | 0.3072 |
| add_dense | 34 | 3830 | 65.01% | 0.3186 |
| all_groups | 59 | 3830 | 68.53% | 0.3857 |
| without_neighbor | 39 | 3830 | 65.46% | 0.3179 |
| without_match | 54 | 3830 | 68.93% | 0.3879 |
| without_dense | 48 | 3830 | 64.99% | 0.3635 |
| selected | 54 | 3830 | 68.93% | 0.3879 |

Real neighbors vs role/length-matched shuffled neighbors: +1.17 Hit@10 points, paired 95% interval [+0.17, +2.19], NDCG change +0.0563. The shuffled control also improves on the reference, so the entire neighbor-vs-reference gain should not be attributed specifically to adjacency.

Dense auxiliary spans use Unicode NFKC normalization and casefolding while retaining punctuation within each window. Short messages reuse their original full-message cosine. Returned messages always retain the original text. The dense group therefore tests this concrete auxiliary representation, not every possible chunking/normalization recipe.

## Semantic pilots

- **cross_encoder**: 200 development questions, entire visible pools. Reference Hit@10 71.00%; fused 55.50%; change -15.50 points, interval [-22.78, -9.50]. Pilot scale decision: STOP. Projected confirmation runtime 0.98 hours. Full pilot details (historical artifact: `cross_encoder_pilot.json`).
- **query_expansion**: 200 development questions, entire visible pools. Reference Hit@10 71.00%; fused 62.00%; change -9.00 points, interval [-15.96, -3.12]. Pilot scale decision: STOP. Projected confirmation runtime 0.76 hours. Full pilot details (historical artifact: `query_expansion_pilot.json`).

Facts pilot: 100 source messages, 0 schema failures. No facts were produced, so quote-support precision is undefined. Grounded message coverage 0.0%. Projected full extraction: 12.3 hours. The extractor succeeded on a separate synthetic message with two explicit personal facts and returned no user facts for generic assistant advice. These sanity inputs were not used in training or retrieval evaluation. The empty corpus outputs indicate inadequate coverage of this conservative extraction recipe, not absence of useful preference evidence. Exact quote matches alone would not establish entailment or correct subject attribution. Bulk extraction did not pass coverage/cost gates. Details (historical artifact: `facts_pilot.json`), sanity check (historical artifact: `fact_sanity.json`), sampling correction to reach 100 messages (historical artifact: `fact_sampling_audit.json`).

Query expansion improved its matched unexpanded lexical-channel fusion, but both lexical-channel fusions underperformed the stronger selected ranker. The negative pilot result applies to this fixed RRF integration; a trained combination was not tested at scale. Similarly, the MiniLM result does not rule out a different or personalization-trained cross-encoder.

## Interpretation and artifacts

Do not compare the absolute Hit percentages to earlier studies on different questions. Original exact-reference labels define these metrics; unannotated messages can still contain useful information. The primary task remains retrieving ten complete original messages. Chunks, neighbor text, reranker windows and expansions are auxiliary views. No gold references or answers entered scoring inputs. All background DF excludes development and confirmation personas. Validation/test were not used.

- Failure rank coverage (historical artifact: `reference_coverage.csv`), failure buckets (historical artifact: `failure_buckets.csv`), 90 sampled failure cases (historical artifact: `failure_audit_90.jsonl`).
- Development group ablations (historical artifact: `development_group_comparisons.csv`), development metrics (historical artifact: `development_metrics.csv`), training/model provenance (historical artifact: `training_report.json`).
- Confirmation metrics (historical artifact: `metrics.csv`), paired comparisons (historical artifact: `comparisons.csv`), question-type breakdown (historical artifact: `by_question_type.csv`), actual selected top ten (historical artifact: `selected_top10.jsonl`).
- Rank-ten tie diagnostics (historical artifact: `tie_diagnostics.csv`), inference reconstruction check (historical artifact: `inference_smoke.json`). Final score ties use chronological source order.
- Selected model/schema (historical artifact: `selection.json`), frozen candidate before confirmation (historical artifact: `frozen_candidate.json`), protocol (historical artifact: `protocol.json`), verification (historical artifact: `verification.json`).
- Embedding runtime and VRAM (historical artifact: `encode_report.json`), reranker assets (historical artifact: `reranker_assets.json`), tests (historical artifact: `tests.log`).

All new artifacts are under bm25_test/next_stage; historical studies, raw sources, split IDs and main scripts/configs are preserved.

All 31 tests pass. Inference reconstructs all 223 message scores in the smoke history exactly; injected reference/answer annotations cannot change model features. Dense similarities and the reconstruction check use `OPENBLAS_NUM_THREADS=4`. The default BLAS thread configuration produced one z-score difference of 0.00000423 in the initial strict feature comparison, so matching this setting is part of numerical reproduction. No tolerance, model or confirmation result was changed to obtain parity. Inference check (historical artifact: `inference_smoke.json`).

The selected model has no ties at the rank-ten cutoff in these 1,815 questions, and uniform tie averaging leaves its metrics unchanged. Learning curves (historical artifact: `learning_curves.png`) show the original-capacity 28-feature controls; they do not include the final feature additions. Encoding timing covers the successful resumed invocation; earlier interrupted performance-optimization attempts are excluded from that timing field.


---

<a id="archived-report-06"></a>

## Archived source: next_stage/models/ms-marco-MiniLM-L6-v2/README.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

---
license: apache-2.0
datasets:
- sentence-transformers/msmarco
language:
- en
base_model:
- cross-encoder/ms-marco-MiniLM-L12-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
tags:
- transformers
---
# Cross-Encoder for MS Marco

This model was trained on the [MS Marco Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) task.

The model can be used for Information Retrieval: Given a query, encode the query with all possible passages (e.g. retrieved with ElasticSearch). Then sort the passages in a decreasing order. See [SBERT.net Retrieve & Re-rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) for more details. The training code is available here: [SBERT.net Training MS Marco](https://github.com/UKPLab/sentence-transformers/tree/master/examples/cross_encoder/training/ms_marco)


## Usage with SentenceTransformers

The usage is easy when you have [SentenceTransformers](https://www.sbert.net/) installed. Then you can use the pre-trained models like this:
```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
scores = model.predict([
    ("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
    ("How many people live in Berlin?", "Berlin is well known for its museums."),
])
print(scores)
# [ 8.607138 -4.320078]
```


## Usage with Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')

features = tokenizer(['How many people live in Berlin?', 'How many people live in Berlin?'], ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.'],  padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    print(scores)
```


## Performance
In the following table, we provide various pre-trained Cross-Encoders together with their performance on the [TREC Deep Learning 2019](https://microsoft.github.io/TREC-2019-Deep-Learning/) and the [MS Marco Passage Reranking](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset. 


| Model-Name        | NDCG@10 (TREC DL 19) | MRR@10 (MS Marco Dev)  | Docs / Sec |
| ------------- |:-------------| -----| --- | 
| **Version 2 models** | | | 
| cross-encoder/ms-marco-TinyBERT-L2-v2 | 69.84 | 32.56 | 9000
| cross-encoder/ms-marco-MiniLM-L2-v2 | 71.01 | 34.85 | 4100
| cross-encoder/ms-marco-MiniLM-L4-v2 | 73.04 | 37.70 | 2500
| cross-encoder/ms-marco-MiniLM-L6-v2 | 74.30 | 39.01 | 1800
| cross-encoder/ms-marco-MiniLM-L12-v2 | 74.31 | 39.02 | 960
| **Version 1 models** | | | 
| cross-encoder/ms-marco-TinyBERT-L2  | 67.43 | 30.15  | 9000
| cross-encoder/ms-marco-TinyBERT-L4  | 68.09 | 34.50  | 2900
| cross-encoder/ms-marco-TinyBERT-L6 |  69.57 | 36.13  | 680
| cross-encoder/ms-marco-electra-base | 71.99 | 36.41 | 340
| **Other models** | | | 
| nboost/pt-tinybert-msmarco | 63.63 | 28.80 | 2900 
| nboost/pt-bert-base-uncased-msmarco | 70.94 | 34.75 | 340 
| nboost/pt-bert-large-msmarco | 73.36 | 36.48 | 100 
| Capreolus/electra-base-msmarco | 71.23 | 36.89 | 340 
| amberoad/bert-multilingual-passage-reranking-msmarco | 68.40 | 35.54 | 330 
| sebastian-hofstaetter/distilbert-cat-margin_mse-T2-msmarco | 72.82 | 37.88 | 720
 
 Note: Runtime was computed on a V100 GPU.

---

<a id="archived-report-07"></a>

## Archived source: position_features/FINDINGS.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Conversation-position feature experiment

**Both new features improve the measured results.** Adding both raises historical-cohort Hit@10 from **70.19% to 71.35%** (+1.16 points, paired persona-bootstrap95 [+0.16, +2.19]); relative position alone reaches **71.57%**. Development also improves, but none exceeds the preregistered one-point development threshold. Development-selected method: **baseline54**. The original selected model is preserved under that rule; this does not mean the new features have no benefit.

The added features are source-turn distance from the latest question-visible message, and normalized source position from earliest (0) to latest (1). These measure order, not elapsed calendar time. No invisible future messages, answer fields or references enter either feature.

## Results

### Development: 5,107 questions / 280 personas, four persona folds

| Method | Hit@10 | NDCG@10 | Tail rescue | Hit gain (pp) | Paired 95% CI (pp) | Gate |
|---|---:|---:|---:|---:|---|---|
| baseline54 | 68.93% | 0.3879 | 20.83% | - | - | - |
| distance55 | 69.92% | 0.4022 | 21.72% | +0.999 | [+0.37, +1.61] | False |
| relative55 | 69.77% | 0.4014 | 21.83% | +0.84 | [+0.22, +1.45] | False |
| both56 | 69.75% | 0.4028 | 22.06% | +0.82 | [+0.16, +1.47] | False |
| random_exact | 11.95% | 0.0274 | 4.36% | - | - | - |

### Historical secondary: 1,815 questions / 100 disjoint personas

| Method | Hit@10 | NDCG@10 | Tail rescue | Hit gain (pp) | Paired 95% CI (pp) | Gate |
|---|---:|---:|---:|---:|---|---|
| baseline54 | 70.19% | 0.4105 | 22.02% | - | - | - |
| distance55 | 71.18% | 0.4222 | 22.72% | +0.99 | [+0.06, +1.92] | False |
| relative55 | 71.57% | 0.4226 | 23.37% | +1.38 | [+0.34, +2.42] | True |
| both56 | 71.35% | 0.4228 | 23.80% | +1.16 | [+0.16, +2.19] | True |
| random_exact | 11.77% | 0.0272 | 4.36% | - | - | - |

The historical secondary cohort has been inspected in earlier studies. These locked comparisons are exploratory replication, not new independent confirmation. Candidate choice uses development results only. Final validation/test were not evaluated.

Distance alone adds exactly **51 net development hits / 5,107 questions = 0.998629 points**. Its failure flag is solely the strict >1-point cutoff, not a negative effect or an interval spanning zero. Relative position adds 43 net development hits; both features add 42. On the historical secondary cohort, distance rescues 52 questions and loses 34, relative position rescues 58 and loses 33, and both rescue 55 and lose 34. These are three prespecified exploratory comparisons; intervals are not adjusted for multiple comparisons.

## Interpretation and limits

The gate requires >1 absolute Hit@10 point, a positive lower paired persona-bootstrap95 bound, and no NDCG point regression. All 300-tree ranker settings, labels, candidate pools, source statistics and persona folds match the original recipe. The retrained baseline reproduces original development and historical secondary predictions bit for bit.

Within one history the two features are related: relative position = 1 - distance / history source-index span (except the singleton convention). Across different history lengths raw distance supplies an absolute scale. Ablations establish whether either representation contributes empirically; they are not two independent notions of time.

Any gain can reflect the way evidence is positioned in this benchmark. It does not establish that newer messages are universally more relevant, or that the method transfers to real timestamped conversations. Reference-position distributions are saved for inspection.

In development, reference rates by conversation quarter are 1.421%, 1.456%, 1.683%, and 0.395%. The historical cohort shows the same drop in the latest quarter: 1.391%, 1.339%, 1.701%, and 0.459%. Thus the useful positional signal includes a lower reference frequency near the end; the experiment does not impose a preference for recent messages.

All six unit tests pass. The saved 54-, 55-, and 56-feature models reproduce batch predictions exactly on a 223-message history; adding answer/reference/future metadata cannot change position features. See inference_smoke.json (historical artifact: `inference_smoke.json`). One evaluation configuration had already changed since the historical study; its current hash was recorded at this experiment's start and remained unchanged. Cached ranker features, source data, model parameters and original predictions reproduce exactly.

All new code, model files, predictions, timing, importance and subgroup reports are in this folder. Unit tests cover source gaps, singleton histories, visibility/cutoffs, translation invariance, invalid ordering and independence from answer/reference metadata. Frozen-input and original-model hashes are checked in verification.json.


---

<a id="archived-report-08"></a>

## Archived source: question_weight_role/FINDINGS.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Question-frequency weights and role removal

**The selected candidate removes `role_assistant`; question-frequency features are not retained.** The fixed weighting group changed Hit@10 by +1.07 percentage points over the role-free control (paired persona-bootstrap 95% interval [-0.11, +2.24]).

This comparison uses 1,775 fresh non-forget training questions from 100 personas, excluding all 180 previously evaluated personas. Every ranker was fitted on the same original 1,450 questions from 80 personas. All four models were frozen before fresh metrics were computed. No validation/test data or hyperparameter tuning was used.

| Method | Features | Hit@10 | NDCG@10 | Tail rescue |
|---|---:|---:|---:|---:|
| Cosine alone | — | 33.69% | 0.1187 | 0.00% |
| Existing cosine + BM25 fusion | — | 40.51% | 0.1593 | 3.51% |
| Same fusion with question weights | — | 43.27% | 0.1763 | 3.93% |
| Current ranker control | 18 | 57.80% | 0.2637 | 5.26% |
| Remove role | 17 | 56.96% | 0.2563 | 5.09% |
| Add question-weight features | 28 | 59.27% | 0.2713 | 6.67% |
| Remove role + add question-weight features | 27 | 58.03% | 0.2631 | 6.07% |
| Exact random baseline | — | 11.73% | 0.0272 | 4.36% |

## Attribution and decision

Removing role alone changes Hit@10 by -0.85 points (95% interval [-1.88, +0.18]) and NDCG@10 by -0.0074; 41 questions rescued and 56 lost. Removal is an explicit user constraint, not a claim that a small or uncertain difference proves role has no effect.

The primary weighting comparison rescues 70 questions and loses 51. Its NDCG@10 change is +0.0068. The frozen gate requires more than +1.00 Hit@10 point, a positive lower paired 95% interval, and no NDCG regression. The combined candidate must also avoid point-estimate regression against the 18-feature control. Primary gate: **FAIL**. Selected model: **`no_role_ranker`**, with **17 features**. Role is excluded from both role-free fitted models, not merely set to zero at inference.

| Comparison | Hit@10 change (pp) | Paired 95% interval | NDCG change | Gate |
|---|---:|---|---:|---|
| no_role_qweight_ranker vs no_role_ranker | +1.07 | [-0.11, +2.24] | +0.0068 | FAIL |
| no_role_ranker vs current_ranker | -0.85 | [-1.88, +0.18] | -0.0074 | FAIL |
| qweight_ranker vs current_ranker | +1.46 | [+0.27, +2.67] | +0.0075 | PASS |
| no_role_qweight_ranker vs qweight_ranker | -1.24 | [-2.34, -0.17] | -0.0082 | FAIL |
| no_role_qweight_ranker vs current_ranker | +0.23 | [-1.00, +1.46] | -0.0007 | FAIL |
| qweight_fusion vs winning_fusion | +2.76 | [+1.49, +4.03] | +0.0169 | PASS |

The highest-scoring model is the 28-feature ranker that retains role and adds weighted features: 59.27% Hit@10. Its diagnostic comparison against the 18-feature control clears the numerical gate (+1.46 points, interval [+0.27, +2.67]). Removing role from that weighted model costs 1.24 points (interval [-2.34, -0.17]) and 0.0082 NDCG. These fixed diagnostic comparisons suggest that the role flag remains useful here; the earlier role breakdown was not an ablation demonstrating that it could be removed without loss. They do not override the user's explicit removal instruction or the frozen primary selection rule.

Question weighting also improves the simple fusion from 40.51% to 43.27% (+2.76 points, interval [+1.49, +4.03]). Thus the result is not that frequency weighting fails in general: it helps the simpler retrieval method and the role-retaining ranker, while its incremental Hit@10 benefit in the role-free ranker remains uncertain. The role-free weighted model is saved for further confirmation, rather than discarded or silently promoted.

## Exact weighting and controls

`weight(t) = clip(log((Nq + 1) / (question_df(t) + 1)) / log(Nq + 1), 0.1, 1.0)`.

Question DF uses 12,066 original active training questions from 520 background personas. Message DF uses 119,909 original messages from those same personas. Each word counts once per question/message. All 280 training and evaluated personas are excluded from background statistics. The formula is the original pilot recipe; no new weight, stoplist, b or k1 was tuned.

The added group contains ten numbers: weighted whole-message BM25 score/rank fraction/z-score/gap and its cosine RRF fusion; weighted chunk maximum/rank fraction/z-score/gap and top-two mean. Original unweighted features are retained. Whole-message scoring stays local-IDF, b=0, k1=2, without stop removal. Chunk scoring stays global parent-message IDF, b=0, k1=1.5, conversational stoplist, 48-token windows and stride 24 for messages over 64 lexical tokens.

The 18-feature control is retrained because global message DF now excludes the fresh personas. Its percentage must be compared with the other methods on this same sample, not with the earlier 60.63% on different questions. The first 12 original training features have exact cache parity; only global chunk statistics and the new weighted views change. Role metadata is still available in original records but is absent from the selected model input schema. Message text and length can still correlate with role.

Tail rescue counts annotated positive message occurrences whose cosine rank exceeds 50. The exact random occurrence baseline is 10/N per question (about 10/230); random Hit@10 also accounts for multiple references. Zero-weighting is not used: common terms retain at least 0.1 of their contribution, so weighting does not create new lexical matches.

## Artifacts

- Frozen protocol (historical artifact: `protocol.json`), selection and exact selected feature schema (historical artifact: `selection.json`), fitted models (historical artifact: `models/`).
- Fresh metrics (historical artifact: `metrics.csv`), paired comparisons (historical artifact: `comparisons.csv`), question-type breakdown (historical artifact: `by_question_type.csv`), training OOF diagnostics (historical artifact: `training_oof_metrics.csv`).
- All question-term frequencies and weights (historical artifact: `question_term_weights.csv`), frozen statistics (historical artifact: `statistics.json`), source/split/model verification (historical artifact: `verification.json`).
- Training provenance (historical artifact: `training_report.json`), feature construction and experiment runner (historical artifact: `run_experiment.py`).
- `run_experiment.rank_messages(question, messages, cosine_scores, content_token_lengths)` applies the selected saved role-free model; actual inference parity check (historical artifact: `inference_smoke.json`).
- Actual selected top-ten rankings for every fresh question (historical artifact: `no_role_ranker_top10.jsonl`).
- All 24 passing checks (historical artifact: `all_tests.log`). The saved role-free model's actual inference reproduces every score and rank for a 254-message history and remains identical when all role labels are flipped.

All original pilot/confirmation artifacts, main scripts/configs, raw data and split memberships are preserved.


---

<a id="archived-report-09"></a>

## Archived source: ranker54_failure_audit/FINDINGS.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# What the 54-feature ranker's misses mean

The frozen model misses all annotated reference messages in its top ten on **541/1,815 questions (29.81%)**. This is a retrieval-reference miss rate, not a measured answer-error rate. All references are present in the candidate pools, and there are no tied rank-ten cutoffs.

| Best reference rank | Misses | Share of misses |
|---|---:|---:|
| 11–20 | 194 | 35.9% |
| 21–50 | 217 | 40.1% |
| >50 | 130 | 24.0% |

The top 20 contain a reference for 80.88% of questions; the top 50 for 92.84%. These are larger-context coverage measurements, not changes to Hit@10.

## Measured patterns

- **145 misses have a reference in at least one primary channel's top ten.** The final combination demotes those references, but this does not establish an erroneous consensus rule. The same ranker retrieves references in 236 successful questions where no primary channel achieves a top-ten reference.
- **396 misses have no primary-channel top-ten reference.** Of these, 184 have no shared non-stoplist query word with any reference and 151 have at most one (specifically exactly one maximum overlap). Weak direct evidence and indirect personalization are plausible explanations, not an exhaustive causal classification.
- **194/541 misses (35.9%) have no reference content-word overlap**, compared with 244/1,274 hits (19.2%). The mismatch is associated with failure, but does not make retrieval impossible: dense/neighbor signals already rescue many no-overlap questions.
- **69 misses retrieve an immediate neighbor of a reference; 92 retrieve a message within two source positions.** The neighboring text must be inspected before calling this a partner-turn substitution or a label-boundary error.
- A reference longer than 64 words occurs in **96.9% of both misses and hits**. That statistic alone does not explain failure. References over 300 words occur in 26.6% of misses versus 18.3% of hits, an association that still does not prove dilution caused a given miss.
- Failure rates are 35.3% for therapy-background questions, 34.3% for neutral preferences, 33.2% for health/medical-condition questions, 24.7% for stereotypical preferences, and 22.5% for anti-stereotypical preferences. These are dataset categories, not inferred traits of actual people.

## Qualitative examples

Fifteen examples were selected by a fixed random seed, one per rank bucket and question type. Excerpts were inspected; they were not exhaustively adjudicated into causal labels.

- `v2:train:12953`, criticism and confidence: the reference describes early doubts about a person's finance career inside a professional biography. Generic advice explicitly discussing criticism ranks first. This illustrates matching the question's topic while missing the specific personal history. Best reference rank: 15.
- `v2:test:04704`, bright sunlight: a reference email mentions a headache, prescription, and resting in a dim room within a longer account of a children's playdate. The useful detail is indirect and embedded in unrelated context. Best reference rank: 21. This is an example of evidence linkage, not a diagnosis inferred from the text. The `test` string is an original CSV identifier; this question belongs to the active training split.
- `v2:train:06714`, restless nights and daytime focus: the reference concerns staying awake until 2 a.m. debugging, but the retrieved top messages also discuss late nights, workload, and lost sleep. These are plausible unannotated alternatives; answering usefulness has not been established. Best reference rank: 77.
- `v2:train:04869`, comparing one's work with others: references include a fictional writer's insecurity and follow-up edits about moonlight metaphors. This calls for careful subject attribution and a benchmark-label audit, rather than treating a character's experience as an established personal fact.

Thus the 29.81% should not be attributed to one cause. It mixes ranking tradeoffs, weak or indirect matching signals, competing conversations, and potentially incomplete or ambiguous reference annotations. Exact causal percentages would require independent evidence/answer adjudication.

## Proposed fixes were checked

The supplied analysis's major counts were confirmed, but its gain estimates overlooked displacement. Fixed top-eight-plus-two-channel quotas reduce Hit@10 to 66.23–66.56%. One explicitly defined, ungated adjacent-turn swap reduces it to 68.76%. The proposed exact-zero maximum-BM25 gate activates on no questions. These are post-hoc diagnostics, not new model-selection results.

[Detailed proposal review and revised plan](#archived-report-10), exact proposal checks (historical artifact: `proposal_checks.json`), full numerical summary (historical artifact: `summary.json`), per-question diagnostics (historical artifact: `per_question.jsonl`), sampled reference/retrieved text (historical artifact: `sampled_examples.jsonl`).

No training, selected-model changes, gold-label changes, or evaluation of the new reserved holdout occurred. Historical studies and raw data were read only.


---

<a id="archived-report-10"></a>

## Archived source: ranker54_failure_audit/PROPOSAL_REVIEW.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Review and revised plan for the 54-feature ranker

The supplied analysis correctly reproduces most diagnostic counts. Its causal explanations are hypotheses, and its projected 78–81% Hit@10 is not supported by measured net gains. Fixed replays of two suggested policies reduce Hit@10. Preserve the current model while testing more selective, learned changes on development data.

This review uses the frozen model's already-evaluated 1,815-question training-persona confirmation set for **post-hoc diagnosis only**. No model was trained, selected, deployed, or changed. The 1,785 questions reserved by `relevance_methods` remain unevaluated. All new artifacts are in this directory.

## What the counts establish

- 1,274 hits and 541 misses: 70.19% Hit@10. Every reference is in its question's candidate pool; there are zero rank-ten cutoff ties.
- Of 541 misses, 194 have their best reference at ranks 11–20, 217 at 21–50, and 130 below 50. These are rank-location categories, not mutually exclusive causal explanations.
- The 145 misses with a primary channel hit split into 90 lexical-only, 38 dense-only, and 17 with a hit in each family. **Only 12 of those 17 have the same positive message in both families' top tens.** A question may have several positive messages.
- Of the 396 misses without a primary-channel top-ten reference, 184 have zero content-word overlap and 151 have only one, using the maximum overlap across that question's references.
- The ranker also **successfully retrieves a reference for 236 questions where no primary channel has a top-ten reference**. Therefore, disagreement with individual channels is not automatically an error in how the model combines evidence.
- Adjacency counts of 69 at distance one and 92 within two source turns are correct. Adjacency alone does not prove the retrieved message belongs to the same conversational episode or supports the answer.

See failure summary (historical artifact: `summary.json`), per-question measurements (historical artifact: `per_question.jsonl`), and proposal checks (historical artifact: `proposal_checks.json`).

## Claims that need correction

| Supplied claim | Correction |
|---|---|
| The trees require lexical/dense consensus. | There is no explicit consensus requirement. The raw features already allow either channel to influence a tree. Confirm any particular decision through tree paths/contributions and matched cases; the observed miss counts alone do not establish this cause. |
| All-channel misses are caused by semantic dilution. | Dilution is plausible in some examples, but indirect relevance, competing evidence, subject attribution, and reference-label ambiguity also occur. No-channel top-ten status does not establish that the available features cannot recover a reference. |
| Raw score margins measure confidence. | LightGBM ranking scores are not calibrated probabilities. A margin of 0.25 is descriptive; any confidence gate needs development calibration. |
| The 92 adjacent-message misses prove partner-turn substitution. | Some adjacent messages can be unrelated or belong to another exchange. References often already include multiple user and assistant turns. Inspect paired text before assigning this cause. |
| 27 adjacent near misses imply +1.5 points for the top-three policy. | The count uses adjacency to any top-ten message. Only five misses have an eligible rank11–20 reference adjacent to an opposite-role top-three message under the stated stricter rule. Even those are oracle opportunities before displacement and candidate choice. |
| The existing dense chunks are 64–128 words or 128 tokens. | The selected model already uses **48-word spans, stride 24**, for messages over 64 lexical words; shorter messages reuse their whole-message vector. This is not a new jump from only whole-message embeddings. |
| All gold evidence is first-person self-disclosure. | Actual reference spans include generic prompts, assistant rewrites, third-person narratives, technical questions, and implied preferences. First-person filtering would remove some positives and does not establish whose experience a quotation describes. |
| High chunk peakiness means relevant evidence; low peakiness means background chatter. | Either pattern can be relevant or irrelevant. Maxima also depend on the number of chunks. Treat this as a feature hypothesis, control for span count, and avoid hard rules. |
| Max-minus-whole is a new chunk feature. | `dense_chunk_minus_whole` already exists. Max-minus-mean over all chunks would be new; max versus top-two mean is largely derivable from existing inputs. |
| Late interaction over 30 candidates costs 5–10 ms on this machine. | No measured local timing supports this, and token encoding/cache construction costs remain. A trained late-interaction model is not interchangeable with arbitrary token vectors from a pooled embedding model. It also conflicts with the user's decision to exclude ColBERT/SPLADE for now. |
| Rescues from different proposals can be added to forecast 78–81%. | The targeted questions overlap, the rules cannot identify gold messages at inference, and new insertions displace existing hits. Joint gains must be measured, not summed. |

## Fixed post-hoc policy checks

These policies were replayed without training or threshold searches. They are diagnostic counterchecks, not fresh adoption tests. Each returns exactly ten original messages.

| Policy | Hits | Rescued misses | Lost hits | Hit@10 | Change |
|---|---:|---:|---:|---:|---:|
| Existing ranker top ten | 1,274 | — | — | 70.19% | — |
| Top eight + unused dense-chunk winner + unused plain BM25 winner | 1,202 | 10 | 82 | 66.23% | −3.97 pp |
| Top eight + unused dense-chunk winner + unused question-weighted BM25 winner | 1,205 | 14 | 83 | 66.39% | −3.80 pp |
| Top eight + unused dense-chunk winner + unused question-weighted chunk-BM25 winner | 1,208 | 17 | 83 | 66.56% | −3.64 pp |
| One adjacent-turn swap from ranks 11–20 into slot ten | 1,248 | 2 | 28 | 68.76% | −1.43 pp |

Quota order is eight ranker messages, then the highest unused dense-chunk message, then the highest unused named lexical message. The supplied lexical channel was ambiguous, so the three interpretations are reported rather than choosing a winner.

The adjacency diagnostic finds opposite-role immediate neighbors of a top-three message among ranks 11–20, picks the highest baseline-ranked eligible candidate, and replaces rank ten. The proposal did not supply a confidence threshold, so this diagnostic is explicitly ungated; it does not rule out a well-calibrated conditional policy. Its NDCG rises slightly to 0.4136 while Hit@10 falls, demonstrating why both metrics matter.

The proposed `bm25_b0_max == 0` gate fires on **0/1,815 questions**. Zero overlap with an unknown gold reference is different from zero overlap with the whole history. In fact, only 2/541 failures have literal BM25 zero for every reference; frequent non-content words can still produce nonzero scores.

Implementation and exact outputs: review_proposal.py (historical artifact: `review_proposal.py`), proposal_checks.json (historical artifact: `proposal_checks.json`).

## Revised sequence

### 1. Small CPU experiment: channel strength and disagreement

Keep the current 54 features, data, original labels, tree capacity, and complete candidate pools. Add one small feature group describing the best lexical rank, best dense rank, minimum across families, rank disagreement, and lexical-only/dense-only high-rank indicators. Use within-query rank fractions or calibrated percentiles; do not treat different raw scores as comparable probabilities. A zero lexical score must not become a strong lexical rank merely through chronological tie breaking.

Predefine the recipe and evaluate baseline versus added features in persona-separated development folds. Evaluate both current hits and current misses. Report rescued/lost questions, NDCG, and tail rescue. Inspect whether apparent gains simply duplicate the existing rank/z-score inputs. Do not assume another feature group will help after the recent weighting/proximity failure.

### 2. CPU experiment: learned use of final neighbor scores

This is more specific than the existing neighbor features: the current 20 features describe neighbors' cosine and weighted BM25 scores/ranks, not their final ranker scores.

Generate first-pass ranker predictions through **nested persona cross-fitting**. Within every outer development split, second-pass training examples get first-pass predictions from inner models that excluded their personas. Outer held-out examples get first-pass predictions from a model trained only on the outer fitting personas.

Add previous/next final-score rank and gap, availability indicators, and an observed role-transition indicator to the original 54 features. Treat same-episode identity as uncertain unless the source provides it. Fit a second-pass ranker over original messages. Preserve individual message IDs and the ten-message budget. Compare against the same first-pass baseline and against the channel-strength group; prune redundant additions.

Prefer this to mandatory turn bundling. If a boundary-swap controller is later tested, train it on all development questions with both rescue opportunities and harmful-displacement examples, and measure net changes. A positive reference found within two messages is not an available inference-time gate.

### 3. Conditional small pilot: sentence evidence with the existing encoder

Only after the CPU work, benchmark sentence encoding on a bounded development sample using the already-installed encoder. This is not a new ColBERT or SPLADE model, but it still has GPU time and storage costs.

Compare the existing 48-word windows with sentence-based spans. Preserve exact source offsets. Avoid snippets that lose their subject or antecedent: permit a small surrounding sentence when needed under a predefined rule. Keep all original messages eligible and aggregate evidence back to them.

Use maximum and top-two sentence similarities, with span-count/length controls. Do not silently add the existing dense max-minus-whole feature again. Measure preprocessing throughput and projected full-study cost before scaling. Shorter spans may isolate useful details or destroy context; measure which happens.

### 4. Evidence-label audit alongside the experiments

Review a stratified development sample including both misses and hits. Separate explicit personal evidence, implied preferences, quoted/fictional or uncertain-subject text, generic topical material, and plausible unannotated supporting messages. Keep original benchmark labels unchanged and store additional judgments separately.

For example, an audited sleep/focus query has a reference about staying up until 2 a.m. debugging, while the top retrieved messages also describe staying up late and struggling with next-day obligations. That is a concrete reason to examine annotation completeness; it is not proof of answer correctness or of a labeling error.

Use first-person markers only as soft evidence-span features, never as a hard filter. Retain assistant messages and third-person material when they may restate user-provided evidence. If Qwen is used for annotation, calibrate it on independently reviewed examples, allow uncertainty, require attribution/supporting quotes, and measure downstream answer quality separately.

### 5. Defer semantic expansion unless the earlier pilots justify its cost

The prior result shows that fixed RRF with the tested expansion channel hurt the stronger ranker; it does not establish that indiscriminate application was the sole reason.

If revisited, define an inference-available gate from development predictions: weak weighted coverage in retrieved messages, calibrated ranker uncertainty, and channel disagreement. Validate that the gate actually activates and identifies cases with net benefit. Gold-reference overlap cannot be used as the gate.

Compare no expansion, expansion as an auxiliary learned feature, and the same feature with the proposed gate. Limit generated text to generic possible supporting statements, never asserted persona facts. HyDE explicitly uses hypothetical documents that may contain false details; these are retrieval probes, not memories or labels. [HyDE paper](https://arxiv.org/abs/2212.10496)

## Evaluation and stopping rules

- Develop on the existing 5,107 questions/280 personas; use nested cross-fitting for any feature derived from a supervised first-pass model.
- Treat this 1,815-question audit as already inspected. Do not tune thresholds or select models on its results.
- Keep the reserved 1,785 questions/100 other personas for one final frozen comparison if a development candidate qualifies. Use the `relevance_methods` background split, which excludes all these personas, and retrain the matched control consistently.
- Adopt only above **+1 absolute Hit@10 point**, with a positive lower paired-persona bootstrap 95% bound and no NDCG regression. Report net rescued-minus-lost questions and actual runtime. Development comparisons are exploratory; fresh confirmation is required for a performance claim.
- Keep exactly ten original messages for the primary metric. Returning 20 messages yields observed Hit@20=80.88%, and returning 50 yields Hit@50=92.84%; neither is an improvement in Hit@10. Reranking only the old top 30 imposes an observed reference-coverage ceiling of 86.34%.
- Do not claim downstream answer accuracy from retrieval Hit@10. Use a fixed answering model, gold answer scoring, and the same context budget for a separate answer-quality test.
- Stop additional exact-match tinkering if the small CPU groups fail. Do not automatically download a larger model or promise 78–81%.

Recommended priority: **small channel-interaction test, then cross-fitted final-neighbor-score features; conditional sentence-evidence pilot; separate label audit.** Drop fixed quotas and automatic partner insertion. Keep ColBERT/SPLADE excluded under the user's hardware preference. [ColBERT's trained token-interaction architecture](https://arxiv.org/abs/2004.12832) is a separate future option, not a free CPU feature.


---

<a id="archived-report-11"></a>

## Archived source: README.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# BM25 experiments

The latest [conversation-position experiment](#archived-report-07) adds source-turn distance from the latest visible message and relative conversation position. On the original 1,815-question historical cohort, the matched 54-feature model reproduces **70.19% Hit@10**, distance alone reaches **71.18%**, relative position alone **71.57%**, and both features **71.35%**. All improve NDCG and have positive paired Hit-gain intervals. Development gains are +0.998629, +0.84 and +0.82 points respectively; none exceeds the frozen >1-point development gate, so the original profile is preserved. The historical cohort was previously inspected, so these comparisons are exploratory rather than a fresh confirmation. All six new tests and exact prediction reconstruction pass. Models, scores and decision (historical artifact: `position_features/selection.json`).

The preceding target80 run is [target80_run/FINDINGS.md](#archived-report-14). **80% was not reached; the existing 54-feature model remains selected.** A 63-feature sentence candidate cleared development screening (+1.35 points), but on 1,785 fresh questions it scored **68.24% Hit@10** versus the matched 54-feature baseline's **67.45%**: **+0.78 points**, paired 95% interval **[-0.48,+2.06]**. This fails the adoption gate. Decision (historical artifact: `target80_run/selection.json`), fresh comparison (historical artifact: `target80_run/confirmation_summary.json`), inference reconstruction (historical artifact: `target80_run/inference_smoke.json`). These fresh percentages use different questions from the historical 70.19% result. Reservation statements in older study reports describe their status at completion; those 1,785 questions have now been evaluated once in target80_run.

A separate completed study follows the [five-stage refinement plan](#archived-report-02). No candidate clears the one-point Hit@10 gate, so the **54-feature model with historical 70.19% confirmation Hit@10 remains selected**. On 5,107 development questions, the matched baseline reaches 68.59%, channel features 68.44%, self-score stacking 69.06%, final-neighbor scoring 68.32%, and channel-plus-neighbor scoring 68.34%. Final-neighbor scores improve NDCG and tail rescue, but lose more questions than they rescue. The sentence pilot reaches 67.0% versus its matched 68.5% baseline; learned and gated cached expansion both remain at 68.5%. A separate agent evidence audit covers 20 questions / 40 messages. All six new checks pass; the 1,785 reserved questions remain unevaluated. Decision (historical artifact: `context_refinement/selection.json`).

The preceding study tests [learned term weights and phrase/proximity features](#archived-report-12). Neither addition clears the frozen one-point adoption gate. On the same 5,107 development questions with nested persona-separated evaluation, the matched 54-feature baseline reaches **68.59% Hit@10**, learned weights **68.36%**, phrase/proximity **68.51%**, and their combination **68.51%**. Negation variants also fail the gate. All new development computation ran on CPU; the 1,785 newly reserved questions were not evaluated. Decision and diagnostics (historical artifact: `relevance_methods/selection.json`). These development percentages are not directly comparable to the previous held-out result below.

The selected model remains from [next_stage/FINDINGS.md](#archived-report-05). The selected experimental ranker reaches **70.19% Hit@10**, **0.4105 NDCG@10**, and **22.02% cosine-tail rescue** on 1,815 fresh non-forget questions from 100 personas. It is fitted on 5,107 development questions from 280 personas, with all choices frozen before confirmation. Current selected model and feature schema (historical artifact: `next_stage/selection.json`).

The current model uses **54 features**: the original 28 (including role and question-frequency weights), 20 directional neighbor features, and six dense chunk-cosine features. It has 300 trees, maximum depth 5 and up to 31 leaves per tree. On the same fresh questions, the original 28-feature/1,450-question control reaches 61.16%; the 28-feature control with the same 5,107 fitting questions and larger trees reaches 64.02%. The new groups add **6.17 points**, paired 95% interval **[+4.29, +8.08]**, over that matched control; total gain over the 1,450-question control is **9.04 points**.

Match statistics were pruned. The fixed MiniLM reranker fusion and local-4B query-expansion fusion failed their 200-question development pilots. The conservative facts recipe returned empty lists for all 100 corpus messages despite passing an explicit-fact sanity check; its projected full pass costs about 12.3 hours, so it was not scaled. All prior studies are preserved. Final validation/test remain untouched.

Actual selected top-ten rankings (historical artifact: `next_stage/selected_top10.jsonl`), paired comparisons (historical artifact: `next_stage/comparisons.csv`), learning curves (historical artifact: `next_stage/learning_curves.png`), inference reconstruction (historical artifact: `next_stage/inference_smoke.json`) and verification (historical artifact: `next_stage/verification.json`) are saved. All 31 checks pass. Rebuilding all 223 message scores in the inference smoke history reproduces the saved batch scores, and injected answer/reference fields cannot change features. Dense CPU similarities were built with `OPENBLAS_NUM_THREADS=4`; use that same setting for numerical reproduction:

```powershell
$env:OPENBLAS_NUM_THREADS = '4'
.venv/Scripts/python.exe -B bm25_test/next_stage/finish.py smoke
```

The preceding study is [question_weight_role/FINDINGS.md](#archived-report-08). It tests question-frequency weights and removal of `role_assistant` on 1,775 questions from 100 training personas. All four rankers use the original 1,450 training questions. Its historical selected candidate has 17 features and no role flag: historical selection (historical artifact: `question_weight_role/selection.json`). The latest study above uses the subsequently authorized 28-feature reference with role retained.

On this fresh sample, the 18-feature control reaches 57.80% Hit@10, removal of role reaches 56.96%, adding question-frequency features while retaining role reaches 59.27%, and combining weighting with role removal reaches 58.03%. Weighting without role gains 1.07 points, but its paired 95% interval [-0.11, +2.24] crosses zero, so the weighted group is not retained under the frozen gate. Removing role reduced NDCG as well; these results do not establish that role is useless. The stronger 28-feature model is preserved as a comparison, not selected against the user's role-removal instruction.

The historical role-free model can be applied through `rank_messages(question, messages, cosine_scores, content_token_lengths)` in question_weight_role/run_experiment.py (historical artifact: `question_weight_role/run_experiment.py`). It returns complete messages in score order. Its inference matches every saved score/rank for a 254-message history and is unchanged when roles are flipped. Its 24 checks pass. Term frequencies and weights (historical artifact: `question_weight_role/question_term_weights.csv`) and all four prior models are retained for inspection.

The previous study is [confirmation/FINDINGS.md](#archived-report-01). It uses 1,882 fresh non-forget questions from 100 training personas, with rankers fitted on the original 80 pilot personas. It completes all three earlier follow-up actions: the complete winning working baseline, independent confirmation, and a small matched ranker test with chunk features. Its results below are historical and use different questions from the latest study.

The standalone fusion baseline is working_baseline.json (historical artifact: `working_baseline.json`): complete-message EmbeddingGemma 2 cosine plus local BM25 with `b=0`, `k1=2`, equal RRF with constant 60, zero-aware lexical votes, and averaged reciprocal contributions for tied positive lexical scores. It is implemented by winning_combination.py (historical artifact: `winning_combination.py`).

The confirmed baseline reaches 43.62% Hit@10 versus cosine's 35.02%. The cosine + metadata ranker reaches 58.34%, the base fusion-feature ranker 59.40%, and the ranker with chunks 60.63%. The chunk increment is 1.22 points with a paired 95% interval of +0.06 to +2.43 points; it narrowly clears the frozen gate. Most of the improvement over raw cosine is already present in the cosine + metadata ranker.

confirmation/selection.json (historical artifact: `confirmation/selection.json`) records the previous experimental candidate. Its model files and exact feature schemas are under `confirmation/models/` and `confirmation/training_report.json`. Actual top-ten outputs for each question in that study are saved in `confirmation/winning_fusion_top10.jsonl`, `confirmation/base_ranker_top10.jsonl` and `confirmation/chunk_ranker_top10.jsonl`.

For a new question-visible JSONL history containing one original message per row (`message_id`, `message_index`, `role`, `text`), run from the project root:

```powershell
.venv/Scripts/python.exe -B bm25_test/winning_combination.py --messages PATH_TO_MESSAGES.jsonl --question "Your question" --output bm25_test/retrieved.json
```

For already computed aligned cosine scores, import `score_messages` from `winning_combination`. Both paths use the same complete winning fusion.

The original screening report remains in [FINDINGS.md](#archived-report-03), and the initial revised experiment plan is [IMPROVED_PLAN.md](#archived-report-04). Those historical artifacts are preserved. All new code, models and results remain in this folder; main project scripts, configuration, raw data and split IDs were not changed. Validation/test have not been used.


---

<a id="archived-report-12"></a>

## Archived source: relevance_methods/FINDINGS.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Learned term weights and phrase/proximity experiment

Development: **5,107 non-forget questions, 280 personas**, four outer folds. Term weighting uses three inner persona folds inside each outer fit. All methods retain the existing 54 baseline features.

| Method | Features | Hit@10 | NDCG@10 | Cosine-tail rescue |
|---|---:|---:|---:|---:|
| baseline54 | 54 | 68.59% | 0.3860 | 20.65% |
| phrase59 | 59 | 68.51% | 0.3837 | 20.58% |
| phrase_neg61 | 61 | 68.44% | 0.3864 | 20.73% |
| learned58 | 58 | 68.36% | 0.3847 | 20.48% |
| combined63 | 63 | 68.51% | 0.3856 | 20.43% |
| combined_neg65 | 65 | 68.30% | 0.3856 | 20.34% |

| Comparison | Hit change (pp) | Paired persona 95% CI | NDCG change | Gate |
|---|---:|---|---:|---|
| phrase59 vs baseline54 | -0.08 | [-0.60, +0.47] | -0.0023 | FAIL |
| phrase_neg61 vs baseline54 | -0.16 | [-0.69, +0.39] | +0.0004 | FAIL |
| learned58 vs baseline54 | -0.23 | [-0.77, +0.31] | -0.0012 | FAIL |
| combined63 vs baseline54 | -0.08 | [-0.60, +0.43] | -0.0004 | FAIL |
| combined_neg65 vs baseline54 | -0.29 | [-0.82, +0.21] | -0.0004 | FAIL |
| phrase_neg61 vs phrase59 | -0.08 | [-0.54, +0.41] | +0.0027 | FAIL |
| combined_neg65 vs combined63 | -0.22 | [-0.64, +0.22] | -0.0000 | FAIL |

Development selection: **baseline54**. Adoption requires >1 absolute Hit point, a positive lower paired-persona bootstrap 95% bound, and no NDCG regression.

**Stop: no new group cleared the development gate.** The 1,785 questions from 100 newly reserved personas were not labeled, encoded or evaluated. The previously selected next_stage model remains the current method.

## What was tested

- Learned query-term weighting: eight query/statistic features predict a bounded 0.5–2 multiplier on existing question-frequency weights. Pairwise logistic fitting uses original positives against cosine-hard and random benchmark negatives. Four score summaries enter the ranker.
- Phrase/proximity: exact bigram/trigram IDF, ordered pairs within four tokens, unordered pairs within eight tokens, and same-sentence coverage. Two additional diagnostic features encode heuristic negation context. Original positions and sentence boundaries are retained.
- No neural model or Qwen labeling was added. Existing whole-message and dense chunk cosine caches were reused. All new computation in development runs on CPU.

## Interpretation

These results test the specified small regularized weighting model and fixed proximity features, not every possible learned lexical model. Original reference labels remain incomplete relevance judgments; hard negatives are unlisted messages, not verified irrelevant messages. Phrase heuristics do not establish semantic contradiction.

The historical 70.19% result used different questions. The table above is development cross-validation on identical questions for all candidates, with recomputed background statistics excluding the new reserved personas. It is not a new held-out estimate of the historical model.

Background statistics use 320 personas, 74,136 messages and 7,466 questions; all 380 previously inspected and 100 reserved personas are excluded.

Protocol (historical artifact: `protocol.json`), development comparisons (historical artifact: `development_comparisons.csv`), standalone lexical scores (historical artifact: `standalone_lexical_metrics.csv`), question-type/zero-overlap results (historical artifact: `subgroup_metrics.csv`), nested cross-fit provenance (historical artifact: `crossfit_audit.json`), learned weights (historical artifact: `learned_term_examples.csv`), timings (historical artifact: `training_timings.csv`), verification (historical artifact: `verification.json`).

All new artifacts remain in this folder. Historical studies, original labels, main scripts, split IDs and validation/test are unchanged.

## Why the learned weighting added little

The learned score has mean Spearman rank correlation **0.9996** with the existing question-frequency-weighted score, and **96.9%** mean overlap in their top ten. This supports substantial redundancy between these two lexical signals. It does not prove that every possible learned weighting model would be redundant.

Standalone Hit@10 increases from **32.70% to 32.97%** when replacing frequency-only weights with the cross-fitted learned weights. The gain is **+0.27 points**, paired persona 95% interval **[-0.02, +0.57]**, and fails the one-point gate. Adding the score to the complete ranker also fails the gate.

The fitted adjustments mainly attenuate scores: the mean multiplier is 0.792 for generic/stoplist terms and 0.850 for other terms. These are descriptive averages, not causal estimates of the value of individual words.

The phrase and negation variants yield small, statistically inconclusive differences. All five extensions have point estimates below the matched baseline; their paired 95% intervals include zero and have upper bounds below +1 point. The tested recipes provide no evidence of the desired incremental gain. Exact-match weighting/proximity cannot directly match absent vocabulary.

## Verification and cost

Development feature construction took **154.7 seconds** for **1,167,923 candidate rows**; ranker training, comparisons and final control fitting took **587.0 seconds**. All new development computation ran on CPU. No ColBERT, SPLADE, Qwen labeling or new embedding inference was used. These times exclude initial data preparation and final provenance verification.

All **8 new unit checks pass**. A separate inference reconstruction reproduces all 54 baseline, 7 phrase/negation and 4 learned-weight features exactly for a 208-message history, and reproduces the loaded ranker predictions. The 16 nested fits preserve persona separation; all 17 term-model optimizers, including the separate full-development fit, converge. Tests (historical artifact: `tests.log`), inference check (historical artifact: `inference_smoke.json`), weight diagnostics (historical artifact: `weight_diagnostics.json`), decision (historical artifact: `selection.json`).

The current selected profile remains the prior 54-feature model (historical artifact: `../next_stage/selection.json`). Its historical 70.19% held-out Hit@10 is unchanged; it should not be compared directly with these development cross-validation percentages. The new 1,785-question reservation was not evaluated.

None of the six methods has a tied score at the rank-ten cutoff. For subgroup and tie-table reproduction, use `python -B bm25_test/relevance_methods/diagnostics.py`; this post-fit entry point loads each prediction array once and avoids repeated decompression in the frozen main script. It does not change scores or model selection.


---

<a id="archived-report-13"></a>

## Archived source: target80_review/IMPROVED_PLAN.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Review and revised plan for 80% Hit@10

Status: analysis and fixed-policy checks only. No retraining, new embeddings, relabeling, or model adoption. The existing 54-feature model remains selected.

## What reaching 80% requires

The original confirmation sample has 1,274 hits out of 1,815 questions (70.19%). Reaching 80% on those questions would require 1,452 hits: **178 additional hits net of regressions**, or 32.9% of the 541 misses. These are planning arithmetic, not a promise on new questions.

- Rescuing all 145 current channel-top-ten misses with no losses would reach **78.18%**. This bounds a strategy confined to those particular misses; it does not cap every possible CPU reranker.
- The channel-hit and within-two-message-adjacency groups overlap in 29 questions. Their union is 208 misses, not 237. Rescuing all 208 without losing any hit would reach 81.65%, an oracle bound requiring knowledge of the references.
- There are 194 reference-near-misses at ranks 11–20. Recovering 178 of them with no losses would require 91.8% recovery. Merely returning twenty messages gives 80.88% Hit@20, which does not satisfy Hit@10.
- The existing top thirty contain a reference for 86.34% of questions, and the top fifty for 92.84%. This shows ranking headroom, not an achievable guarantee. Keep the full original-message candidate pools in the proposed lightweight ranker.

## Corrections to the supplied analysis

1. **The main counts are reproducible.** The 541 misses split into 145 with a channel hit and 396 without one. The 145 split into 90 lexical-only, 38 dense-only and 17 both-channel cases. In the 396 cases, maximum content overlap across references is zero for 184 and one for 151.
2. **The proposed causal explanations remain hypotheses.** Those counts do not establish that trees require lexical/dense agreement. Metadata, neighbor scores, annotation boundaries, and competing messages can also explain ordering. Examine per-message feature contributions and controlled group ablations before assigning a cause.
3. **Both channels can hit different references.** Only 12 of the 17 both-channel misses have a single reference message reached by both channel families. Five involve different reference messages. Query-level and message-level claims must be separated.
4. **Adjacency is not proof of a conversation match or annotation mistake.** The 92 cases measure source-index distance. Some are deep misses; 19 have best reference rank below fifty. The benchmark often labels several turns, not necessarily just one side of a user/assistant pair. Inspect actual content before treating a neighbor as interchangeable evidence.
5. **The baseline already uses short dense windows:** messages longer than 64 lexical words use 48-word windows with stride 24. The proposed improvement must be sentence/short-span segmentation beyond this existing representation, not a claim that the model only has 128-token or full-message embeddings.
6. **A zero-reference-overlap gate leaks unavailable information.** We do not know the reference at inference. The proposed observable gate, maximum BM25 score across candidates equal to zero, fires on **0/1,815** original confirmation questions and **0/5,107** development questions. Even the stoplist-filtered chunk-score maximum is nonzero on every question.
7. **First-person text is not guaranteed personal evidence.** Original references include assistant rewrites, quoted emails, fictional narratives and generic setup turns. First-person markers can describe a character or someone else; third-person passages can still be useful. Use an auxiliary view and attribution flags, never a hard candidate filter or automatic user-fact assertion.
8. **Several peakiness signals already exist.** The 54 features include dense chunk maximum, top-two mean, and maximum-minus-whole-message cosine. New sentence-based views may add information; duplicate arithmetic features are not independent evidence. High peakiness alone does not prove relevance.
9. **The 78–81% forecast is unsupported.** Rescue opportunities overlap and replacement policies can lose existing hits. The earlier expansion experiment tested a particular fixed-RRF integration; it did not establish that indiscriminate application was the sole cause of failure.

## Fixed-policy checks using saved predictions

These policies use no reference labels when choosing messages. They are retrospective diagnostics on already inspected samples, not independent confirmation of a new model. No threshold or policy was tuned from these results.

| Policy | Original confirmation Hit@10 | Rescued | Lost | Net hits | Development OOF Hit@10 |
|---|---:|---:|---:|---:|---:|
| Baseline | 70.19% | — | — | — | 68.59% |
| Top eight + highest new dense-chunk candidate + highest new raw BM25 candidate | 66.23% | 10 | 82 | -72 | 65.40% |
| Same, using question-weighted chunk BM25 | 66.56% | 17 | 83 | -66 | 65.52% |
| One automatic adjacency insertion | 68.76% | 2 | 28 | -26 | 67.67% |

The adjacency check keeps the top nine, then inserts the highest base-ranked message at ranks 11–20 adjacent to a top-three message, if one exists; otherwise it retains the original tenth. This is a concrete ungated version of the proposal, not a test of a learned confidence gate. It demonstrates that automatic insertion is not free. Quota selection deduplicates candidates and always returns exactly ten original messages.

Development uses the matched 54-feature baseline from relevance_methods with its background statistics; original confirmation uses the unchanged next_stage selected scores. Compare variants only against their own same-sample baseline. Checks (historical artifact: `policy_checks.csv`), script (historical artifact: `check_proposals.py`).

## Stage A: CPU-only learned correction of the current ranker

Purpose: recover valuable single-channel candidates and nearby evidence without unconditional replacement.

Keep the 54 features, role, question-frequency weighting, full candidate pools and original labels. Test two feature groups separately, then together:

**A1. Channel disagreement and strength.** Collapse correlated lexical variants into a lexical family and cosine variants into a semantic family. Add best lexical percentile rank, best semantic percentile rank, best rank across families, signed/absolute rank disagreement, and separate one-family-dominant flags. Lexical zero scores cannot count as strong evidence; use tie-aware ranks. Prefer within-question ranks over a blind maximum of differently distributed z-scores. Query-level score concentration and agreement can condition whether an isolated channel peak deserves trust.

**A2. First-pass ranker and neighbor evidence.** Add the first-pass 54-feature score/rank, margin relative to its tenth candidate, and directional scores/ranks of visible neighbors at offsets ±1 and ±2. These differ from the existing neighbor features, which propagate cosine and weighted BM25 rather than the final learned ranker score. Include availability and local coherence indicators; never cross missing source indices or assume every adjacent turn belongs to the same episode.

Train a small second-stage ranker on these inputs, retaining the original 54 features and first-pass score. Compare it with simply appending A1 to the original ranker. Add a fixed small blend grid with the original score only inside development; retain the unchanged model as the zero-change control. Do not call arbitrary raw score margins calibrated confidence probabilities.

**Prevent stacking leakage:** use four outer persona folds. Inside each outer training partition, fit first-pass rankers on three inner persona folds and create held-inner predictions for training the second stage. Score outer held personas with a first-pass ranker fitted only on outer training personas. A globally generated OOF score array is not sufficient for nested second-stage evaluation if its training-fold models include outer held personas. For final fitting, use development cross-fitted inputs for the correction model and a full-development first-pass model for inference.

Evaluate ordinary top-ten output first. A learned one-slot replacement policy is a separate ablation, trained on OOF swaps and judged by **rescued minus lost** questions. Do not hard-code a lexical/dense quota or automatic partner insertion.

Stop CPU feature additions if these groups fail the predefined gain gate. The previous learned lexical weights and phrase/proximity features already failed; do not repeat those experiments under new names.

## Stage B: Sentence evidence using the existing embedding model

Purpose: supply genuinely new evidence for implicit details that the current whole-message/48-word views do not isolate well.

1. Split source messages into sentences, preserving original spans, punctuation, negation and parent-message IDs. For very short fragments or pronoun-dependent sentences, add a contextual view with the preceding sentence from the same message. Split overlong sentences into a fixed short-window recipe. Freeze the recipe before quality evaluation.
2. Deduplicate spans and encode with the same pinned EmbeddingGemma model. Reuse query embeddings and existing message/chunk embeddings. This requires new inference, but no new neural model and no ColBERT/SPLADE. Measure CPU and small-batch GPU throughput before deciding full cost; do not promise a runtime.
3. Pilot on a seeded 200-question development-persona sample and its complete candidate histories. Candidate selection must not use reference labels. If the representation is promising, compute features for full development and test with persona-separated OOF evaluation before opening fresh confirmation.
4. Add sentence maximum, top-two mean, within-question rank and standardized score, sentence-minus-existing-chunk score, and log span count. Span count helps account for the greater chance of a random high maximum in long messages. Retain complete-message context for coreference and attribution.
5. Reuse the same sentence embeddings for a soft personal-evidence view: first-person/preferences/constraints cues can choose auxiliary spans, but preserve every original candidate and both roles. Record whether supporting text is user-authored, quoted, rewritten or fictional when that can be determined; do not infer a diagnosis or personal trait from a weak cue.
6. Test sentence features alone and on top of the best Stage A model. Report rescue separately for the original channel-hit and all-channel-miss categories as diagnostic slices, never as inference inputs.

Set a measured cost ceiling before bulk encoding, e.g. a short pilot followed by a maximum two-hour projected full pass. A small pilot cannot provide reliable evidence for a one-point gain by itself; use it for compatibility, cost and gross failure screening, then use full development for the adoption test.

## Stage C: Conditional vocabulary expansion or source descriptors, only if needed

The proposed zero-BM25 gate does not work on these data. Instead, train a query-level usefulness gate using development OOF results and only observable signals: disagreement among candidate lists, calibrated first-pass uncertainty, score concentration, and support coverage in retrieved text. Freeze its operating point on development and measure rescue precision, existing-hit regressions and fraction of queries processed.

Reuse the available local Qwen model for a bounded experiment, not an immediate bulk job. Generate topical retrieval hints or source-grounded evidence descriptors; do not fabricate first-person persona statements and treat them as facts. Keep original-query retrieval intact. Add the expansion score as an optional learned feature or controlled gated contribution rather than unconditional RRF. The earlier negative RRF result does not rule out this different integration.

For source descriptors, repair and manually audit the extraction recipe before scaling: the previous 100-message pass produced no facts. Check support and correct subject attribution, not merely exact quote presence. Separate descriptions of fictional characters and quoted authors from user statements. An extraction system that guesses hidden persona attributes is not a valid fix for retrieval.

HyDE generates hypothetical text that can contain false details; the representation is a retrieval aid, not evidence about the user. [Original HyDE paper](https://arxiv.org/abs/2212.10496).

## Hardware and methods excluded for now

Do not add ColBERT or SPLADE under the user's current GPU constraint. Restricting ColBERT to thirty candidates does not eliminate document-token encoding, query encoding, or token-vector storage; the claimed 5–10 ms CPU latency was not measured on this machine. The architecture requires a model trained for token-level matching; raw token vectors from the current pooled encoder are not a drop-in equivalent. [ColBERT paper](https://arxiv.org/abs/2004.12832).

## Evaluation and decision rules

- Development: the existing 5,107 questions / 280 personas. Use the preserved persona folds and nested evaluation for supervised first-pass features.
- The old 1,815-question result is now diagnostic evidence after repeated inspection. Do not tune against it and then report a new score there as independent confirmation.
- Final confirmation: the 1,785 questions from the 100 personas already reserved in relevance_methods. Their labels have not been evaluated. Use background statistics excluding those personas; compare a rebuilt matched 54-feature baseline and the frozen final candidate on exactly the same questions. Existing relevance_methods statistics and baseline caches provide this isolation.
- Keep the official original labels and exactly ten unique original messages. A separate human/Qwen relevance audit can diagnose label quality, but relabeling the evaluation set cannot establish a comparable improvement from 70.19% to 80%.
- Select feature groups on development with >1 absolute Hit@10 point, positive lower paired-persona bootstrap 95% bound, and no NDCG regression. Test each group separately, combined, and by leave-one-out pruning. Freeze one final model, integration rule and thresholds before confirmation.
- Report Hit@10, NDCG@10, tail rescue, rescued/lost questions, runtime, cache size and confidence intervals. The 80% goal is a fresh-confirmation point-estimate target; its confidence interval must also be reported. Smaller samples or an observed 80% do not prove the population rate is at least 80%.
- If a change clears the adoption gate but stays below 80%, retain only the demonstrated improvement and report the remaining gap. If none clears the gate, keep the current selected model. No forecast of guaranteed combined gains.

Recommended execution order: **learned CPU correction → sentence evidence → conditional semantic hints/source descriptors**. The proposal's automatic quota and automatic adjacency rules are rejected in their tested forms; useful parts of their motivation remain in the learned correction experiment.


---

<a id="archived-report-14"></a>

## Archived source: target80_run/FINDINGS.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Target80 experiment results

**80% was not reached. The existing 54-feature model remains selected.** On 1,785 fresh questions, the matched 54-feature baseline scores **67.45% Hit@10** and the 63-feature sentence candidate scores **68.24%**. The gain is **+0.78 points**, paired persona 95% interval **[-0.48,+2.06]**, below the adoption requirement. The earlier 70.19% result used different questions.

The study tests learned corrections to the 54-feature ranker, sentence evidence with the same pinned embedding model, and a bounded learned expansion pilot. Full original-message pools, exact-reference labels and ten-message output are preserved.

## Development comparisons

Development uses 5,107 questions/280 personas and background frequencies excluding the reserved 100 personas. Supervised first-pass features use four outer persona folds and three inner folds per outer fit.

| Method | Hit@10 | NDCG@10 | Tail rescue |
|---|---:|---:|---:|
| baseline | 68.59% | 0.3860 | 20.65% |
| or | 68.73% | 0.3843 | 20.34% |
| stack | 69.10% | 0.4076 | 23.21% |
| or_stack | 69.22% | 0.4269 | 26.47% |
| one_slot | 67.30% | 0.3932 | 22.53% |
| sentence | 69.94% | 0.3962 | 22.04% |
| A_sentence | 69.47% | 0.4317 | 26.81% |

Selected development method: **B/sentence**. Its Hit@10 is **69.94%**. Selection requires >1 absolute Hit point, positive lower paired-persona bootstrap 95% bound, and no NDCG regression.

Blends are selected from 0, 0.25, 0.5 and 1 using within-query standardized scores. Group retention and correction controls use identical questions. A one-slot learned policy is a separate classifier ablation, not a fixed channel quota.

| Comparison | Hit change(pp) | Paired persona95% CI | NDCG change | Gate |
|---|---:|---|---:|---|
| or vs baseline | +0.14 | [-0.42, +0.69] | -0.0017 | FAIL |
| stack vs baseline | +0.51 | [+0.11, +0.92] | +0.0216 | FAIL |
| or_stack vs baseline | +0.63 | [+0.06, +1.20] | +0.0410 | FAIL |
| one_slot vs baseline | -1.29 | [-1.71, -0.89] | +0.0072 | FAIL |
| sentence vs best_A | +1.35 | [+0.65, +2.07] | +0.0102 | PASS |
| A_sentence vs best_A | +0.88 | [+0.27, +1.52] | +0.0457 | FAIL |

## Sentence experiment

Seeded 200-question runtime/quality pilot: sentence-only Hit@10 **43.50%**, existing 48-word chunk cosine **45.50%**. Projected full encoding 34.9min; full-scale decision True. This pilot is not an independent confirmation or a precise estimate of a one-point gain.

Encoded/reused 483,660 unique spans; successful full invocation took 28.0min. Peak allocated GPU memory 2525MiB. Sentence splitting preserves source text and adds contextual views for short/pronoun-dependent sentences. Long sentences use 24-word windows/stride12. Personal-cue views are auxiliary hints, not asserted user facts.

## Conditional expansion pilot

Reused 200 cached question-only Qwen expansions with recorded provenance. The new pilot trains a small ranker with expansion features and a query-level usefulness gate under nested persona exclusion. Gate utility is a proxy learned from matched small pilot models; it is not proven calibrated for the full ranker. Full scale decision: **False**. No personal facts were generated.

| Alpha | Gated | Processed | Hit@10 | Change(pp) | Gate |
|---|---|---:|---:|---:|---|
| 0.25 | False | 100.0% | 69.50% | +1.50 | FAIL |
| 0.25 | True | 0.0% | 68.00% | +0.00 | FAIL |
| 0.5 | False | 100.0% | 70.00% | +2.00 | FAIL |
| 0.5 | True | 0.0% | 68.00% | +0.00 | FAIL |
| 1 | False | 100.0% | 71.00% | +3.00 | FAIL |
| 1 | True | 0.0% | 68.00% | +0.00 | FAIL |

## Fresh confirmation and decision

On **1,785 reserved questions/100 personas**, matched 54-feature baseline Hit@10 is **67.45%**, candidate **68.24%**. Change **+0.78pp**, paired persona95% CI **[-0.48,+2.06]**. Rescued 75, lost 61. Candidate adopted: **False**. **80% target reached: False**.

## Scope and reproducibility

Historical 70.19% was measured on a different sample and cannot be compared directly with development OOF percentages or a new confirmation sample. Observed improvements are benchmark retrieval improvements, not demonstrated downstream answer-accuracy gains. Validation/test, historical labels, source scripts and previous selected models are preserved. All new artifacts are under this folder. ColBERT/SPLADE and new large neural models were not used.

Protocol (historical artifact: `protocol.json`), StageA comparisons (historical artifact: `A_comparisons.csv`), StageA selection (historical artifact: `A_selection.json`), nested fit provenance (historical artifact: `nested/outer0_audit.json`), frozen candidate (historical artifact: `frozen_candidate.json`), decision (historical artifact: `selection.json`), tests (historical artifact: `tests.log`), verification (historical artifact: `verification.json`).

[Supplementary uncertainty, blend-grid and tie diagnostics](#archived-report-15).

All six new unit checks pass. The separate 63-feature inference reconstruction matches the reserved-history batch features and model scores and is invariant to injected gold fields. Inference check (historical artifact: `inference_smoke.json`). Windows denied cache hard-link creation; a byte-identical file copy materialized the completed cache. No frozen model, vectors, or predictive source was changed.


---

<a id="archived-report-15"></a>

## Archived source: target80_run/SUPPLEMENT.md

Historical text follows. Selection decisions, reservation status, future plans and removed paths describe that source document at its completion; the current selection is stated at the start of this consolidated report.

# Supplementary diagnostics

These outputs were computed after model selection; they do not change the frozen model or its integration rule.

All predefined blend-grid results (historical artifact: `development_blend_grid.csv`), cutoff ties (historical artifact: `tie_diagnostics.csv`), nested baseline verification (historical artifact: `nested_verification.json`), sentence-cache verification (historical artifact: `sentence_verification.json`), implementation notes (historical artifact: `implementation_notes.json`).

The one-slot classifier estimates message relevance and uses that ordering as a proxy for useful swaps. It is not a direct optimizer of query-level Hit@10 or a demonstrated calibrated probability model. The negative result does not rule out other learned slate policies.

Sentence features use case-preserving source spans, while the previous48-word auxiliary spans use NFKC/casefold text. The gain applies to these concrete views rather than sentence boundaries alone.

The expansion pilot uses a matched unexpanded-global-BM25 control with the same learned ranker inputs, plus the strong selected-model control. Neither expansion comparison establishes a reliable incremental gain. The fitted usefulness gate processes0% at its fixed operating point, so the gated branch reproduces the reference. This is a small-pilot gate limitation, not evidence that all future uncertainty gates are useless.

## Fresh Hit@10 uncertainty

| Method | Hit@10 | Persona-bootstrap95% CI |
|---|---:|---|
| baseline54 | 67.45% | [65.52%,69.36%] |
| candidate | 68.24% | [66.10%,70.45%] |

Model choices were frozen before reserved labels were loaded. Bootstrap intervals use10000 resamples of whole personas. No post-confirmation parameter changes were made.



## Standalone 55-feature package verification before cleanup

```json
{
  "passed": true,
  "histories": 16,
  "candidate_rows": 3736,
  "features": 55,
  "max_feature_error": 0.0,
  "all_saved_model_scores_bit_exact": true,
  "all_top10_exact": true,
  "injected_gold_fields_cannot_change_features": true,
  "checks": [
    {
      "question_id": "v2:train:00062",
      "question_type": "therapy_background",
      "messages": 223,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:06267",
      "question_type": "stereotypical_pref",
      "messages": 220,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:val:01727",
      "question_type": "neutral_preferences",
      "messages": 212,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:03257",
      "question_type": "anti_stereotypical_pref",
      "messages": 253,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:04701",
      "question_type": "health_and_medical_conditions",
      "messages": 253,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:01056",
      "question_type": "therapy_background",
      "messages": 223,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:12749",
      "question_type": "stereotypical_pref",
      "messages": 223,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:13191",
      "question_type": "neutral_preferences",
      "messages": 251,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:06560",
      "question_type": "anti_stereotypical_pref",
      "messages": 251,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:14653",
      "question_type": "health_and_medical_conditions",
      "messages": 251,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:val:01265",
      "question_type": "therapy_background",
      "messages": 227,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:16304",
      "question_type": "stereotypical_pref",
      "messages": 253,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:17446",
      "question_type": "neutral_preferences",
      "messages": 251,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:13868",
      "question_type": "anti_stereotypical_pref",
      "messages": 247,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:02911",
      "question_type": "health_and_medical_conditions",
      "messages": 199,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    },
    {
      "question_id": "v2:train:09029",
      "question_type": "therapy_background",
      "messages": 199,
      "max_feature_error": 0.0,
      "saved_scores_bit_exact": true,
      "top10_exact": true
    }
  ]
}
```


## Earlier package cleanup verification (historical)

At the original cleanup the experiment folder contained `REPORT.md` and `ranker55.py`. The training source has since moved to `scripts/ranker55.py`, leaving this report in the experiment folder. All older experiment subfolders, alternative models, scripts and embedding caches were removed after exact reconstruction checks. Approximately 8.28 GB were removed. All 15 original Markdown documents are retained above. At that earlier cleanup, the 55-feature model/statistics were embedded in the architecture file. The subsequent training-only revision removed those assets.

The sampled verification IDs that contain `v2:val:` identify original CSV provenance; these examples belong to the active training split. No active validation/test split was evaluated.

```json
{
  "passed": true,
  "features": 55,
  "trees": 300,
  "fixture_messages": 223,
  "saved_scores_bit_exact": true,
  "model_sha256": "15a3852fcd09bc3bf7d070ccc13f49571e4209af82e9198d57fcf9c7f012fe93",
  "only_two_files_remain": true,
  "protected_files_verified_unchanged": 833,
  "architecture_file_sha256": "1bc41ef6820800884f2f87a4f20031ec806c604d03c10e716df6a67ea4420a05",
  "removed_experiment_bytes": 8276895998
}
```


## Training-only revision verification

The code now contains no embedded weights, fitted statistics, training identities, samples, or saved predictions. The input validation, external feature-file training, 300-tree fitting and model reload were checked on a small synthetic dataset; temporary test artifacts were removed. No full benchmark training or new accuracy evaluation was run.

```json
{
  "training_source_only": true,
  "embedded_assets_removed": true,
  "feature_construction_valid": true,
  "gold_fields_cannot_change_features": true,
  "feature_schema_order_checked": true,
  "external_npz_training_and_model_reload_passed": true,
  "synthetic_questions": 60,
  "synthetic_candidate_rows": 1200,
  "fitted_trees": 300,
  "full_dataset_training_run": false,
  "source_bytes": 16542,
  "source_sha256": "a52b01159c00ce2d0f5a4695f3214132575758e8bf1b1712b091c8459187bfab"
}
```


## Training-source correctness review and overwrite behavior

The user requested existing model outputs to be overwritten. Both the command line and Python training functions now support this. A new model is fully written and flushed in a temporary file beside its destination, then atomically replaces the existing output. Failed training, serialization or replacement leaves the previous output intact, and temporary files are cleaned up. Input-data and script path collisions are rejected.

The review also corrected:

- An off-by-one visibility error: `source_end_index` is exclusive, matching `history[:end]` in the dataset loader. A message at the cutoff is rejected.
- Fractional, boolean, negative or overflowing source indices/token counts, which could otherwise silently distort ordering or lengths.
- Invalid query, message or chunk embeddings: exact row counts, 768 dimensions, finite real values and L2 normalization are checked before dot products/clipping. Extra chunk vectors can no longer be silently ignored.
- Invalid corpus counts or query-term document frequencies outside the corpus range. These fail before embedding work begins.
- Training data with no positive/negative ranking contrast, complex feature values, overflowing offsets, or malformed feature archives.
- Repeated NPZ decompression: each required array is loaded once, and the archive is closed before fitting and saving.

**18 regression checks passed.** Valid feature matrices match the prior implementation bit for bit across 24 synthetic query/history cases, including 64/65-word chunk boundaries, source gaps, stopwords and empty lexical queries. A 300-tree model trained on 1,200 synthetic rows produces identical predictions to the previous training implementation. Tests cover API/CLI overwrite, model reload, failure preservation, schema validation and injected gold-metadata independence. This is a correctness check, not a new benchmark accuracy result. Temporary review code, synthetic datasets and test models were removed; only the training source and consolidated report remain.

Reviewed training source SHA-256: `61c098b77b0d1b8e51f3a31b0203e256982f30e090e1681aa05980b31c5144db`.
