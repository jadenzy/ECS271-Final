# Laptop LightGBM memory-selector plan

The consolidated proposal is workable with the protocol and resource corrections in [PLAN_REVIEW.md](PLAN_REVIEW.md), reviewed 2026-10-27. That review defines the revised Phase 0-5 ordering and implementation queue. The primary result uses one current query and visible history.

## Current state

- Existing persona-disjoint V2 splits: train 18,532 questions / 800 personas; validation 1,002 / 40; test 1,242 / 50. IDs retain original CSV prefixes; membership comes from the split files.
- Training excludes 2,061 sensitive_info and 64 ask_to_forget questions. Every retained training question has all reference messages present. Preserve these exclusions and the raw sources.
- Candidates are complete original user or assistant messages. Read raw messages directly using the active split IDs; regenerate BM25/cosine scores and labels before training; old generated retrieval data was deleted.
- Exact-reference labeling is implemented per original message. Regenerate training labels for the current unit. Historical counts below describe the removed combined-message format and must be recomputed.
- Training baseline diagnostics are implemented in scripts/diagnose_retrieval.py: full RRF rankings plus BM25/cosine/RRF positive-rank histograms, median/90th percentile and per-question Recall/Hit/full-reference coverage at 5/10/20/30/50. RRF uses constant 60 by default; outputs live in data/rrf/ and results/selector_diagnostics_single_message/v2/train/.
- Historical memory-reuse audit (recompute for single-message IDs): 28,546 positive occurrences correspond to 28,532 stable memory IDs. Only 14 IDs are positive for two distinct questions, across seven personas; none for more than two. The 4,944 cosine-rank>50 occurrences correspond to 4,939 IDs. Exact-text grouping within each persona also has a maximum of two distinct positive questions. Results are under results/selector_diagnostics/v2/train/memory_reuse/.
- Validation/test labels, selected-memory Qwen evaluation and the LightGBM trainer are pending. No selector training or new Qwen labeling has run.
- The sole frozen answer/extraction model is Unsloth Qwen3.5-4B Q4_K_M GGUF through the native llama.cpp CUDA backend. The dedicated server module owns all server/API work; the evaluator handles data and grading. FlashAttention and prompt caching are enabled; fit adjustment, context shifting and thinking are disabled. Historical results retain their original identities.
- LightGBM is the chosen learned selector. Keep EmbeddingGemma and Qwen frozen. Use BM25, cosine and reciprocal-rank fusion (RRF) as reference baselines.
- Hardware verified: 6 GiB RTX 3060 Laptop GPU and 15.76 GiB physical RAM. The prior combined-message format had 92,387 / 4,635 / 5,845 train/validation/test memory IDs; these are historical counts. A 20-feature float32 training matrix is 163.6 MiB before LightGBM overhead; use streaming/memmaps and one GPU model at a time.
- Primary ranking selection is non-forget type-macro Hit@10, supported by downstream answer accuracy and per-type reporting. Average questions within type, then types equally. Full-reference coverage is secondary. Reference Hit can count generic setup turns, so use audited content-bearing diagnostics and Qwen outcomes alongside it. Ranking metrics report eligibility coverage; absent eligible types are N/A.

## 1. Audit reference coverage and baseline rankings

Implemented deliverable: scripts/diagnose_retrieval.py for current training diagnostics. Remaining deliverables: scripts/audit_reference_coverage.py, validation eligibility/diagnostics and excluded-question cause analysis under results/selector_diagnostics/.

- Verify example_id/message_id joins, counts, persona disjointness, source hashes, full-history candidate construction and consistent BM25 corpus scope. Confirm answer fields and annotated snippet fields are never appended to retrieval inputs or used to construct candidate pools.
- Create a separate validation eligibility manifest and direct labels for complete-reference validation questions. Preserve active validation IDs; report coverage and excluded reasons by persona/question type. Missing references must never silently become negative labels.
- Freeze matching/recovery rules on training/validation. Exact test labels or extraction may be prepared blindly once rules/prompts are frozen; test recoveries or examples cannot guide development. Keep normalized/fuzzy recovery tiers separate from exact labels.
- Diagnose the 2,125 excluded training questions: distinguish absent messages from whitespace/Unicode/serialization differences. Keep the upstream sensitive_info diagnosis visible. Normalized/fuzzy matches are review candidates only; do not restore questions or alter labels automatically.
- Measure positive-rank distributions under BM25, cosine and RRF; Recall/Hit at 5/10/20/30/50; average precision; and score/candidate-count distributions. Record tie handling and the RRF constant. Use the same candidates across baselines.
- Add a small validation-tuned weighted z-score fusion baseline. Quantify tail-positive companion hits within +/-1 and +/-2 visible messages against matched random-neighbor controls before adding neighborhood features.
- Separate full-pool coverage from shortlist coverage. All visible memories are currently scored, so a reference below top 30 remains available to a full-pool ranker. If introducing a shortlist, measure its union recall; later reranking cannot recover omitted candidates.

Completion criterion: reproducible coverage/joins reports, a validation eligibility manifest and a baseline table before learned or judge-generated labels.

## 2. Run a small Qwen validation ceiling check

Planned deliverable: selected-memory support for the existing local evaluator and results/selector_diagnostics/qwen_pilot/.

- Start with 100 complete-reference validation questions for adapter/ceiling smoke checks, then use a fixed 300-question downstream pilot across personas, types and reference-coverage strata. Save IDs before comparisons; reuse questions, prompts, models and token accounting. Keep up to 200 eligible non-forget questions for a nested label audit where feasible.
- Compare gold-message-only context, a label-informed budget oracle, and BM25/cosine/RRF-selected context. Gold/oracle modes are diagnostics and never deployment inputs. Report unbudgeted gold results separately from budget-matched results.
- Use planned budgets of 1,024 / 2,048 / 4,096 Qwen memory tokens. These are experiments, not current evaluator defaults. Count wrappers/separators, reserve room for question/options and the answer, and set a larger total context when needed. Never truncate a memory or substitute embedding-token counts.
- Define the budget oracle as reference-coverage optimization over candidates under the same cost accounting; disclose any approximation. Reference coverage is not an oracle for answer correctness.
- Add an explicitly gold-dependent ever-positive-memory oracle to measure annotation-preserving pruning. It is not a deployment feature or an answer-accuracy upper bound.
- Report answer accuracy, NLL, tokens, packed memory count and reference coverage, with paired outcomes and persona-bootstrap intervals. Low gold accuracy motivates checking annotation/answering limitations; high gold accuracy with weak retrieval motivates ranking/packing improvements. Neither proves a cause alone.

Completion criterion: a matched pilot indicating where further work is most useful before a large labeling run.

## 3. Build richer, reproducible features

Planned deliverable: scripts/build_selector_features.py, a feature schema/provenance record and data/selector_features/.

- Join scores and labels by example_id/message_id, with contiguous question groups. Retain the two-score feature set as an ablation.
- Add within-question rank percentiles, z-scores and gaps from the best BM25/cosine candidate, safely normalized BM25 and RRF. Specify ties, zero variance and zero denominators.
- Add message role, character length, Qwen token cost and normalized history position. Save tokenizer identity and feature order/category encoding for inference.
- Measure visible-neighbor max/mean scores and whole-message token lengths by role. Sentence max-cosine is a separately benchmarked auxiliary feature, not a replacement for original complete-message memories. Count/deduplicate segments and benchmark 5,000 segments before projecting runtime.
- Features may use question text, visible history and retrieval computations available at inference. Exclude reference membership, answers/options, persona IDs and source split prefixes. Use CSV question_type/topic as evaluation strata unless independent deployment availability is established.
- A predicted-type head may use question embeddings, with out-of-fold training probabilities grouped by persona. Gold type is an oracle ablation only. Training has no sensitive_info class while validation/test retain 93/123 such questions; define an unsupported-class policy and report the gap.
- Later ablate preference/constraint phrasing and neighbor cosine summaries. Bound neighbors by visible history and real sessions where available; V2 currently provides role/content and position without a guaranteed session ID.
- Defer an "ever positive" learned prior. If justified later, use a LightGBM-based prior fitted out of fold by persona; validation/inference priors use training personas only.
- A different, deployable prior is grounded memory-only fact extraction. Benchmark 500 training memories and inspect 100 outputs before scaling. Preserve support quotes and speaker/subject attribution; quoted emails and assistant rewrites are not automatically user facts. Add fact cosine/has-facts only after quality and runtime checks. Compare a fixed 150-200-persona subset with and without these features; test prompt processing follows freeze. Details and gates are in PLAN_REVIEW.md.
- Cross-question features are disabled in the primary single-query deployment protocol. Candidate memories repeat across a median of 24 questions per persona, but this does not make the benchmark's other questions available during deployment. Gold positive frequency remains an audit statistic, never a direct inference feature.
- A separately reported benchmark-batch ablation may use the declared available unlabeled query set for the same persona, excluding the current query. Possible features are top-k frequency and mean reciprocal rank across those queries. Use only cases where the memory was visible in the other query's candidate pool; record the number of permitted queries as a denominator. No labels/answers, cross-persona aggregation or train/validation/test mixing are allowed.
- For online deployment, aggregate only genuinely observed prior queries with valid temporal/visibility information. Future questions and an assumed CSV-row chronology are unavailable. Match training and evaluation feature availability to the declared deployment protocol; persona-disjoint splitting alone does not establish availability of other queries.

Completion criterion: aligned finite features, inference-parity tests and no annotation-derived inputs.

## 4. Train controlled LightGBM baselines

Planned deliverable: scripts/train_lgb_selector.py, saved model/feature schema and validation reports.

- Use LGBMRanker with lambdarank, binary labels and label_gain=[0, 1]. Supply question sizes as group/eval_group. Fit on training personas and select settings on held-out validation personas; never randomly split candidate rows.
- Compare two-score and expanded-feature models against BM25, cosine, tuned z-score fusion and RRF using identical candidate pools, budgets and packing. LightGBM remains the required learned model; logistic regression is omitted from the immediate queue.
- Use a small tree-complexity/learning-rate search. Early-stop on a custom group-aware non-forget type-macro NDCG@10 metric with first-metric-only stopping; built-in query-averaged NDCG is not the same aggregation. Also report NDCG/Recall/Hit at 5/10/20, average precision, packed counts and budgeted coverage. Choose finalists by non-forget type-macro Hit@10 and downstream answer accuracy.
- Ablate positive monotonic constraints on raw BM25/cosine scores against unconstrained trees. Do not assume these constraints improve every question type.
- Keep binary exact-reference labels with gains [0, 1]. Each candidate is one original message; there are no member-count gains.
- Treat predictions as ranking scores, not calibrated probabilities. Do not import classifier class balancing or apply a probability threshold without a separate justified experiment. Plain row accuracy is not a selection metric.

Completion criterion: a reproducible comparison against fusion, with feature contributions identified. A weak result calls for diagnosing signal/supervision before broadening parameter searches.

## 5. Evaluate packing and target expensive improvements

Planned deliverables: validation selection reports, a conditional hard-negative audit and a frozen final-test protocol.

- Evaluate the same Qwen pilot, then shortlisted settings on the full validation split. Preserve all questions for answer evaluation; report complete/incomplete-reference strata and the eligible denominator for ranking metrics.
- Compare budget-only packing with a validation-tuned raw-score threshold plus budget, near-duplicate suppression and neighbor-message expansion; deduplicate overlapping source messages. A raw-score threshold is allowed, but is not a calibrated probability. Ablate chronological/score order and user-only/all-message candidates one factor at a time, with at most 12 pre-registered initial pilot conditions.
- If results point to missing positives, audit the top 5-10 unannotated candidates for about 200 fixed validation questions. Qwen judgments indicate possible annotation gaps; manually review a subset. Do not treat judgments as ground truth or silently flip exact labels.
- Freeze the audit action threshold first; the review proposes a 10% confirmed-useful-negative trigger, with manual adjudication and persona-aware uncertainty. This is an operational proposal, not an automatic Qwen-confidence cutoff.
- Only if justified, extend hard-candidate supervision to 2,000-3,000 training questions. Use batched relevance scoring without free-text generation where the existing local runtime supports it. Preserve judgments separately; test soft supervision or training-only negative masks explicitly. A cosine threshold such as 0.9 does not establish a false negative.
- Add a stronger encoder/cross-encoder only if diagnostics justify its cost. Rescore the full pool or a measured high-recall union shortlist; report coverage and latency. Top-30 rescoring cannot recover omitted memories.
- Benchmark a pinned 0.6B-class reranker on 100 pairs before a small full-pool pilot. In training, 317/340 short neutral tail positives miss both retrieval top-50 lists, so a top-30-only study cannot diagnose their recoverability. Gold-reference Hit alone is insufficient to reject a model that improves downstream answers.
- Freeze features, model, candidate policy, budget and prompt ordering before test evaluation. Report full-test Qwen accuracy/tokens and complete-reference-subset ranking metrics with coverage and paired persona-bootstrap intervals. Keep V1/LoCoMo as comparison datasets.

## Immediate next action and constraints

1. Implement reference coverage/recovery auditing, derived validation labels and a metric/protocol manifest.
2. Add validation weighted z-score fusion and the companion-versus-random-neighbor diagnostic.
3. Implement selected-memory Qwen evaluation with token preflight and a cached gold/baseline pilot.
4. Build streaming core features and the grouped LightGBM trainer with the custom non-forget macro stopping metric.
5. Benchmark sentence embeddings and fact extraction at small scale; if the early checkpoint indicates a semantic-ranking gap, include an optional 100-pair reranker timing check and small full-pool probe before bulk extraction. Choose what to scale from measured gains and cost.

scripts/diagnose_retrieval.py is implemented for training; the future scripts above remain planned. Training/extraction jobs and final test evaluation have not started. See PLAN_REVIEW.md for the revised phase details, gates and source references.

Preserve raw sources, current split IDs, exclusions and exact binary labels. Keep results under results/, generated training artifacts under data/, and record source/config/model hashes and seeds. Selector training has not started.

LightGBM references: https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html and https://lightgbm.readthedocs.io/en/stable/Parameters.html.
