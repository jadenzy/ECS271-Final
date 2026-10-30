# Review of the consolidated laptop plan

Current-format update: retrieval now uses one complete original user or assistant message. This supersedes the combined-message units, half-length features and member-count labels discussed in this historical review. Its measured candidate counts and retrieval results require recomputation for single-message IDs. See README.md for the active scripts and paths.

Reviewed 2026-10-27. Verdict: workable as a staged research plan, with the changes below. This review updates planning only; it does not launch training, extraction, model downloads or test evaluation.

## Actual machine and data

- GPU confirmed: NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB VRAM. Physical RAM: 15.76 GiB.
- Model migration update: only Unsloth Qwen3.5-4B Q4_K_M GGUF is retained for Qwen inference. The 0.8B and 2B weights and model-selection options are removed. The pinned native llama.cpp CUDA runtime uses FlashAttention and prefix caching; historical results retain their original model identities. Every complete prompt is checked by the active tokenizer against the explicit context cap. Run one GPU model at a time.
- Current unique candidate memories: train 92,387; validation 4,635; test 5,845. Validation plus test is 10,480 memory IDs.
- A 2,144,257 by 20 float32 matrix occupies 163.6 MiB. LightGBM datasets, binning, IDs and temporary copies add memory. Stream JSONL inputs into typed arrays/memmaps instead of loading all score objects into RAM.
- 500,000 auxiliary 768-dimensional float16 embeddings would occupy 732.4 MiB before metadata. Their actual count and runtime are unknown until segmentation, deduplication and a throughput pilot. Preserve existing float32 caches; do not convert them in place.
- LightGBM 4.7.0 is installed. RapidFuzz is not installed. The existing evaluator supports full-history and no-memory modes; selected-memory/oracle/extraction modes still require implementation.

## Changes required before treating the proposal as a frozen protocol

1. **Define the metric precisely.** Within each question type, average Hit@k across eligible questions; then average the non-forget type scores equally. Report the number of types, eligible questions and excluded questions. Missing-reference types are N/A, not zero. Downstream QA includes all validation/test questions and reports both per-type and non-forget type-macro accuracy. Full-reference coverage is secondary. Annotation Hit@k can still count generic setup units as success, so it cannot replace answer evaluation.
2. **Use a custom non-forget macro-NDCG stopping metric.** Built-in NDCG averages queries and does not by itself implement equal weighting of question types. Use contiguous question groups, a group-aware custom metric and an explicit first-metric-only early-stopping callback. Keep NDCG@5/10/20 as diagnostics and use non-forget type-macro NDCG@10 for the first stopping rule.
3. **Keep test processing separate from development.** Freeze exact/normalized matching policy, extraction prompts and selection settings on training/validation. Deterministic blind test preprocessing is permissible once those rules are frozen; inspecting test recoveries or extraction outputs to adjust them is not. Run the exploratory extraction pilot on training memories and validation, with test extraction deferred until the prompt/schema are frozen.
4. **Recovery is an audit, not an automatic relabel.** Preserve exact labels and current exclusions. Produce separate exact, Unicode/whitespace-normalized and fuzzy candidate-match tiers. Require same-role matches within the correct visible history. A similarity around 0.95 is not evidence of equivalent meaning and may match generic turns or altered constraints. Review fuzzy candidates, report failures by type and never claim that normalization repairs genuinely absent sensitive_info content.
5. **The prior oracle is diagnostic only.** Restricting validation candidates to memories that are ever positive across validation questions uses gold labels unavailable at deployment. It bounds annotation-preserving pruning, not Qwen answer accuracy. Report it beside gold-message-only and gold-positive-pair-only contexts; never feed this oracle flag to the primary selector or confuse it with a memory-only extracted-fact feature.
6. **Separate CPU features from new encoder work.** Rank/z-score/gap/fusion, neighbor scores and length/position are cheap. Sentence-level max-cosine requires new segmentation, caching and embeddings. Count/deduplicate segments, benchmark 5,000 representative segments, then estimate total wall time and peak VRAM. Keep sentence splits auxiliary; preserve full-pair retrieval IDs and texts.
7. **Predicted type needs inference parity.** Train any question-type head with persona-grouped out-of-fold probabilities for training examples. Train a final head on training personas for validation/test inference. Gold type is an explicitly labeled oracle ablation only. Training contains no sensitive_info questions, while validation/test retain 93/123 such questions; define unknown-class handling and do not assume the head can learn that missing class.
8. **Ground fact extraction and attribute speakers.** Extract from visible memory text only, without the question, gold snippets, answers or CSV persona profiles. Cache structured facts with supporting source quotes and subject/speaker attribution. Distinguish user facts from quoted third-party emails, suggestions and assistant-generated rewrites. Generic setup turns may yield no facts. Parse failures and unsupported outputs are missing/error states, not automatically "none". A fixed extractor can be audited on held-out personas; learned label-based heads require out-of-fold training. Ever-positive AUC is a secondary annotation diagnostic, not the acceptance gate.
9. **A training subset is an experiment, not a sample-size conclusion.** A 150-200-persona extraction pilot is sensible for cost control, but few features do not establish that 800 personas are unnecessary. Compare Phase 1 and Phase 2 on exactly the same subset, retain Phase 1's full-training baseline, and inspect a learning curve before scaling.
10. **Treat monotonic constraints as an ablation.** Compare unconstrained trees against positive constraints on appropriate score features. Keep the binary targets primary. LightGBM is the required learned model; a logistic-regression baseline is optional and is omitted from immediate work, consistent with the chosen model direction.
11. **Thresholding does not require pretending rank scores are probabilities.** Compare budget-only packing with a validation-tuned score-threshold-plus-budget policy. A raw-score threshold is allowed; calibration is needed if it is interpreted as probability/confidence. Count actual Qwen tokens, wrappers and separators, deduplicate overlapping source messages after expansion, and check full prompt plus answer reserve fits context. A 4,096-token memory budget cannot use a 4,096-token total context without room for the rest of the prompt.
12. **A top-30 cross-encoder cannot address the hard tail.** In training, 317 of 340 short neutral cosine-tail positives also fall outside BM25's top 50. Test the 0.6B reranker over the full visible pool on a small pilot or a measured high-recall union, then compare shortlists for latency. Use the documented task prompt/yes-no logit scoring. Its small size makes a benchmark reasonable; it does not establish speed or personalization quality on this laptop.
13. **Do not sweep the full Cartesian product of models, budgets, ordering, roles and deduplication.** Start with a fixed 300-question validation pilot, at most 12 pre-registered conditions, and shared cached answers. Sweep one factor at a time, then run finalists on all 1,002 validation questions. Use paired persona-bootstrap intervals; independent-binomial standard errors around 1.5 percentage points do not account for persona clustering, per-type macro averaging or repeated tuning.
14. **Do not require a large fact-extraction pass before a small reranker probe.** After the CPU baseline and gold-context checkpoint, a 100-pair reranker timing check and small full-pool validation probe can be compared with the extraction pilot if results indicate a semantic-ranking gap. Run GPU pilots sequentially. Pick the next signal from measured downstream gain and cost; neither 92k extraction nor a full reranking stage is mandatory.

## Revised execution order

### Phase 0: Protocol, eligibility and baseline audit

- Freeze primary single-query inputs, type-macro metric definitions, test-handling rules, candidate unit, cache keys and a small set of experimental gates.
- Implement reference-coverage auditing and derived validation eligibility labels without modifying active IDs. Audit normalized/fuzzy recovery of excluded training questions and validation; keep the test tier sealed until policy freeze.
- Reuse the verified chronological BM25 tie policy, and report the size of tied/zero-score groups. Add weighted z-score fusion, tuning a small cosine-weight grid on validation. Retain cosine, BM25 and RRF as recorded reference baselines.
- Quantify cosine-tail companion hits within +/-1 and +/-2 visible pairs at top 10/50 against matched random neighbors from the same query/history. Avoid using reference labels in the resulting features. The annotation often spans a generic setup and a content-bearing companion; inspect an audited subset separately.

Exit: eligibility/type-coverage tables, baseline table, neighbor/control result and a versioned protocol manifest.

### Phase 1: CPU features and first LightGBM model

- Build streaming feature tables with raw/within-question scores, tuned fusion, visible-neighbor max/mean, total/user/assistant token lengths and position. Preserve the two-score ablation.
- Train LGBMRanker on the full current training set with question groups and the custom validation stopping metric. Compare unconstrained/constrained trees with a small fixed search.
- Add predicted-type probabilities only after the out-of-fold pipeline and unsupported-class policy pass tests. Gold-type results remain diagnostic.
- Benchmark sentence segmentation/deduplication and 5,000 auxiliary embeddings separately; add user/assistant sentence max-cosine only if its measured cost is acceptable. Do not change original pair embeddings or retrieval inputs.

Exit: reproducible fusion and LightGBM results on the same validation examples, with type-level metrics and a memory/latency profile.

### Early downstream checkpoint: gold and selected-memory Qwen

- Implement selected-memory Qwen support before any large extraction job. Start with 100 complete-reference validation questions for smoke/ceiling checks, then use the fixed 300-question pilot, preserving complete/incomplete-reference strata.
- Compare no memory, cosine, tuned fusion, Phase 1, gold-message-only, gold-pair-only and the declared gold prior oracle. Keep unbudgeted oracle checks separate from budget-matched comparisons.
- Use 1,024/2,048/4,096 memory-token budgets, native Qwen token accounting and a context setting large enough for the full prompt. Cache by model/runtime/tokenizer/prompt/settings hashes and selected source text, not question ID alone.

Exit: evidence that ranking gains translate into useful answers, or a clear reason to revisit annotations/representation before spending more GPU time.

### Phase 2: Memory-only fact features, gated by measurements

- Benchmark 500 length-stratified training memories with the pinned Qwen3.5-4B Q4_K_M text extractor. Record prefill/decode throughput, output length, peak VRAM, parse rate, supported-fact precision and speaker attribution. Review 100 pilot outputs before scaling.
- Cache a short structured fact list plus supporting quotes or a valid empty list. Add fact-to-question cosine and a grounded has-facts flag. Prompt/model/format changes invalidate the cache.
- Extract for a fixed 150-200 training-persona subset and required validation memories only after the pilot passes. Report exact counts; compare the same-subset Phase 1/2 models and keep the full-training Phase 1 comparator. Test extraction happens after prompt freeze.
- Evaluate source-grounding quality and downstream gains. Treat ever-positive AUC as supporting analysis, not a substitute for question-specific ranking/QA.
- If the early checkpoint shows a semantic-ranking gap, compare this small extraction pilot with an optional 0.6B reranker probe before committing to bulk extraction. The later cross-encoder phase covers scaling/integration, not a requirement to postpone every small feasibility test.

### Phase 3: Conditional hard-negative audit

- On up to 200 fixed complete-reference non-forget validation questions, score top-10 exact-label-negative candidates with a verified yes/no forward-pass path. Model judgments are possible usefulness labels, not ground truth.
- Freeze the action threshold before seeing the audit; the proposed review trigger is a confirmed useful-negative rate of at least 10%, backed by manual adjudication and a persona-aware interval. This is an operational proposal, not a universal statistical cutoff.
- Only if justified, test soft supervision or masking on the training subset. Preserve exact labels, separate all generated judgments and require a matched validation comparison before adopting the change.

### Phase 4: Packing and final evaluation

- Ablate threshold/budget packing, conservative duplicate suppression, short-pair expansion, chronological/score ordering and user-only/full pairs one factor at a time. Avoid suppressing distinct updates or forget requests as duplicates merely because they are similar.
- Model selection is driven by non-forget type-macro Hit@10 and answer accuracy under the same budget; retain coverage/NDCG/Recall diagnostics and report per-type denominators. A proposed practical improvement target is 2 percentage points, interpreted with paired uncertainty and latency rather than as a significance cutoff. Downstream answer gains can justify a model even when annotation Hit@10 does not improve.
- Evaluate finalists on all validation questions; freeze prompts, features, models, weights, candidate pools and budgets. Run one final test-evaluation round over all 1,242 questions with pre-registered baselines and no test-driven retuning.

### Phase 5: Optional cross-encoder and declared batch experiment

- If useful answer/retrieval gains remain insufficient, benchmark a pinned Qwen3-Reranker-0.6B-class model on about 100 representative pairs before a full-pool pilot. Compare full-pool versus measured-recall shortlists, adapting instructions to personalization. Keep original context intact within the chosen cap.
- Cross-question/exclusivity features stay disabled for the primary protocol. A separate declared-availability batch study may use other unlabeled persona queries; it excludes the current query, labels/answers and cross-split aggregation. Online features can use genuinely observed past queries only.

## Immediate implementation queue

1. Reference-coverage/recovery auditor, validation eligibility labels and protocol/metric manifest.
2. Validation weighted-fusion baseline and companion-versus-random-neighbor diagnostic.
3. Selected-memory Qwen adapter with exact token preflight and cached gold/baseline pilot.
4. Streaming core-feature builder and LightGBM trainer with custom non-forget type-macro stopping.
5. Small sentence-embedding/fact-extraction throughput and grounding benchmarks, plus an optional 100-pair reranker timing check and small full-pool probe if indicated. Choose which auxiliary signal to scale from measured gains and cost.

The query-type head, bulk extraction, soft-label jobs and cross-encoder are gated later work. The immediate queue does not change datasets or run a final test experiment.

## Primary references

- [LightGBM ranker groups and custom metrics](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)
- [LightGBM early stopping](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html)
- [Qwen3-Reranker-0.6B model card and yes/no scoring](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [EmbeddingGemma 2 model card and selective encoder loading](https://huggingface.co/google/embeddinggemma-2)
