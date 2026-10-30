# Single-message memory retrieval

Each retrieval memory is **one complete original user or assistant message**. Its text is preserved exactly, with the role, source index and original message object stored as metadata. Retrieval messages remain whole and unchanged. Auxiliary chunks supply features while the retrieval unit stays the original message. System/persona messages and empty/non-text turns are excluded from retrieval.

Original sources remain under `data/raw/`, and active IDs under `data/splits/`. V2 training contains 18,532 questions across 800 personas after the documented incomplete-reference exclusions. Validation/test membership is preserved. V1 and LoCoMo remain comparison datasets. Historical results retain the retrieval units and model identities recorded when they were generated; regenerate retrieval outputs for the current single-message format.

Datasets and processed reference labels are versioned; oversized dataset files use Git LFS. Trained models and generated ranker feature/embedding caches stay local and are excluded from Git.

## Setup

Use the existing Python 3.13 environment:

```powershell
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.lock.txt
python scripts/download_models.py --only embeddinggemma2
```

EmbeddingGemma uses the CUDA PyTorch build pinned in the lock file. Model revisions and asset SHA-256 hashes are pinned in `configs/assets.lock.json`.

## Build the complete 55-feature bundle

```powershell
python scripts/build_ranker_features.py
python scripts/ranker55.py --features data/ranker55/v2/train/features.npz --output models/ranker55.txt
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --diversity-weight 0.1
```

The single builder replaces the separate BM25 and embedding scripts. It reads original messages from `data/raw/` and active IDs from `data/splits/`, fits training corpus statistics, caches embeddings, and calls the unchanged `ranker55.feature_matrix()` over each question's full visible history. It does not train LightGBM or invoke Qwen. The second command trains the existing binary reference-only ranker; the third selects a labeling pilot using its saved weights.

The default exports **all active V2 training questions** to `data/ranker55/v2/train/`. `--limit-questions 100` limits exported queries, while statistics still use the full active training corpus. `--question-ids PATH` selects an explicit output cohort. For a fold, `--fit-question-ids PATH` restricts fitting statistics to a JSON list of active training IDs; it is separate from the output cohort. Corpus statistics must match those used to train the saved model; new statistics do not reproduce a historical checkpoint's inputs automatically.

To build held-out or comparison features, reuse frozen training statistics:

```powershell
python scripts/build_ranker_features.py --splits val test
python scripts/build_ranker_features.py --datasets v1 locomo
```

The builder automatically reuses `data/ranker55/statistics.json`, falling back to `data/ranker55/v2/train/statistics.json`. For a custom output folder, it checks that folder's shared file and training-bundle copy first, then the project defaults. `--statistics PATH` overrides auto-detection. Missing or invalid explicit files fail instead of silently falling back. When no saved statistics exist, a V2 training build fits and publishes them; held-out/comparison-only builds explain that a training build or explicit file is needed. Use `--refit-statistics` to intentionally refit on V2 training data. `--fit-question-ids` also bypasses auto-reuse for fold-specific fitting. Held-out runs reject overlapping fitting personas and never fit IDF or question-frequency weights. Every bundle receives its own statistics copy and fingerprint. Statistics contain global training parent-message `n/df`, training-question `nq/qdf`, fitting IDs/personas, and source hashes. Shared histories contribute one DF vote per source message ID, not once per question; repeated occurrences with distinct IDs each count. QDF counts term presence per question.

### Bundle contents

| Artifact under `data/ranker55/{dataset}/{split}/` | Contents |
| --- | --- |
| `features.npz` | Float32 `X` with 55 ordered columns; `feature_names`, `offsets`, `question_ids`, `persona_ids`, row-aligned `message_ids`, `message_indices`, `content_token_lengths`, labels, and `metadata_json` |
| `statistics.json` | Frozen fitting statistics and provenance |
| `embeddings/memory_embeddings.npy` | Normalized 768-dimensional vectors for unique complete message texts |
| `embeddings/question_embeddings.npy` | Vectors for unique question texts |
| `embeddings/chunk_embeddings.npy` | Vectors for unique dense chunk texts |
| `embeddings/*_input_token_lengths.npy` | Prefix/special-token-inclusive lengths for memory, question and chunk inputs |
| `embeddings/memory_content_token_lengths.npy` | Bare message lengths without prefix or special tokens, used by `log_content_tokens` |
| `embeddings/messages.jsonl`, `questions.jsonl` | Original IDs/text, roles/order, and embedding-row mappings |
| `embeddings/unique_texts.jsonl`, `chunk_texts.jsonl` | Text associated with each embedding row |
| `embeddings/message_chunk_offsets.npy`, `message_chunk_indices.npy` | Ragged chunk mapping per unique whole-message embedding row; short messages have no auxiliary chunks |
| `embeddings/*_done.npy`, `*_embedding_metadata.json` | Resumption masks and validated embedding recipes |
| `bm25/`, `cosine/` | Complete score rankings extracted from the feature matrix, plus provenance reports |
| `reference_labels/` | Exact grade-2 reference rows for V2 training only |
| `report.json` | Feature/statistics identity, source hashes, output hashes, and counts |

The training NPZ contains three distinct label arrays: `reference_labels` uses **2/0** for annotation membership; `judged_labels` uses **2/-1**, preserving non-references as unjudged; `y` uses **1/0** solely to reproduce the existing binary reference-only ranker. V2 validation/test and comparison exports have **-1** labels throughout, because those runs do not load reference annotations. They cannot accidentally be passed as valid binary training data. Future Qwen judgments must not be replaced by baseline `y` or zero-filled unknowns.

### Embeddings and lexical features

The pinned EmbeddingGemma 2 model uses 768-dimensional masked mean pooling (including prefix tokens), followed by L2 normalization. Memory and chunk inputs use `title: none | text: `; questions use `task: search result | query: `. Identical text within each channel is embedded once. Whole-message text and IDs are preserved, including source gaps and V1 negative cutoffs. Answers and choices never enter the encoder or feature matrix.

Messages over **64 lexical words** receive auxiliary **48-word windows with stride 24** using the existing ranker recipe; short messages reuse whole-message cosine. Original whole-message inputs still must fit the **1,200-token** encoder cap. Oversized inputs fail before encoding; chunks never silently substitute for a truncated whole message.

Whole-message BM25 uses `k1=2, b=0` and visible-pool IDF. Lexical chunks use `k1=1.5, b=0`, the conversational stoplist, and frozen global training parent-message DF. Question-frequency weighting uses frozen training QDF. All formulas, feature order, neighbors, dense-chunk aggregations and relative-position semantics remain in `scripts/ranker55.py`.

Default embedding batches are 16 inputs with a 4,800 padded-token budget. `--device cpu` is supported. CUDA OOM reduces batch size. Completed caches are validated and reused; interrupted caches resume from completion masks. `--rebuild-cache` overwrites stale caches. Completed NPZ/score exports overwrite previous files after feature construction succeeds; feature matrices use an on-disk working array to limit RAM usage. Custom output roots stay under `data/ranker55/`, separate from raw data and split IDs. Existing historical caches elsewhere are left intact.

## Direct reference labels

```powershell
python scripts/label_reference_messages.py --splits train
```

The labeler reads the reference snippet from each active V2 question's original CSV. A message receives **2** exactly when its original `(role, content)` matches a reference for that question, otherwise **0**. All matching occurrences are positive. An adjacent assistant reply receives its own independent label. Zero denotes annotation non-membership and does not prove semantic irrelevance. New rows and reports declare `schema_version: 3`; the diagnostics also support legacy schema 2 with binary reference labels.

Outputs are `data/reference_labels/v2/{split}.jsonl` and `report.json`. Candidate entries contain `message_id`, `message_index`, `role`, and `label`. Unmatched references raise an error, and empty references must exist in the raw history. All requested splits are staged before completed files are replaced. Validation/test labeling is supported, but those splits still contain incomplete references; it does not silently change membership.

## Select messages for Qwen labeling

```powershell
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --limit-questions 100
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --limit-questions 0
```

The first command selects a deterministic pilot of up to 100 cached training questions; the second selects all questions present in the cache. The default saved LightGBM text model is `models/ranker55.txt`; use `--model PATH` for another filename or the original 54-feature model. The script loads existing weights, without retraining or invoking Qwen. It scores the full visible history and selects **40 distinct non-reference messages per question**: 20 ranker, 5 question-weighted BM25, 5 dense-chunk cosine, 5 neighbors, and 5 random remaining messages. Original references (including repeated matching occurrences) receive label **2**, outside the budget. Fewer than 40 eligible messages is an error.

The external NPZ must contain the following arrays, with `allow_pickle=False` compatibility:

| Array | Meaning |
| --- | --- |
| `X` | Full question-grouped feature matrix, computed with the saved model's recipe and training statistics |
| `feature_names` | Unicode string array matching the exact ordered 54 or 55 names in `scripts/ranker55.py` |
| `offsets` | Integer group boundaries: starts at 0, ends at the number of rows in `X` |
| `question_ids` | One Unicode example ID per group; must belong to the current V2 training split |
| `message_ids` | One Unicode message ID per row; each group must exactly match the visible history in source order |

The unified builder produces this cache directly with explicit row identities and source hashes. The selector checks identities and model feature order, but the cache producer remains responsible for using the fitted model's statistics. Any cached `y` is ignored. A 54-feature model can use the matching first 54 columns of a 55-feature cache. No feature values, model parameters, or inference pool limits are changed. The selector reuses the builder's cached arrays. With diversity enabled, it finds `embeddings/` beside the NPZ automatically; `--embedding-root` can override that path.

Duplicate/reference selections are replaced by the next eligible result from the same channel. Neighbors are source-index distance 1 or 2 from the first 30 selected non-reference messages; source gaps are not bridged. If fewer than 5 remain, the ranker fills that allocation and records `source: ranker_refill`. Ties use chronological order, and random selection is seeded per question so changing pilot size does not change that question's candidates.

Optional soft diversity within the lexical, dense and neighbor allocations uses the existing complete-message vectors, without a hard similarity cutoff:

```powershell
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --diversity-weight 0.1
```

The preference considers the next 20 eligible results per channel, subtracting `weight * max(0, cosine_to_already_selected)` from normalized source-rank quality. It never reduces the 40-message budget. The top 20 ranker messages and uniform random sample are unaffected.

For persona-separated candidate selection, load saved fold models instead of the single model:

```powershell
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --fold-manifest models/ranker55_folds.json
```

The JSON manifest has a `models` list. Each entry contains `path` (relative to the manifest), `sha256`, `training_persona_ids`, and `held_out_persona_ids`, using IDs such as `v2:7`. Every selected question must be routed to exactly one model whose declared training personas exclude its persona. Hash mismatches, overlapping training/held personas, duplicate held-out assignments and missing routes fail. Use genuinely persona-held-out models for the supervision experiment; the script does not create them. A single model's training overlap cannot be inferred from its weights and is explicitly recorded as `single_model_training_overlap_unknown`, not out-of-fold.

Outputs overwrite `data/label_candidates/v2/train.jsonl` and `report.json` after successful selection. Each question has `references` (label 2), `candidates` (`label: null`, `status: pending_qwen`, source/rank/score metadata), and `unjudged_message_ids` for the unselected remainder (`unselected_label: null`). The report records source counts, fallback counts, model/input hashes and selection provenance. No non-reference receives an invented negative label. Later Qwen labeling can assign 0/1 only to evaluated candidates; unselected messages must remain unjudged. Retrieval features and output contain no answer keys or choices. Full-history retrieval remains unchanged at inference.

For a one-question runtime/sample check:

```powershell
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --limit-questions 1 --diversity-weight 0.1 --output-root data/label_candidates/v2/one_question
```

The report's `runtime_seconds` separates dataset loading, feature loading/validation, reference loading, model loading, embedding-cache loading, prediction and selection. Its total is measured before publishing the report; command startup/shutdown is outside that total. `train.jsonl` contains the selected question and its actual reference/candidate texts. This runtime check does not measure held-out retrieval accuracy or Qwen answer utility.

## Training retrieval diagnostics

After a full V2 training feature build (without a question limit):

```powershell
python scripts/diagnose_retrieval.py --rrf-constant 60 --k 5 10 20 30 50
```

Diagnostics read the new bundle at `data/ranker55/v2/train/` (`--feature-root` overrides it). Its BM25 channel is the retained `bm25_b0` feature (`k1=2, b=0`), so it differs from historical `k1=1.5, b=0.75` baseline reports. RRF uses `1/(60 + bm25_rank) + 1/(60 + cosine_rank)`. Labels identify positives after fusion. Input reports must declare `memory_unit: single_message`; stale formats are rejected. RRF scores go to `data/rrf/v2/scores/train.jsonl`. New metrics, rank distributions, CSV/PNG histograms and reports go to `results/selector_diagnostics_single_message/v2/train/`, keeping historical reports intact.

Recall@k averages the fraction of each question's positive messages retrieved. Hit@k averages whether at least one is retrieved, and full-reference coverage@k averages whether all are retrieved. Rank medians and 90th percentiles pool positive message occurrences. Score ties use chronological source-message order.

## Selected ranker

The selected 55-feature ranker uses LightGBM `LGBMRanker` with LambdaRank and binary labels (`label_gain=[0, 1]`). Training is configured for LightGBM's CUDA backend; the installed LightGBM build must include CUDA support and will fail explicitly otherwise rather than silently falling back to CPU. When reproducing the existing binary reference-only model, map reference grade 2 to binary 1 explicitly; `scripts/ranker55.py` still expects binary training labels. Candidate exports with unknown labels are not binary training data and must not be zero-filled. Use one candidate row per original message, group rows by question, and retain persona-disjoint splits. Role, complete-message length, position, retrieval ranks, z-scores and gaps are candidate features. Gold question type and cross-question label aggregates stay outside the primary feature set. Recompute retrieval baselines and memory-reuse statistics for this granularity before comparing new results with historical reports.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Tests use tiny synthetic histories and a fake encoder. They cover complete-message preservation, V1 visibility, stable IDs, LoCoMo metadata, exact independent labels, answer isolation, cache limits/resumption and BM25/cosine/RRF alignment. They do not load pretrained weights or start Qwen. Direct-loading regression tests run without any prepared-message directory and verify that missing split files fail before model loading.

## Qwen evaluation

To fit a ranker now using the existing binary reference labels and tune on **V2 validation**:

```powershell
python scripts/tune_ranker55.py --build-val --wait-for-training 14400
```

This command waits for an existing training feature build without restarting it. If the validation NPZ is missing, it builds V2 val with the training bundle's frozen statistics. It then evaluates 27 settings: `min_child_samples` in `{50,100,200}`, paired `num_leaves/max_depth` in `{15/4,31/5,63/6}`, and `reg_lambda` in `{1,5,10}`. The default `100 / 31 / 5 / 5` is fitted first and saved immediately. Features, binary label gains, learning rate, and 300 boosting rounds remain fixed. All fitting uses training rows; validation selects settings and test is not loaded.

Override the grid with `--min-child-samples 50 100`, `--tree-shapes 15:4 31:5`, or `--reg-lambda 1 5`. For one fit, `scripts/ranker55.py` accepts scalar `--min-child-samples`, `--num-leaves`, `--max-depth`, and `--reg-lambda` options.

Selection uses non-forget **type-macro Hit@10** over all validation questions. A setting replaces the default only if it improves by strictly more than one percentage point and does not reduce type-macro NDCG@10 (`--adoption-threshold 0.01`). Ties favor earlier source messages. Reports include per-type and question-weighted results, incremental Hit@10 and question-level tail rescue over cosine, plus complete-reference-only results. Missing annotated reference messages remain in the NDCG ideal denominator and cannot create hits. These validation-selected results are development measurements; final test evaluation remains separate.

The adopted model overwrites `models/ranker55.txt`, ready for selector-mode Qwen evaluation. Trial models stay under `models/ranker55_trials/`; `report.json`, `trials.csv`, `per_question.jsonl`, and `REPORT.md` go under `results/ranker55_tuning/v2/val/`. The tuner maps the builder's uncompressed NPZ arrays and reuses a binned LightGBM training dataset across configurations. It never uses Qwen-generated judgments; adding those still requires a separate labeling and graded-training pass.

The local evaluator uses **Unsloth Qwen3.5-4B Q4_K_M GGUF** through the native llama.cpp CUDA server. It is the sole supported Qwen model. All layers are requested on the GPU, FlashAttention and prompt caching are enabled, thinking and context shifting are disabled, and requests are serial. Original raw datasets and split IDs are loaded directly.

| Dataset | Evaluation questions | Supported `--mode` values |
| --- | --- | --- |
| V2 | Preserved test IDs | `full_history`, `no_memory`, `reference_only`, `selector` |
| V1 | Full comparison dataset | `full_history`, `selector` |
| LoCoMo | Full comparison dataset | `full_history`, `selector` |

```powershell
python scripts/evaluate_qwen.py --dataset v2 --dry-run
python scripts/evaluate_qwen.py --mode no_memory --dry-run
python scripts/evaluate_qwen.py --dataset v2 --mode reference_only --dry-run
python scripts/evaluate_qwen.py --dataset v1 --dry-run
python scripts/evaluate_qwen.py --dataset locomo --dry-run
python scripts/evaluate_qwen.py --dataset v2 --mode selector --dry-run
python scripts/evaluate_qwen.py --dataset v1 --mode selector --dry-run
python scripts/evaluate_qwen.py --dataset locomo --mode selector --dry-run
```

There are no `--split` or `--subset` arguments. V2 always uses the preserved test IDs; their existing compact/full provenance remains report metadata. V1 and LoCoMo use all comparison IDs. `--limit` remains available for small smoke tests. Automatic resume follows the same dataset rules.

`reference_only` reads only the original `related_conversation_snippet` annotation, corresponding to reference label **2**. It never reads Qwen-generated labels or adds label **1** messages. This is an oracle baseline, with reference messages preserved verbatim in original snippet order. If an annotated message is absent from the visible history, it is still included and the discrepancy is counted in predictions and the report.

`selector` loads `models/ranker55.txt` and `data/ranker55/{dataset}/{split}/features.npz` by default (`split=test` for V2, `comparison` otherwise). Override those paths with `--selector-model` and `--selector-features`. The default is a **2,048 Qwen-token memory budget** on all three datasets. Use `--memory-tokens 1024` or `--memory-tokens 4096` to change it (`--token-budget` is an alias). The selector scores the complete visible message pool, considers messages in descending score order, and skips messages that do not fit so smaller lower-ranked messages can still be included. Ties favor earlier source indices. Accepted messages stay whole and are presented chronologically with their original roles and LoCoMo session/timestamp metadata.

LightGBM training, parameter tuning, and saved selector prediction explicitly use the CPU (`device_type=cpu`, four threads). Qwen inference continues on the GPU. The local packing tokenizer loads vocabulary only on the CPU.

The budget uses the active Qwen native tokenizer: complete templated prompt tokens minus the instruction/question/options-only prompt tokens. It includes history headings, role/index labels, separators, session headers, and chat framing. The complete prompt must also fit the actual server context with one answer token reserved. No message is truncated; if none fits, the prompt contains only instructions and the question/options. Selected counts, actual token usage, skipped-message counts, and packing time are recorded per question. There is no default message-count cap. For a fixed-count comparison, explicitly use `--top-k 10`; it is mutually exclusive with `--memory-tokens`.

Token-budget packing formats source message blocks once per conversation boundary and loads the same GGUF vocabulary locally through the pinned `llama.dll`, without loading another set of model weights or allocating a GPU inference context. It renders the GGUF's Jinja template locally and caches exact counts at Qwen35 pre-tokenizer boundaries, preserving whitespace and separators. Startup probes compare local prompt text, token counts, and token IDs with the active server. Each final selected prompt is checked again with `/apply-template` and `/tokenize` before scoring; a mismatch stops the run. Packing therefore uses two HTTP requests per question instead of two per candidate. The final full prompt and question-only count are reused during scoring. The native binding checks the DLL hash and supports the currently pinned b11669 ABI; a runtime upgrade requires updating that binding. Full-history, reference-only, and explicit top-k modes retain their native HTTP rendering path.

For token-budget selector evaluation, total context capacity is **memory budget + 1,000 tokens** on all datasets: 1,024 → 2,024; 2,048 → 3,048; 4,096 → 5,096. The extra capacity covers instructions, question/options, their formatting, and the one-token answer. `selector_context_overhead_tokens` in `configs/qwen_server.json` stores this 1,000-token allowance; the old fixed `selector_context_tokens` setting has been removed. The native server supports `--mode selector --memory-tokens N` with the same default calculation. If the non-memory portion needs more space, packing reduces the usable memory budget; complete prompts are never truncated. Explicit `--top-k` evaluation retains the dataset's full-history context capacity because it has no memory-token budget.

The 40-message labeling budget does not apply here; annotations, cached labels and correct answers do not participate in selection. Model/feature hashes and selection settings, including the token budget, are checked when resuming. Token-budget results and explicit top-k results use separate default output directories.

Selector evaluation needs a completed saved model and a matching feature bundle built with the frozen training statistics. Once the training build is finished, prepare these inputs separately:

```powershell
python scripts/build_ranker_features.py --datasets v2 --splits test
python scripts/build_ranker_features.py --datasets v1 locomo
```

The evaluator never generates embeddings or fits a model. Its dry run validates selector files and performs one CPU ranking, without starting Qwen. Token packing is explicitly reported as pending during a dry run because exact counts require the active native Qwen tokenizer. Missing or mismatched inputs fail before Qwen starts.

`scripts/qwen_llamacpp_server.py` owns the native process and API. Its `LlamaCppServer` interface starts `models/llama.cpp/llama-server.exe`, checks readiness, handles tokenization/scoring requests and retries, samples resources, and shuts down its owned process. `scripts/evaluate_qwen.py` loads data, prepares messages, calls that interface, grades returned scores, and saves reports. Shared launch settings come from `configs/qwen_server.json`. `--dry-run` inspects data/settings without loading Qwen weights. The server module uses a direct `requests.Session` against llama.cpp native endpoints: `/completion`, `/tokenize`, `/apply-template`, `/health`, and `/props`. Raw token IDs and pre-sampling log-probabilities come directly from `/completion`; prompt caching is enabled explicitly. Full histories are checked by the native tokenizer before scoring and never truncated. Dataset caps remain 41,216 (V2), 33,280 (V1), and 32,256 (LoCoMo); no-memory context is 4,096. These originate from the prior exact tokenizer audit and require a complete native-runtime audit before declaring all prompts verified.

All previous `results/` files retain their original model identities. New native outputs use `qwen35_4b_q4_k_m_*_raw` directories. Automatic resume excludes other backends, and explicit reuse of an incompatible run is rejected. Model/runtime revisions, SHA-256 hashes and launch settings are recorded in the asset lock and new protocols. Token-cost features must identify the tokenizer/runtime that produced them.

To start the server directly from the project root when ready:

```powershell
.venv/Scripts/python.exe scripts/qwen_llamacpp_server.py
```

Non-thinking text generation uses `temperature=1.0`, `top_p=1.0`, `top_k=20`, `min_p=0.0`, `presence_penalty=2.0`, and `repetition_penalty=1.0`, configured under `generation` in `configs/qwen_server.json`. Sampling is enabled for generated text. Answer-label evaluation uses raw pre-sampling probabilities.

Use `--dataset v2`, `--dataset v1`, or `--dataset locomo` to select the measured full-history limit, and `--mode no_memory` for the shared no-memory context setting. `--print-settings` prints the resolved command/settings without loading weights or opening a port. The server module manages the evaluation server lifecycle automatically when called by the evaluator.

The historical tokenizer audit covers all 23,351 active questions and supplies the initial context caps. Every prompt is counted again by the active native GGUF tokenizer before inference; complete prompt plus answer must fit without truncation. Historical counts and provenance remain in `data/qwen_prompt_lengths/`.

For a direct native request to a running llama.cpp server:

```python
import requests

with requests.Session() as session:
    session.trust_env = False
    prompt = session.post(
        "http://127.0.0.1:8080/apply-template",
        json={"messages": [{"role": "user", "content": "Hello"}]}, timeout=30,
    )
    prompt.raise_for_status()
    response = session.post(
        "http://127.0.0.1:8080/completion",
        json={"prompt": prompt.json()["prompt"], "n_predict": 64, "cache_prompt": True},
        timeout=120,
    )
    response.raise_for_status()
    print(response.json()["content"])
```

Evaluation requests raw token-ID log-probabilities with `post_sampling_probs=false`. The same native server supplies text generation and tokenization.

The native CUDA runtime handles prompt batching, FlashAttention and prefix caching. Its fit adjustment and context shifting are disabled, so context settings remain explicit. llama.cpp is the sole Qwen inference server; the obsolete Transformers server and its configuration have been removed.

Restore and verify the new assets when needed:

```powershell
python scripts/download_models.py --only qwen llama_cpp
python scripts/download_models.py --only qwen llama_cpp --verify-only
```

Debugging checks use synthetic outputs and do not launch Qwen:

```powershell
python -m unittest discover -s tests
```

