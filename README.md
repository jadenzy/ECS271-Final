# Single-message memory retrieval

Each retrieval memory is **one complete original user or assistant message**. Its text is preserved exactly, with the role, source index and original message object stored as metadata. Retrieval messages remain whole and unchanged. Auxiliary chunks supply features while the retrieval unit stays the original message. System/persona messages and empty/non-text turns are excluded from retrieval, and the full-history baseline leaves the system/persona messages out as well, so every evaluation mode reads only conversation messages.

Original sources remain under `data/raw/`, and active IDs under `data/splits/`. V2 training contains 18,532 questions across 800 personas after the documented incomplete-reference exclusions. Validation (909 questions, 40 personas) and test (1,119 questions, 50 personas) exclude the `sensitive_info` type, whose reference snippets are absent from the histories; see [Validation and test exclusions](#validation-and-test-exclusions). V1 and LoCoMo remain comparison datasets. Historical results retain the retrieval units and model identities recorded when they were generated; regenerate retrieval outputs for the current single-message format.

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
python scripts/ranker55.py --features data/ranker55/v2/train/features.npz --output models/subtlex_mix/alpha_0.5/ranker55.txt
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --diversity-weight 0.1
```

The single builder replaces the separate BM25 and embedding scripts. It reads original messages from `data/raw/` and active IDs from `data/splits/`, loads the pinned SUBTLEX prior, caches embeddings, and calls `ranker55.feature_matrix()` over each question's full visible history. It does not train LightGBM or invoke Qwen. The second command trains the existing binary reference-only ranker; the third selects a labeling pilot using its saved weights.

The default exports **all active V2 training questions** using the SUBTLEX-US/visible-history mixture with `alpha=0.5`. Outputs go to `data/ranker55/v2/train/`. `--alpha` sets a different mixture weight; `--limit-questions` and `--question-ids PATH` select an output cohort. Each question's local frequencies use its own complete visible history.

Validation/test and comparison builds use the same pinned prior and alpha, with no V2 statistics-fitting prerequisite:

```powershell
python scripts/build_ranker_features.py --splits val test
python scripts/build_ranker_features.py --datasets v1 locomo
```

Each bundle and its parent output folder contain a `statistics.json` recipe manifest with alpha, normalization, and external source hashes. The builder does not fit global message DF or question-frequency statistics. The former `--statistics`, `--refit-statistics`, `--fit-question-ids`, and `--lexical-recipe` options have been removed. Older feature matrices and weights need rebuilding/retraining; their embedding caches can still be read using `--embedding-cache-root` once complete.

### Bundle contents

| Artifact under `data/ranker55/{dataset}/{split}/` | Contents |
| --- | --- |
| `features.npz` | Float32 `X` with 55 ordered columns; `feature_names`, `offsets`, `question_ids`, `persona_ids`, row-aligned `message_ids`, `message_indices`, `content_token_lengths`, labels, and `metadata_json` |
| `statistics.json` | Pinned SUBTLEX recipe, alpha, normalization, and source hashes |
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

Whole-message BM25 uses `k1=2, b=0` and visible-pool IDF. Lexical chunks use `k1=1.5, b=0`, the conversational stoplist, and visible parent-message DF. Weighted lexical channels use the SUBTLEX/history mixture described below. All feature order, neighbors, dense-chunk aggregations and relative-position semantics remain in `scripts/ranker55.py`.

Default embedding batches are 16 inputs with a 4,800 padded-token budget. `--device cpu` is supported. CUDA OOM reduces batch size. Completed caches are validated and reused; interrupted caches resume from completion masks. `--rebuild-cache` overwrites stale caches. Completed NPZ/score exports overwrite previous files after feature construction succeeds; feature matrices use an on-disk working array to limit RAM usage. Custom output roots stay under `data/ranker55/`, separate from raw data and split IDs. Existing historical caches elsewhere are left intact.

Use `--feature-workers 2` to calculate two question feature groups concurrently on CPU after embeddings finish. Workers read the embedding caches through read-only memory maps; the parent writes results in question order with at most twice the worker count queued. The default is one worker. Restarting reuses completed embeddings but recalculates unfinished feature exports from the beginning.

## SUBTLEX-US mixture

The `subtlex_mix` lexical recipe uses the processed prior at
`data/external/subtlex_us/word_frequencies.json`, pinned by
`configs/subtlex_us.lock.json`. It mixes external token probability with token
frequency in the current visible history: `p = alpha * p_external + (1-alpha) * p_history`.
Only complete source messages contribute local counts; chunks and other queries
do not. Separate source occurrences with identical text still count separately.
The external distribution assigns a 0.5 pseudo-count to each locally observed
unknown token and renormalizes over the union vocabulary. Both sources use NFKC,
casefold, and `\w+` tokenization, including the same contraction splitting.

The weight is `clip(-ln(max(p, 1e-9))/ln(1e9), 0.1, 1)`. Weighted lexical
channels replace training-question-frequency multipliers with this weight.
Chunk BM25 uses visible parent-message DF. Whole-message BM25, embedding-based
features, roles, lengths, and position retain their definitions. This recipe
needs no V2 training corpus statistics, although training a ranker still needs
training labels.

After matching bundles have completed, recalculate features
using their existing caches read-only (no encoder or GPU model is loaded):

```powershell
python scripts/build_ranker_features.py --alpha 0.5 --embedding-cache-root data/ranker55 --datasets v2 --feature-workers 4
python scripts/build_ranker_features.py --alpha 0.5 --embedding-cache-root data/ranker55 --datasets v2 --splits val test --feature-workers 4
python scripts/build_ranker_features.py --alpha 0.5 --embedding-cache-root data/ranker55 --datasets v1 locomo --feature-workers 4
```

Alpha 0.5 was checked on V2 validation. Alpha changes 18 columns: the 10 `qweight_*` features and the neighbor weighted-BM25 value and rank. Each alpha was refit with the selected setting (100 / 63 / 6 / 10, forget weight 0.5):

| Alpha | 0 (history only) | 0.1 | 0.25 | **0.5** | 0.75 | 0.9 | 1 (SUBTLEX only) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Non-forget type-macro Hit@10 | 65.63% | 65.63% | 65.79% | **67.20%** | 65.81% | 66.64% | 66.17% |
| NDCG@10 | 37.62% | 37.27% | 37.16% | **38.00%** | 37.52% | 37.86% | 37.48% |

No alpha beats 0.5. Alphas 0, 0.1, 0.25 and 0.75 are lower by about 1.4–1.6 points, and their 95% persona-bootstrap intervals exclude zero. The tree setting was tuned at 0.5, which slightly favors it. Details are in `results/ranker55_alpha/v2/val/report.json`.
Outputs are under `data/ranker55/{dataset}/{split}/` for every alpha; changing alpha replaces the feature exports after successful construction. Use `--output-root` to keep an additional experiment separately. The manifest and model sidecar retain alpha;
`statistics.json` records the external recipe instead of V2-derived counts.
Cache reuse requires the same complete input cohort/order, source hashes,
embedding recipe, and completed masks as the baseline export. A limited-question
build needs a correspondingly matching cache. Reuse never repairs partial caches
and cannot be combined with `--rebuild-cache`. Without `--embedding-cache-root`,
the normal encoder path creates or resumes caches inside each dataset/split output directory.

Train new weights from the new features, using a separate destination:

```powershell
python scripts/ranker55.py --features data/ranker55/v2/train/features.npz --output models/subtlex_mix/alpha_0.5/ranker55.txt
```

The model receives a `.txt.recipe.json` sidecar binding its hash to alpha, source
hashes, and lexical semantics. Candidate selection and selector evaluation reject
incompatible recipes even when all 55 feature names match. Keep this sidecar with
the model. Models without a matching sidecar and bundles without SUBTLEX recipe
metadata are rejected. Candidate diversity reads the external cache path recorded by the
builder; `--embedding-root` can override that location after moving artifacts.

## Direct reference labels

```powershell
python scripts/label_reference_messages.py --splits train
```

The labeler reads the reference snippet from each active V2 question's original CSV. A message receives **2** exactly when its original `(role, content)` matches a reference for that question, otherwise **0**. All matching occurrences are positive. An adjacent assistant reply receives its own independent label. Zero denotes annotation non-membership and does not prove semantic irrelevance. New rows and reports declare `schema_version: 3`; the diagnostics also support legacy schema 2 with binary reference labels.

Outputs are `data/reference_labels/v2/{split}.jsonl` and `report.json`. Candidate entries contain `message_id`, `message_index`, `role`, and `label`. Unmatched references raise an error, and empty references must exist in the raw history. All requested splits are staged before completed files are replaced. Validation/test labeling is supported, but those splits still contain a few incomplete references (4 validation and 7 test `ask_to_forget` questions each lack one assistant acknowledgment); labeling does not silently change membership.

## Validation and test exclusions

`sensitive_info` questions are removed from V2 validation and test. Their annotated reference snippets appear nowhere in the linked 32K histories (0 of 222 validation and 0 of 313 test reference messages), so no retriever can find them and every ranker scored 0% Hit@10 on the type. Training already excluded it (`data/splits/v2/training_exclusions.json`).

```powershell
python scripts/exclude_eval_questions.py            # dry run: prints what would change
python scripts/exclude_eval_questions.py --apply
```

| Split | Questions before | Removed | Questions after | Personas |
| --- | ---: | ---: | ---: | ---: |
| Validation | 1,002 | 93 | 909 | 40 |
| Test | 1,242 | 123 | 1,119 | 50 |

The script rewrites `data/splits/v2/{val,test}_ids.json`, updates `persona_split.json`, and lists every removed ID in `data/splits/v2/evaluation_exclusions.json`. It filters the matching feature bundles under `data/ranker55/subtlex/v2/` instead of rebuilding them: a question's features depend only on its own history, so the retained rows are copied unchanged and verified equal, and the previous bundle stays beside the new one as `features_before_exclusion.npz`. The embedding caches under `data/ranker55/v2/{val,test}/` still describe the old splits, so rebuilding those bundles from scratch needs the embeddings regenerated. Four validation and seven test `ask_to_forget` questions still lack one of their four references (the assistant's acknowledgment) and stay in.

Effect on the numbers in this README:

- **Validation Hit@10 and NDCG@10** are non-forget type-macro averages, now over five question types instead of six. The removed type contributed exactly 0 for every ranker, so every earlier average, difference and interval is multiplied by 6/5; the order of models and the significance of each comparison are unchanged. Reports under `results/ranker55_*/v2/val/` were written before the removal and keep the six-type values.
- **V2 test accuracy** is recomputed from the saved predictions on the 1,119 active questions (`python scripts/compare_selector_runs.py --active-only`). The run folders under `results/` keep all 1,242 predictions and their original summaries.

`scripts/reference_hit_check.py --split val` (or `test`) measures the reference Hit@10 of saved rankers on either split, on the CPU; test references are read for measurement only. On the current splits, with feature bundles rebuilt on an RTX 3060 laptop whose embeddings differ slightly from the original build (validation 66.91% here against 67.20% in the tables below):

| Ranker | Validation Hit@10 | Validation NDCG@10 | Test Hit@10 | Test NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| **References only (current model)** | **66.91%** | **37.89%** | **68.61%** | **39.19%** |
| Qwen grades, best of 27 settings | 63.77% | 34.53% | 66.69% | 36.57% |

Per-type results are in `results/reference_hit/v2/{val,test}/report.json`; `report_before_exclusion.json` beside each holds the same measurement on the old splits.

## Select messages for Qwen labeling

```powershell
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --limit-questions 100
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --limit-questions 0
```

The first command selects a deterministic pilot of up to 100 cached training questions; the second selects all questions present in the cache. The default saved LightGBM text model is `models/subtlex_mix/alpha_0.5/ranker55.txt`; use `--model PATH` for another filename or the original 54-feature model. The script loads existing weights, without retraining or invoking Qwen. It scores the full visible history and selects **20 distinct non-reference messages per question**: 12 ranker, 3 question-weighted BM25, 3 dense-chunk cosine, and 2 random remaining messages. Original references (including repeated matching occurrences) receive label **2**, outside the budget. Fewer than 20 eligible messages is an error.

The external NPZ must contain the following arrays, with `allow_pickle=False` compatibility:

| Array | Meaning |
| --- | --- |
| `X` | Full question-grouped feature matrix, computed with the saved model's recipe and SUBTLEX recipe parameters |
| `feature_names` | Unicode string array matching the exact ordered 54 or 55 names in `scripts/ranker55.py` |
| `offsets` | Integer group boundaries: starts at 0, ends at the number of rows in `X` |
| `question_ids` | One Unicode example ID per group; must belong to the current V2 training split |
| `message_ids` | One Unicode message ID per row; each group must exactly match the visible history in source order |

The unified builder produces this cache directly with explicit row identities and source hashes. The selector checks identities and model feature order, but the cache producer remains responsible for using the saved model's pinned SUBTLEX recipe and alpha. Any cached `y` is ignored. A 54-feature model can use the matching first 54 columns of a 55-feature cache. No feature values, model parameters, or inference pool limits are changed. The selector reuses the builder's cached arrays. With diversity enabled, it uses the embedding-cache path recorded in the bundle; `--embedding-root` can override that path.

Duplicate/reference selections are replaced by the next eligible result from the same channel. Ties use chronological order, and random selection is seeded per question so changing pilot size does not change that question's candidates.

Optional soft diversity within the lexical and dense allocations uses the existing complete-message vectors, without a hard similarity cutoff:

```powershell
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --diversity-weight 0.1
```

The preference considers the next 12 eligible results per channel, subtracting `weight * max(0, cosine_to_already_selected)` from normalized source-rank quality. It never reduces the 20-message budget. The top 12 ranker messages and uniform random sample are unaffected.

For persona-separated candidate selection, load saved fold models instead of the single model:

```powershell
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --fold-manifest models/ranker55_folds.json
```

The JSON manifest has a `models` list. Each entry contains `path` (relative to the manifest), `sha256`, `training_persona_ids`, and `held_out_persona_ids`, using IDs such as `v2:7`. Every selected question must be routed to exactly one model whose declared training personas exclude its persona. Hash mismatches, overlapping training/held personas, duplicate held-out assignments and missing routes fail. Use genuinely persona-held-out models for the supervision experiment; the script does not create them. A single model's training overlap cannot be inferred from its weights and is explicitly recorded as `single_model_training_overlap_unknown`, not out-of-fold.

Outputs overwrite `data/label_candidates/v2/train.jsonl` and `report.json` after successful selection. Each question has `references` (label 2), `candidates` (`label: null`, `status: pending_qwen`, source/rank/score metadata), and `unjudged_message_ids` for the unselected remainder (`unselected_label: null`). The report records source counts, model/input hashes and selection provenance. No non-reference receives an invented negative label. Later Qwen labeling can assign 0/1 only to evaluated candidates; unselected messages must remain unjudged. Retrieval features and output contain no answer keys or choices. Full-history retrieval remains unchanged at inference.

### Qwen judging and graded training data

```powershell
python scripts/label_reference_messages.py --judge
python scripts/build_graded_training.py --all-rows
python scripts/tune_ranker55.py --train-features data/ranker55/v2/train_graded_all/features.npz --output models/qwen_graded_all/ranker55.txt --output-root results/ranker55_tuning_graded_all/v2/val
```

The judge scores each pending candidate in `data/label_candidates/v2/train.jsonl` with the frozen Qwen (question plus one message; never references, answers or options) and writes `data/qwen_labels/v2/train.jsonl`: per question, every visible message is labeled **2** (original reference), **1**/**0** (Qwen-judged useful / not useful candidate) or **-1** (unjudged). Completed questions are durable and a restart resumes after the last complete line; the input hash, model, judge prompt fingerprint and candidate plan must match. `report.json` is refreshed every 20 questions.

`--candidate-plan` judges only the first N picks of each allocation, best-ranked first, and leaves the remaining candidates unjudged (-1); without it every candidate is judged. Files from the current selector need no plan. The existing `data/label_candidates/v2/train.jsonl` predates the 20-message selector, so its run uses `--candidate-plan ranker=12,qweight_bm25=3,dense_chunk=3,random=2`. Each output folder holds a single plan.

It runs `--replicas 3` llama-server processes with `--parallel 4` slots each on ports 8080-8082 (about 12.5 GB VRAM). Each server scores its slots in one loop, so extra processes raise throughput where extra slots do not; the RAM prompt cache is disabled (`--cache-ram 0`) because judge prompts are never reused. On an RTX 5080 this is about 19.5 requests/s, roughly 53 questions/min with 20 candidates. Server and connection failures restart the servers (up to 5 times without progress; context goes to `judge-failures.log`, logs to `servers/N/`); a GPU memory error removes one replica, then halves slots. Batched GPU numerics flip about 0.5-0.8% of labels between different slot layouts; the raw a/b log-probabilities are saved with every judgment.

`build_graded_training.py` joins those labels to `data/ranker55/v2/train/features.npz` by row identity. By default it writes `data/ranker55/v2/train_graded/` with only judged rows (references, useful and rejected candidates), graded `y`, and rebuilt offsets; unjudged messages are dropped. `--all-rows` writes `data/ranker55/v2/train_graded_all/` with every row: unjudged messages get `y` 0, as in reference-only training, and keep -1 in `judged_labels`. It requires every training question to be judged; `--allow-partial` builds a judged subset that `ranker55.py` can fit but `tune_ranker55.py` rejects.

**Judged-rows-only training does not work.** The full 20-candidate run judged 370,640 candidates, 17,066 (4.6%) of them useful. The graded set has 427,709 rows (21–28 per question, against about 230 visible). Tuned on V2 validation over the same 27 settings, alongside a control with the same rows where only references are positive:

| Training data | Non-forget type-macro Hit@10 | NDCG@10 | `ask_to_forget` Hit@10 |
| --- | ---: | ---: | ---: |
| **All rows, references only (current model)** | **67.20%** | **38.00%** | **93.19%** |
| Judged rows, Qwen grades 2/1/0 | 35.17% | 15.62% | 65.45% |
| Judged rows, references only (control) | 31.29% | 13.50% | 54.97% |

The candidates were picked because they score high, so within the judged rows a high retrieval score no longer separates references. On BM25, the rejected candidates actually rank above the references (mean rank fraction 0.33 vs 0.41). The model also never sees the ordinary messages it must rank at inference. The Qwen labels still carry signal: graded beats the control at all 27 settings, by +3.9 points Hit@10 on average. Neither model is adopted. Details are in `results/ranker55_graded_comparison/v2/val/report.json`.

**All-rows graded training does not help either.** With `--all-rows` (4,273,942 rows: 57,069 references at 2, 17,066 Qwen-useful at 1, the rest 0):

| Training labels (all rows) | Non-forget type-macro Hit@10 | NDCG@10 | `ask_to_forget` Hit@10 | Change vs current (95% persona bootstrap) |
| --- | ---: | ---: | ---: | --- |
| **References only (current model)** | **67.20%** | **38.00%** | 93.19% | – |
| Qwen grades, same setting (100 / 63 / 6 / 10) | 63.36% | 34.49% | 94.76% | −3.84 (−6.10 to −1.72) |
| Qwen grades, best of 27 settings (100 / 63 / 6 / 1) | 64.19% | 34.49% | 95.29% | −3.01 (−5.12 to −0.97) |

Qwen's useful messages already rank above the references on every channel (mean cosine rank fraction 0.09 vs 0.21), so rewarding them pulls the ranker toward plain similarity and away from the references. The current reference-only model stays. Details are in `results/ranker55_graded_all_comparison/v2/val/report.json`.

Qwen answer accuracy on V2 test (1,119 questions, selector mode, 2,048 memory tokens) agrees:

| Ranker | Accuracy | Persona 95% CI | Output |
| --- | ---: | --- | --- |
| **References only (current model)** | **39.41%** | 36.66% to 42.07% | `results/qwen35_4bselector_raw/personamem_v2/32k/compact/test_memory_2048_references_only` |
| Qwen grades, best of 27 settings | 37.27% | 34.43% to 40.24% | `.../compact/test_memory_2048_qwen_graded_all` |

Paired by question, the Qwen-graded ranker is −2.14 points (persona bootstrap −3.61 to −0.61). It answers 34 questions the reference-only ranker misses and misses 58 that it answers. It is lower on every question type except `health_and_medical_conditions` (+1.5), most on `anti_stereotypical_pref` (−5.3). The two rankers share about two thirds of their selected messages (mean Jaccard 0.67). The run folders still hold all 1,242 predictions; these figures use the 1,119 active test questions.

On PersonaMem V1 (589 comparison questions, 20 personas, same settings), the order flips but the difference is not significant:

| Ranker | Accuracy | Persona 95% CI | Output |
| --- | ---: | --- | --- |
| References only (current model) | 60.95% | 56.79% to 65.34% | `results/qwen35_4bselector_raw/personamem_v1/32k/comparison_memory_2048_references_only` |
| Qwen grades, best of 27 settings | 62.65% | 59.48% to 65.78% | `.../comparison_memory_2048_qwen_graded_all` |

Paired by question this is +1.70 points (persona bootstrap −1.37 to +4.51). The Qwen-graded ranker gets 27 questions right that the other misses and 17 the other way; it is equal or higher on every V1 question type, most on `track_full_preference_evolution` (+3.6). The rankers share 61% of their selected messages. For reference, Qwen with the full V1 history scores 62.48% (measured with the persona messages still in the history).

On LoCoMo-MC10 (1,986 questions, 10 conversations, same settings), the Qwen-graded ranker is much better:

| Ranker | Accuracy | Persona 95% CI | Output |
| --- | ---: | --- | --- |
| References only (current model) | 60.42% | 57.86% to 63.13% | `results/qwen35_4bselector_raw/locomo_mc10/full/comparison_memory_2048_references_only` |
| **Qwen grades, best of 27 settings** | **67.98%** | 65.82% to 70.28% | `.../comparison_memory_2048_qwen_graded_all` |

Paired by question this is +7.55 points (conversation bootstrap +5.17 to +10.28; 247 questions gained, 97 lost). It is +10 to +14 points on `single_hop`, `multi_hop`, `open_domain` and `temporal_reasoning`, and −5.8 on `adversarial` (unanswerable questions, where more on-topic memory makes Qwen pick an answer). The rankers share only 48% of their selected messages. Qwen with the full LoCoMo history scores 68.83%. LoCoMo questions ask for facts stated in the conversation, so ranking by topical match is what they need; PersonaMem V2 references are often indirect, where the same bias hurts. The forget-weight refit also matters here: the previous reference-only model (forget weight 1.0, `results/qwen35_4bselector_raw_2/locomo_mc10/full/comparison_memory_2048`) scored 63.19%, 2.77 points above the current one (paired −4.26 to −1.27).

**Overall, across the three evaluations** (2,048 memory tokens for both rankers; full history uses the whole visible context):

| Dataset | Questions | Full history | References-only ranker | Qwen-graded ranker | Qwen grades − references only (paired 95%) |
| --- | ---: | ---: | ---: | ---: | --- |
| PersonaMem V2 test | 1,119 | 37.18%† | **39.41%** | 37.27% | −2.14 (−3.61 to −0.61) |
| PersonaMem V1 | 589 | 62.48%† | 60.95% | **62.65%** | +1.70 (−1.21 to +4.66) |
| LoCoMo-MC10 | 1,986 | **68.83%** | 60.42% | 67.98% | +7.55 (+5.22 to +10.40) |
| **Macro average** (each dataset equal) | 3,694 | **56.16%** | 53.59% | 55.96% | **+2.37** (+0.94 to +3.81) |
| **Pooled** (all questions) | 3,694 | **58.23%** | 54.14% | 57.82% | **+3.68** (+2.17 to +5.40) |

Intervals resample each dataset's groups (personas or conversations) within that dataset, 5,000 draws. Across datasets the Qwen-graded ranker is better on both averages, with intervals above zero; it loses only on PersonaMem V2, the dataset the rankers were selected on. Full history is level with the Qwen-graded ranker on average (macro +0.20 points, −1.48 to +1.99) but uses the whole context, up to 38k tokens per prompt instead of 2,048. The pooled average is more than half LoCoMo. The reference-only model remains the default in the scripts.

† Scored with the dataset's persona (system) messages still in the history. Full history no longer includes them (see [Qwen evaluation](#qwen-evaluation)), and these two baselines have not been rerun without them.

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

The selected 55-feature ranker uses LightGBM `LGBMRanker` with LambdaRank and linear relevance gains `label_gain=[0,1,2]`. The trainer and tuner accept supplied NPZ `y` values 0, 1, or 2; binary 0/1 labels remain valid. Grade 2 has twice the gain of grade 1. Unjudged -1/null rows must be excluded before fitting, with question offsets rebuilt. Candidate JSONL files are not loaded automatically. The feature builder still exports binary reference-membership `y` alongside separate original-reference and unjudged arrays; using three grades requires supplying the intended graded `y`. Tuning validation metrics continue to measure original-reference retrieval. Use one candidate row per original message, retain complete question groups, and preserve persona-disjoint splits.

Training gives `ask_to_forget` questions weight **0.5** and every other question weight 1 (`--forget-weight` in `ranker55.py` and `tune_ranker55.py`; `1` restores equal weights). Every row of a question carries its weight, so it scales whole questions without changing the order learned within one. Forget questions are easy and about a fifth of training; at full weight they pulled the ranker away from preference evidence. On V2 validation with the tuned setting (100 / 63 / 6 / 10):

| Forget weight | Non-forget type-macro Hit@10 | NDCG@10 | `ask_to_forget` Hit@10 | All-question Hit@10 |
| --- | ---: | ---: | ---: | ---: |
| 1.0 (previous) | 65.77% | 35.63% | 96.34% | 72.28% |
| **0.5 (current)** | **67.20%** | **38.00%** | 93.19% | **72.83%** |
| 0.25 | 68.1% | 38.7% | 90.58% | 72.83% |
| 0.1 | 68.9% | 40.3% | 78.53% | 70.96% |
| 0 (forget dropped) | 69.0% | 40.8% | 43.98% | 63.81% |

The weight was chosen on validation, so these are development results. Rows 0.25, 0.1 and 0, and the logistic-regression and no-role figures below, were not saved at full precision: they are the earlier six-type figures multiplied by 6/5 and shown to one decimal. The question type is used only for training weights, never as a ranking input. In the same experiments, larger trees (127/7, 255/8), 600 rounds, `max_bin` 255, type-balanced weights, logistic regression on the same 55 features (42.6%), and an extra message-specificity feature did not improve Hit@10. Role matters only together with other features: references are 50.0% assistant messages, the same as all messages (49.8%), yet setting `role_assistant` to 0 in training and validation lowers Hit@10 by about 1.2 points for this model (67.20% → 66.0%, persona bootstrap −3.2 to +0.8) and by 2.2 for the all-rows Qwen-graded model (64.19% → 62.0%, −3.7 to −0.7), whose useful labels are 67.7% assistant messages. `models/subtlex_mix/alpha_0.5/ranker55.txt` is the weighted refit; the previous model remains as `ranker55_trials/trial_17.txt`, and the comparison is in `results/ranker55_forget_weight/v2/val/report.json`.

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

This command waits for an existing training feature build without restarting it. If the validation NPZ is missing, it builds V2 val with the training bundle's SUBTLEX recipe and alpha. It then evaluates 27 settings: `min_child_samples` in `{50,100,200}`, paired `num_leaves/max_depth` in `{15/4,31/5,63/6}`, and `reg_lambda` in `{1,5,10}`. The default `100 / 31 / 5 / 5` is fitted first and saved immediately. Features, binary label gains, learning rate, 300 boosting rounds and the 0.5 forget-question weight remain fixed. All fitting uses training rows; validation selects settings and test is not loaded.

Override the grid with `--min-child-samples 50 100`, `--tree-shapes 15:4 31:5`, or `--reg-lambda 1 5`. For one fit, `scripts/ranker55.py` accepts scalar `--min-child-samples`, `--num-leaves`, `--max-depth`, and `--reg-lambda` options.

Selection uses non-forget **type-macro Hit@10** over all validation questions (five question types since the `sensitive_info` removal). A setting replaces the default only if it improves by strictly more than one percentage point and does not reduce type-macro NDCG@10 (`--adoption-threshold 0.01`). Ties favor earlier source messages. Reports include per-type and question-weighted results, incremental Hit@10 and question-level tail rescue over cosine, plus complete-reference-only results. Missing annotated reference messages remain in the NDCG ideal denominator and cannot create hits. These validation-selected results are development measurements; final test evaluation remains separate.

The adopted model overwrites `models/subtlex_mix/alpha_0.5/ranker55.txt`, ready for selector-mode Qwen evaluation. Trial models stay under `models/ranker55_trials/`; `report.json`, `trials.csv`, `per_question.jsonl`, and `REPORT.md` go under `results/ranker55_tuning/v2/val/`. The tuner maps the builder's uncompressed NPZ arrays and reuses a binned LightGBM training dataset across configurations. It never uses Qwen-generated judgments; adding those still requires a separate labeling and graded-training pass.

The local evaluator uses **Unsloth Qwen3.5-4B Q4_K_M GGUF** through the native llama.cpp CUDA server. It is the sole supported Qwen model. All layers are requested on the GPU, FlashAttention and prompt caching are enabled, thinking and context shifting are disabled, and requests are serial. Original raw datasets and split IDs are loaded directly.

| Dataset | Evaluation questions | Supported `--mode` values |
| --- | --- | --- |
| V2 | Active test IDs (1,119) | `full_history`, `no_memory`, `reference_only`, `selector` |
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

There are no `--split` or `--subset` arguments. V2 always uses the active test IDs in `data/splits/v2/test_ids.json`; their existing compact/full provenance remains report metadata. V1 and LoCoMo use all comparison IDs. `--limit` remains available for small smoke tests. Automatic resume follows the same dataset rules.

`full_history` passes every visible user and assistant message and nothing else. The datasets' own system messages are left out: in V2 the JSON persona profile at the head of each history (about 1,080 Qwen tokens), in V1 the persona description that opens each session (4 to 7 per history, about 890 tokens in all). The selector and the oracle never saw these messages, because retrieval memories are user and assistant messages only, so with them in, full history read evidence no ranker could select. The remaining messages keep their source index labels. LoCoMo has no system messages and its prompts are unchanged. This departs from the official PersonaMem setup, which keeps the profile in the context, so full-history results here are not comparable with published ones. New V1 and V2 full-history runs write to `comparison_messages_only` and `test_messages_only`. The saved runs in `comparison` and `test` (62.48% and 37.18%) were scored with the persona messages and have not been rerun; the evaluator refuses to resume them.

`reference_only` reads only the original `related_conversation_snippet` annotation, corresponding to reference label **2**. It never reads Qwen-generated labels or adds label **1** messages. This is an oracle baseline, with reference messages preserved verbatim in original snippet order. If an annotated message is absent from the visible history, it is still included and the discrepancy is counted in predictions and the report.

`selector` loads `models/subtlex_mix/alpha_0.5/ranker55.txt` and `data/ranker55/{dataset}/{split}/features.npz` by default (`split=test` for V2, `comparison` otherwise). Override those paths with `--selector-model` and `--selector-features`. The default is a **2,048 Qwen-token memory budget** on every dataset. Use `--memory-tokens 1024` or `--memory-tokens 4096` to change it (`--token-budget` is an alias). The selector scores the complete visible message pool, considers messages in descending score order, and skips messages that do not fit so smaller lower-ranked messages can still be included. Ties favor earlier source indices. Accepted messages stay whole and are presented chronologically with their original roles and LoCoMo session/timestamp metadata.

LightGBM training, parameter tuning, and saved selector prediction explicitly use the CPU (`device_type=cpu`, four threads). Qwen inference continues on the GPU. The local packing tokenizer loads vocabulary only on the CPU.

The budget uses the active Qwen native tokenizer: complete templated prompt tokens minus the instruction/question/options-only prompt tokens. It includes history headings, role/index labels, separators, session headers, and chat framing. The complete prompt must also fit the actual server context with one answer token reserved. No message is truncated; if none fits, the prompt contains only instructions and the question/options. Selected counts, actual token usage, skipped-message counts, and packing time are recorded per question. There is no default message-count cap. For a fixed-count comparison, explicitly use `--top-k 10`; it is mutually exclusive with `--memory-tokens`.

`--score-cutoff D --floor-tokens F` adds a relative score cutoff to token-budget packing: once `F` memory tokens are packed, packing stops at the first message scoring more than `D` below the question's top score (`--floor-tokens` defaults to 0 and cannot exceed the budget; neither works with `--top-k`). Both values are recorded in the protocol, checked on resume, and added to the default output name. With `--score-cutoff 1.5 --floor-tokens 1000` on every evaluation question of the three datasets, the references-only ranker keeps its accuracy (macro +0.14 points, −0.75 to +1.04) with 26% fewer memory tokens, and the Qwen-graded ranker uses 35% fewer and loses 0.87 points (−1.75 to +0.04). The study behind this is `results/selection_threshold_study/REPORT.md`; it was written when LongBench was a fourth dataset and before the V2 `sensitive_info` questions were removed, and still reports both.

Token-budget packing formats source message blocks once per conversation boundary and loads the same GGUF vocabulary locally through the pinned `llama.dll`, without loading another set of model weights or allocating a GPU inference context. It renders the GGUF's Jinja template locally and caches exact counts at Qwen35 pre-tokenizer boundaries, preserving whitespace and separators. Startup probes compare local prompt text, token counts, and token IDs with the active server. Each final selected prompt is checked again with `/apply-template` and `/tokenize` before scoring; a mismatch stops the run. Packing therefore uses two HTTP requests per question instead of two per candidate. The final full prompt and question-only count are reused during scoring. The native binding checks the DLL hash and supports the currently pinned b11669 ABI; a runtime upgrade requires updating that binding. Full-history, reference-only, and explicit top-k modes retain their native HTTP rendering path.

For token-budget selector evaluation, total context capacity is **memory budget + 1,000 tokens** on all datasets: 1,024 → 2,024; 2,048 → 3,048; 4,096 → 5,096. The extra capacity covers instructions, question/options, their formatting, and the one-token answer. `selector_context_overhead_tokens` in `configs/qwen_server.json` stores this 1,000-token allowance; the old fixed `selector_context_tokens` setting has been removed. The native server supports `--mode selector --memory-tokens N` with the same default calculation. If the non-memory portion needs more space, packing reduces the usable memory budget; complete prompts are never truncated. Explicit `--top-k` evaluation retains the dataset's full-history context capacity because it has no memory-token budget.

The 20-message labeling budget does not apply here; annotations, cached labels and correct answers do not participate in selection. Model/feature hashes and selection settings, including the token budget, are checked when resuming. Token-budget results and explicit top-k results use separate default output directories.

Selector evaluation needs a completed saved model and a matching feature bundle built with the frozen SUBTLEX recipe parameters. Once the training build is finished, prepare these inputs separately:

```powershell
python scripts/build_ranker_features.py --datasets v2 --splits test
python scripts/build_ranker_features.py --datasets v1 locomo
```

The evaluator never generates embeddings or fits a model. Its dry run validates selector files and performs one CPU ranking, without starting Qwen. Token packing is explicitly reported as pending during a dry run because exact counts require the active native Qwen tokenizer. Missing or mismatched inputs fail before Qwen starts.

`scripts/qwen_llamacpp_server.py` owns the native process and API. Its `LlamaCppServer` interface starts `models/llama.cpp/llama-server.exe`, checks readiness, handles tokenization/scoring requests and retries, samples resources, and shuts down its owned process. `scripts/evaluate_qwen.py` loads data, prepares messages, calls that interface, grades returned scores, and saves reports. Shared launch settings come from `configs/qwen_server.json`. `--dry-run` inspects data/settings without loading Qwen weights. The server module uses a direct `requests.Session` against llama.cpp native endpoints: `/completion`, `/tokenize`, `/apply-template`, `/health`, and `/props`. Raw token IDs and pre-sampling log-probabilities come directly from `/completion`; prompt caching is enabled explicitly. Full histories are checked by the native tokenizer before scoring and never truncated. Dataset caps remain 41,216 (V2), 33,280 (V1), and 32,256 (LoCoMo); no-memory context is 4,096. Without the persona messages the longest V2 and V1 full-history prompts are 36,743 and 31,641 tokens by the local tokenizer, so both still fit. These originate from the prior exact tokenizer audit and require a complete native-runtime audit before declaring all prompts verified.

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

