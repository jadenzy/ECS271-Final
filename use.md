# How to use the scripts

This guide covers all **19 Python files in `scripts/`**: 15 command-line tools and four shared modules. Commands and defaults were checked against the current source. The project retrieves complete conversation messages, trains a 55-feature LightGBM ranker, and evaluates a frozen Qwen model on multiple-choice questions.

## 1. Set up and choose a workflow

Run the commands below in PowerShell from the repository root, with the project environment activated:

```powershell
Set-Location C:\Users\jaden\Code\ECS271
.venv/Scripts/Activate.ps1
python --version
```

The existing environment uses Python 3.13.9. If setting up a fresh checkout, create the environment first with `py -3.13 -m venv .venv`, then activate it and install dependencies:

```powershell
python -m pip install -r requirements.lock.txt
python -m pip install lightgbm==4.7.0
```

LightGBM is needed for training and saved-model prediction but is missing from `requirements.lock.txt`; 4.7.0 is the version installed in the current environment. The lock file pins a CUDA PyTorch build, so a fresh installation needs a package source that provides that wheel. You can also invoke `.venv/Scripts/python.exe` directly in place of `python` if activation is unavailable.

Prepare the inputs:

- Restore actual dataset contents under `data/raw/` and the matching IDs under `data/splits/`. If using a Git LFS checkout, run `git lfs pull` to obtain LFS-managed contents. The model downloader does **not** download datasets or create splits.
- Keep the provided `data/external/subtlex_us/word_frequencies.json` together with `configs/subtlex_us.lock.json`.
- Download the model assets needed for your task; see `download_models.py` below.
- Feature encoding supports CPU or CUDA. The pinned Qwen runtime is a Windows x64 CUDA build and is configured for GPU inference. Ranker training and prediction run on CPU.

Choose the relevant stages:

| Goal | Sequence |
| --- | --- |
| Train a reference-only selector | Download EmbeddingGemma -> build V2 training features -> train `ranker55.py`, or tune with `tune_ranker55.py` |
| Inspect lexical/dense retrieval | Build the complete V2 training bundle -> `diagnose_retrieval.py` |
| Add Qwen relevance judgments | Train a selector -> select candidates -> label with `--judge` -> build graded training features -> train/tune a separate model |
| Evaluate an existing selector | Obtain matching saved model and feature bundle -> download Qwen/runtime -> evaluator dry run -> evaluator |
| Run a Qwen baseline | Download Qwen/runtime -> evaluator dry run -> evaluator with the desired mode |
| Compare completed runs | Create a run mapping JSON -> `compare_selector_runs.py` |

Example reference-only workflow (these commands generate features and replace the selected model output):

```powershell
python scripts/download_models.py --only embeddinggemma2
python scripts/build_ranker_features.py --datasets v2 --splits train
python scripts/build_ranker_features.py --datasets v2 --splits val test
python scripts/tune_ranker55.py
python scripts/download_models.py --only qwen llama_cpp
python scripts/evaluate_qwen.py --dataset v2 --mode selector --dry-run
python scripts/evaluate_qwen.py --dataset v2 --mode selector
```

If compatible features and trained weights already exist, start at the evaluation steps. Qwen evaluation and judgment commands manage their own server processes; you do not need to start the standalone server first.

### Existing artifacts in this checkout

The existing `data/ranker55/v2/{train,val,test}/features.npz` files are older exports without the required `subtlex_mix_v1` recipe metadata. Their validation/test bundles also contain the old 1,002/1,242-question cohorts, rather than the active 909/1,119 questions. Current consumers reject those files; file existence alone does not mean a bundle is usable. Rebuild the standard bundles with `build_ranker_features.py` before using the default training/evaluation commands above. Their old held-out embedding caches may also require `--rebuild-cache` because the split membership changed.

The existing `data/ranker55/subtlex/v2/{val,test}/features.npz` files have the current alpha-0.5 recipe, active question counts, and a recipe matching the saved reference-trained model. The following dry run passed using the existing test bundle:

```powershell
python scripts/evaluate_qwen.py --dataset v2 --mode selector --selector-features data/ranker55/subtlex/v2/test/features.npz --limit 1 --dry-run
```

Use that same `--selector-features` option for inference with the existing test bundle. No features, caches, weights, or split files were changed while preparing this guide.

## 2. File index

| File in `scripts/` | Purpose | Execution |
| --- | --- | --- |
| `download_models.py` | Download and verify pinned embedding, Qwen, and native runtime assets | CLI; network for downloads |
| `prepare_subtlex_us.py` | Convert raw SUBTLEX-US counts into a normalized lexical prior | CLI; CPU |
| `build_ranker_features.py` | Build embeddings, all 55 features, rankings, and training reference targets | CLI; CPU/CUDA encoding, CPU features |
| `ranker55.py` | Define the feature recipe and train one LightGBM ranker | CLI and library; CPU |
| `tune_ranker55.py` | Train a parameter grid and choose a model on V2 validation | CLI; CPU, with possible validation feature build |
| `select_label_candidates.py` | Select non-reference training messages for Qwen to judge | CLI; CPU |
| `label_reference_messages.py` | Create exact reference labels, or obtain Qwen relevance judgments | CLI; CPU for exact labels, GPU for Qwen |
| `build_graded_training.py` | Join Qwen judgments to feature rows for graded training | CLI; CPU |
| `diagnose_retrieval.py` | Compare BM25, cosine, and reciprocal-rank fusion against references | CLI; CPU |
| `reference_hit_check.py` | Measure saved rankers' reference Hit@10 and NDCG@10 | CLI; CPU |
| `evaluate_qwen.py` | Evaluate Qwen with full history, no memory, references, or selected memory | CLI; GPU inference |
| `qwen_llamacpp_server.py` | Own native Qwen startup, HTTP scoring/tokenization, monitoring, and cleanup | CLI and library; GPU when started |
| `option_rotation_pilot.py` | Measure the effect of rotating answer-option positions | CLI; GPU, or CPU summaries |
| `finetune_embedding_pilot.py` | Fine-tune the embedding model on V2 training references and measure validation retrieval | CLI; GPU |
| `reader_validation.py` | Reader-side tests on V2 validation: prompt wording, Qwen per-message filter, score cutoff | CLI; GPU |
| `evaluate_embedding_ab_cutoff.py` | Score the frozen fine-tuned ranker with a relative score cutoff, optionally with a larger token cap | CLI; GPU |
| `evaluate_embedding_ab_mmr.py` | Score the frozen fine-tuned ranker with redundancy-aware (MMR) packing, four answer orders | CLI; GPU |
| `evaluate_selector.py` | Train a two-pass yes/no message selector on the 57 features (no ranker) and score it with Qwen | CLI; CPU training, GPU scoring |
| `evaluate_graded_selector.py` | The same two-pass selector trained on Qwen-graded 0/1/2 labels, references first | CLI; CPU training, GPU scoring |
| `neural_ranker.py` | Train a neural message ranker on the 57 features plus projected embedding interactions (reference loss, optional answer loss), validate it, and score the evaluation datasets | CLI; GPU (small) |
| `evaluate_neural_ranker.py` | Score a frozen neural ranker with Qwen at the 3.0-cutoff baseline's per-question message counts | CLI; GPU |
| `compare_selector_runs.py` | Compare saved prediction accuracies with paired bootstrap intervals | CLI; CPU |
| `exclude_eval_questions.py` | Remove a question type from V2 evaluation IDs and filter selected bundles | CLI; CPU; changes split files with `--apply` |
| `raw_datasets.py` | Read raw benchmarks, answers, cutoffs, IDs, and source hashes | Importable module |
| `message_dataset.py` | Expose each visible complete message as one retrieval memory | Importable module |
| `subtlex_mixture.py` | Validate and mix the external lexical prior with visible-history frequencies | Importable module |
| `feature_contract.py` | Check compatibility between feature recipes and saved ranker weights | Importable module |

For the complete argument list of any CLI, use `python scripts/FILE.py --help`. The four import-only modules have no CLI or `--help` handler. `scripts/__pycache__/` contains generated Python bytecode and is not a tool to run.

## 3. Setup and feature construction

### `download_models.py`

**Purpose:** Download the public assets pinned in `configs/assets.lock.json`, validate their sizes and SHA-256 hashes, and skip valid existing files. Rerunning repairs missing/changed assets and reuses download caches where possible.

```powershell
# Download all three groups.
python scripts/download_models.py

# Or download only the assets for the next task.
python scripts/download_models.py --only embeddinggemma2
python scripts/download_models.py --only qwen llama_cpp

# Offline integrity check; does not download or repair files.
python scripts/download_models.py --verify-only
```

**Outputs:** `models/embeddinggemma-2/`, `models/Qwen3.5-4B-GGUF/`, and `models/llama.cpp/`; staging files live under `.cache/model_downloads/`. `--only` also applies to `--verify-only`. A failed verification returns a nonzero exit code. Python packages, raw datasets, and GPU drivers are separate prerequisites.

### `prepare_subtlex_us.py`

**Purpose:** Read the tab-separated SUBTLEX-US `Word` and `FREQcount` columns, normalize words with NFKC/casefold and word tokenization, combine counts, and write token probabilities with source metadata.

```powershell
python scripts/prepare_subtlex_us.py --input data/raw/SUBTLEXusrawdata/SUBTLEXusrawdata.dat --output data/external/subtlex_us/rebuilt_word_frequencies.json
```

`--input` is required; `--encoding` defaults to `cp1252`. Without `--output`, the script replaces `data/external/subtlex_us/word_frequencies.json`.

**Output:** A normalized JSON prior. The example uses a separate filename because feature construction verifies the exact processed-file hash in `configs/subtlex_us.lock.json`. Regeneration includes a creation timestamp, so even identical source counts can produce a different file hash. This tool does not update the lock manifest. Use the provided pinned prior for the normal workflow; a custom prior needs a matching manifest passed to the feature builder.

### `build_ranker_features.py`

**Purpose:** Read original messages and active split IDs, encode complete messages/questions and auxiliary dense spans with pinned EmbeddingGemma 2, and compute every feature in `ranker55.FEATURE_NAMES`. It also exports complete BM25/cosine rankings. It does not train a selector or call Qwen.

```powershell
# Default: all active V2 training questions, alpha 0.5.
python scripts/build_ranker_features.py

# Held-out V2 bundles and comparison datasets.
python scripts/build_ranker_features.py --datasets v2 --splits val test
python scripts/build_ranker_features.py --datasets v1 locomo

# Small CPU build in its own output directory.
python scripts/build_ranker_features.py --limit-questions 10 --device cpu --feature-workers 1 --output-root data/ranker55/smoke

# Recalculate features using a completed matching embedding cache; no encoder loads.
python scripts/build_ranker_features.py --datasets v2 --splits train --embedding-cache-root data/ranker55 --output-root data/ranker55/alpha_025 --alpha 0.25
```

**Inputs:** Raw sources, split IDs, the pinned SUBTLEX prior/manifest, and EmbeddingGemma weights unless using read-only embedding caches. Only V2 training loads exact reference targets. V2 accepts `train`, `val`, and `test`; V1 and LoCoMo accept only `comparison`. With no `--splits`, V2 uses `train` and the comparison datasets use `comparison`.

**Outputs:** `data/ranker55/{dataset}/{split}/features.npz`, `statistics.json`, `report.json`, `embeddings/`, `bm25/`, `cosine/`, and, for V2 training, `reference_labels/`. The output root must stay under `data/ranker55/`. The NPZ contains the 55-column matrix, question group offsets, IDs, token lengths, labels, and recipe/provenance metadata.

Useful controls:

- `--alpha 0.5`: external/history mixture weight, from 0 to 1. Models must match the bundle's alpha and recipe.
- `--device auto|cuda|cpu`: `auto` chooses CUDA when available.
- `--batch-size 16`, `--batch-tokens 4800`: embedding batch limits.
- `--feature-workers 4`: current CLI default for CPU feature processes; use 1 or 2 to reduce concurrency.
- `--question-ids PATH`: export a JSON list of selected active IDs; `--limit-questions N` limits each split, with 0 meaning all.
- `--split-root PATH`: alternative split directory.
- `--subtlex-prior PATH`, `--subtlex-manifest PATH`: use a compatible custom pinned prior.
- `--embedding-cache-root PATH`: read completed caches under `{dataset}/{split}/embeddings/`. Inputs, order, source hashes, and recipes must match exactly; partial caches are rejected.
- `--rebuild-cache`: overwrite incompatible caches in a normal encoding build. It cannot be combined with read-only cache reuse.

Normal encoding resumes matching interrupted embedding caches. Feature exports are recalculated and completed files are replaced; changing alpha at the same output root replaces the previous exports. Whole messages must fit the 1,200-token encoder limit; the builder raises an error instead of truncating them. A small build is useful for a smoke run, but the tuner and training retrieval diagnostics require the complete active training cohort.

## 4. Ranker training and relevance labels

### `ranker55.py`

**Purpose:** Define the ordered 55 features and their construction, and fit one 300-tree CPU LightGBM LambdaRank model from an existing NPZ. It accepts training grades 0/1/2 with linear gains. It performs no hyperparameter search or held-out evaluation.

```powershell
# Print feature names, embedding recipe, and training parameters without fitting.
python scripts/ranker55.py --describe

# Fit from the standard training bundle.
python scripts/ranker55.py --features data/ranker55/v2/train/features.npz --output models/subtlex_mix/alpha_0.5/ranker55.txt

# Fit a separate model from a graded bundle.
python scripts/ranker55.py --features data/ranker55/v2/train_graded_all/features.npz --output models/qwen_graded_all/ranker55.txt
```

**Inputs:** NPZ arrays `X`, `y`, `offsets`, `feature_names`, and compatible recipe metadata. The default question weighting also needs `question_ids` and active V2 training metadata. Validation/test bundles have unjudged targets and cannot serve as training inputs.

**Outputs:** The model text file plus its `.txt.recipe.json` sidecar. Existing model outputs are replaced after successful fitting. Keep the sidecar with the weights.

Defaults are `--min-child-samples 100`, `--num-leaves 31`, `--max-depth 5`, `--reg-lambda 5.0`, and `--forget-weight 0.5`. The last option gives `ask_to_forget` questions half weight; `--forget-weight 1` weights all questions equally. `feature_matrix()` is also imported by the feature builder.

### `tune_ranker55.py`

**Purpose:** Fit candidate parameter settings on V2 training data, measure reference retrieval on V2 validation, and select a saved ranker. Supplied training `y` can be binary or graded; validation annotations are used for measurement/selection, and test is not used.

```powershell
python scripts/tune_ranker55.py

# Tune a separately saved graded model.
python scripts/tune_ranker55.py --train-features data/ranker55/v2/train_graded_all/features.npz --output models/qwen_graded_all/ranker55.txt --output-root results/ranker55_tuning_graded_all/v2/val

# A single-setting check, still requiring complete active training/validation bundles.
python scripts/tune_ranker55.py --min-child-samples 100 --tree-shapes 31:5 --reg-lambda 5 --output models/tuning_smoke/ranker55.txt --output-root results/tuning_smoke/v2/val
```

**Inputs:** By default, `data/ranker55/v2/{train,val}/features.npz`, matching recipe/statistics metadata, all active training questions, and disjoint training/validation personas. Use `--train-features` and `--val-features` to override paths. If the standard validation bundle is missing, the CLI builds it automatically with matching alpha, which can load EmbeddingGemma and use a GPU. Build validation explicitly first to keep tuning CPU-only.

The default grid has 27 unique settings: minimum child samples 50/100/200, paired leaves/depth shapes `15:4`, `31:5`, `63:6`, and L2 penalties 1/5/10. The default setting is evaluated first. Adoption requires non-forget type-macro Hit@10 to improve by **more than** `--adoption-threshold` (default 0.01, or one percentage point) without lower NDCG@10. `--forget-weight` defaults to 0.5.

**Outputs:** Selected model and recipe sidecar, trial weights in `ranker55_trials/` beside the model, and `report.json`, `trials.csv`, `per_question.jsonl`, and `REPORT.md` under `results/ranker55_tuning/v2/val/` by default. Model destinations must be under `models/`; reports must be under `results/`. Use separate destinations to preserve earlier experiments.

### `select_label_candidates.py`

**Purpose:** Use an existing saved selector to choose **20 distinct non-reference messages per V2 training question** for Qwen labeling: 12 ranker picks, three weighted-BM25 picks, three dense-chunk picks, and two random picks. Exact references remain grade 2 outside that budget. This step uses CPU prediction and does not call Qwen or retrain the ranker.

```powershell
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz

# Pilot in a separate directory, with optional soft diversity.
python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz --limit-questions 100 --diversity-weight 0.1 --output-root data/label_candidates/pilot/v2
```

**Inputs:** A complete-visible-pool NPZ, active raw training histories/references, saved weights, and the matching recipe sidecar. `--features` is required; the default model is `models/subtlex_mix/alpha_0.5/ranker55.txt`. Selected non-references remain unjudged until the Qwen stage; cached `y` is ignored.

**Outputs:** `train.jsonl` and `report.json` under `data/label_candidates/v2/` by default. `--output-root` is the final directory and must be under `data/label_candidates/`. Rerunning replaces the candidate export. `--limit-questions 0` (the default) selects all cached training questions; `--seed` defaults to 271.

Optional `--diversity-weight` (0 to 1, default 0) uses complete-message embedding caches to encourage variety within lexical/dense allocations. `--embedding-root` overrides the bundle's recorded cache location. `--model PATH` selects other compatible weights. Alternatively, `--fold-manifest PATH` routes questions to saved persona-held-out models; it is mutually exclusive with `--model` and does not train folds. Its JSON `models` entries declare `path`, `sha256`, `training_persona_ids`, and `held_out_persona_ids`, with IDs such as `v2:7`.

### `label_reference_messages.py`

**Purpose:** Support two separate labeling operations.

**Exact-reference mode:** Match the original `(role, content)` pairs from V2 reference snippets to visible complete messages. Every exact matching occurrence gets 2; other messages get 0, meaning annotation non-membership rather than verified irrelevance.

```powershell
python scripts/label_reference_messages.py --splits train
```

This mode needs raw data and active IDs, with no model or rankings. It writes `data/reference_labels/v2/train.jsonl` and `report.json`. `--splits` also accepts `val` and `test`, but unmatched references raise an error; the current held-out splits retain a few incomplete references, so do not assume exact-label exports will succeed for them. `--split-root` (alias `--input-root`) changes the split directory. `--output-root` must stay under `data/reference_labels/` and has `/v2` appended.

**Qwen judge mode:** Score selected non-reference candidates using raw a/b label probabilities: useful is 1, not useful is 0, references remain 2, and unselected messages remain -1 in the memory label array.

```powershell
# Small pilot using one process and one request slot.
python scripts/label_reference_messages.py --judge --candidates data/label_candidates/pilot/v2/train.jsonl --limit-questions 100 --replicas 1 --parallel 1 --output-root data/qwen_labels/pilot

# Full candidate export, using the same conservative concurrency.
python scripts/label_reference_messages.py --judge --replicas 1 --parallel 1
```

Judge mode requires Qwen/runtime assets and a candidate export. It always uses active V2 training data; exact-mode `--splits` and `--split-root` do not change the judge cohort. The default candidate file is `data/label_candidates/v2/train.jsonl`. The default output is `data/qwen_labels/v2/`; a custom `--output-root` must stay under `data/qwen_labels/` and has `/v2` appended. Outputs include `train.jsonl`, `protocol.json`, `report.json`, and logs under `servers/`.

Completed questions resume on rerun with compatible inputs/model/prompt/plan. `--limit-questions 0` means all candidate questions. The actual defaults are three server replicas and four slots per replica, which require substantially more VRAM than the examples above. Ports start at the configured port and increase for each replica. The judge reduces concurrency on GPU-memory errors. Optional `--candidate-plan "ranker=12,qweight_bm25=3,dense_chunk=3,random=2"` restricts each allocation to its first N candidates; use a separate output root when changing plans or inputs.

### `build_graded_training.py`

**Purpose:** Join existing Qwen judgments to V2 training feature rows, validate message identities and grades, and create an NPZ for graded ranking. This step performs no inference or training.

```powershell
python scripts/build_graded_training.py --all-rows

# Join a partial pilot for a ranker55.py smoke fit.
python scripts/build_graded_training.py --features data/ranker55/v2/train/features.npz --labels data/qwen_labels/pilot/v2/train.jsonl --allow-partial --all-rows --output-root data/ranker55/v2/pilot_graded
```

**Default inputs:** `data/ranker55/v2/train/features.npz` and `data/qwen_labels/v2/train.jsonl`; override with `--features` and `--labels`.

- Without `--all-rows`, unjudged rows are dropped and offsets rebuilt; output is `data/ranker55/v2/train_graded/`.
- With `--all-rows`, every row is retained. Unjudged rows get training `y=0`, while `judged_labels=-1` preserves their unknown status; output is `data/ranker55/v2/train_graded_all/`. The source documents poorer retrieval from judged-only training, so this is the full-history training variant used in the examples.
- Every source question must be judged unless `--allow-partial` is supplied. Partial outputs can be fitted by `ranker55.py`; the tuner requires every active training question.

**Outputs:** `features.npz`, a copied `statistics.json` when present, and `report.json`. A custom output must remain under `data/ranker55/` and separate from the source bundle. These are new training inputs; fit or tune a model afterward.

## 5. Retrieval diagnostics

### `diagnose_retrieval.py`

**Purpose:** Fuse complete BM25/cosine rankings with reciprocal-rank fusion (RRF) and measure reference recall, Hit@k, full reference coverage, and positive-rank distributions. It reads full V2 training exports and needs no model or GPU.

```powershell
python scripts/diagnose_retrieval.py
python scripts/diagnose_retrieval.py --feature-root data/ranker55/v2/train --rrf-constant 60 --k 5 10 20 30 50 --no-plot
```

**Inputs:** `bm25/`, `cosine/`, and `reference_labels/` exports and reports inside the feature bundle directory. `--feature-root` defaults to `data/ranker55/v2/train`. The separate exact-label export in `data/reference_labels/` is not required when the builder has produced these bundled labels. Limited training bundles are rejected because all active training questions must be present.

**Outputs:** `data/rrf/v2/scores/train.jsonl` and reports under `results/selector_diagnostics_single_message/v2/train/`: `summary.json`, `REPORT.md`, `metrics.csv`, `per_question.jsonl`, `positive_ranks.jsonl`, and histogram CSV/PNG. `--no-plot` skips the PNG. `--output-root` has `/v2/train` appended and must stay under `results/selector_diagnostics_single_message/`; `--rrf-root` has `/v2/scores/train.jsonl` appended and must stay under `data/rrf/`.

### `reference_hit_check.py`

**Purpose:** Measure saved rankers against original V2 validation or test references on CPU. The headline is non-forget type-macro Hit@10, with NDCG@10 and per-type results. It does not tune or train a model; test annotations are read only for measurement.

```powershell
python scripts/reference_hit_check.py --split val --features data/ranker55/v2/val/features.npz
python scripts/reference_hit_check.py --split test --features data/ranker55/v2/test/features.npz

# Choose the models explicitly using label=path entries.
python scripts/reference_hit_check.py --split val --features data/ranker55/v2/val/features.npz --models references_only=models/subtlex_mix/alpha_0.5/ranker55.txt
```

**Inputs:** The selected split's matching feature bundle, source annotations, weights, and recipe sidecars. The script's own default feature path is **`data/ranker55/subtlex/v2/{split}/features.npz`**, unlike the builder's standard output; the examples explicitly point to the standard path. Without `--models`, it compares `models/subtlex_mix/alpha_0.5/ranker55.txt` and `models/qwen_graded_all/ranker55.txt`, so both must exist and be compatible.

**Output:** `results/reference_hit/v2/{split}/report.json`. `--tag NAME` changes the filename to `report_NAME.json` to preserve another measurement.

## 6. Qwen inference and answer evaluation

### `qwen_llamacpp_server.py`

**Purpose:** Provide the native server lifecycle and API used by the evaluator and judge: startup/readiness, prompt rendering/tokenization, raw label scoring, retries, resource monitoring, and owned-process cleanup. It also exposes a standalone server command.

```powershell
# Inspect settings without loading weights or opening a port.
python scripts/qwen_llamacpp_server.py --dataset v2 --print-settings
python scripts/qwen_llamacpp_server.py --mode selector --memory-tokens 2048 --print-settings

# Run a standalone server until stopped (Ctrl+C).
python scripts/qwen_llamacpp_server.py --dataset v2
```

**Inputs:** `configs/qwen_server.json`, `models/llama.cpp/llama-server.exe` and its DLLs, and `models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf`. The runtime uses native `/completion`, `/tokenize`, `/apply-template`, `/health`, and `/props` endpoints through `requests`.

`--dataset` accepts `v2`, `v1`, or `locomo`. `--mode` accepts `full_history`, `no_memory`, `reference_only`, and `selector`; this selects launch settings, not a dataset evaluation. `--ctx-size` overrides the context capacity; `--port` overrides the port. The host is restricted to `127.0.0.1`.

Current configured context capacities are 41,216 tokens for V2, 33,280 for V1, 32,256 for LoCoMo, and 4,096 for no-memory. Selector mode defaults to 2,048 memory tokens plus 1,000 context-overhead tokens. Generation settings come from the JSON config; the current `generation.top_k` is 40. Context shifting and automatic fit adjustment are disabled.

**Output:** A running HTTP server; the standalone command does not create an evaluation report. Evaluator/judge callers write server logs in their own output folders. Leave the server port free when using those tools, because they launch their own server.

### `evaluate_qwen.py`

**Purpose:** Evaluate the frozen Qwen answer model with one of four memory inputs, select the answer by raw pre-sampling label probabilities, and write accuracy/NLL/resource reports. It reads original datasets directly and does not build embeddings or train weights.

| Mode | Input to Qwen | Supported datasets |
| --- | --- | --- |
| `full_history` | All visible conversation messages, excluding source system/persona messages | V2, V1, LoCoMo |
| `no_memory` | Instructions, question, and options only; source histories are not loaded | V2 |
| `reference_only` | Original annotated reference snippets, an oracle baseline, shown under the same heading and source-position labels as `selector` | V2 |
| `selector` | Complete messages chosen by saved LightGBM scores and packed into a memory budget, or explicit top-k | V2, V1, LoCoMo |

The CLI evaluates **V2 test** or the **full V1/LoCoMo comparison** split. It has no `--split` option. LoCoMo has ten answer labels; V1/V2 have four.

```powershell
# Validate data/settings without launching Qwen.
python scripts/evaluate_qwen.py --dataset v2 --mode full_history --dry-run
python scripts/evaluate_qwen.py --dataset v2 --mode selector --memory-tokens 2048 --dry-run

# Ten-question inference pilot in a separate output folder.
python scripts/evaluate_qwen.py --dataset v2 --mode selector --limit 10 --output results/selector_smoke

# Full baseline or selector runs.
python scripts/evaluate_qwen.py --dataset v2 --mode full_history
python scripts/evaluate_qwen.py --dataset v2 --mode no_memory
python scripts/evaluate_qwen.py --dataset v2 --mode reference_only
python scripts/evaluate_qwen.py --dataset v2 --mode selector --memory-tokens 2048
python scripts/evaluate_qwen.py --dataset v1 --mode selector
python scripts/evaluate_qwen.py --dataset locomo --mode full_history

# Explicit message-count selection instead of token packing.
python scripts/evaluate_qwen.py --dataset v2 --mode selector --top-k 20

# Stop packing when a candidate is more than 1.5 below the best score,
# after first filling at least 1000 memory tokens when possible.
python scripts/evaluate_qwen.py --dataset v2 --mode selector --memory-tokens 2048 --score-cutoff 1.5 --floor-tokens 1000
```

**Selector inputs:** By default, `models/subtlex_mix/alpha_0.5/ranker55.txt` and `data/ranker55/{dataset}/{split}/features.npz`, with compatible recipe/provenance and the model sidecar. Override using `--selector-model PATH` and `--selector-features PATH`. A selector dry run validates these inputs and does one CPU ranking, but exact token packing remains pending until the native tokenizer is running.

`--memory-tokens` (alias `--token-budget`) defaults to 2,048 and is mutually exclusive with `--top-k`. `--score-cutoff` is a score gap from the best message, not an absolute relevance threshold; it only applies to token-budget selection. `--floor-tokens` requires a cutoff and cannot exceed the memory budget. Packing preserves whole messages, then restores chronological order. Token-budget context is the memory budget plus configured overhead; explicit top-k uses the dataset's full-history capacity. Complete prompts are checked before scoring and never truncated.

**Outputs:** `predictions.jsonl`, `protocol.json`, `summary.json`, `REPORT.md`, and `qwen-server.log` in the run directory. Current default families are `results/qwen35_4b{mode}_raw/`, followed by `personamem_{dataset}/32k/` or `locomo_mc10/full/`, a `compact/` component where applicable, and a run name identifying split/limits/budget. `--output` chooses an explicit directory inside the project. Use distinct directories for distinct selector models/settings.

**Resume:** Repeat the same command/output to resume compatible saved predictions. With no arguments (or only `--dry-run`), the script first searches for the latest compatible unfinished run for the current model; if none exists, the default is full-history V2 test. Supply explicit dataset/mode arguments to choose a particular evaluation. Changed data, answer ordering, model identity, memory inputs, or selector settings can make a saved run incompatible.

### `option_rotation_pilot.py`

**Purpose:** Check whether answer positions affect Qwen predictions. For each sampled V2 or V1 question, keep the memory fixed, score every cyclic answer-option rotation, map scores back to answer texts, and compare one-pass accuracy with combined rotation scores. The memory is the selected messages (`--mode selector`, default), every visible user and assistant message (`--mode full_history`, with source system/persona messages excluded as in the evaluator), or the annotated V2 reference snippets (`--mode reference_only`, the oracle). Rotation 0 sends exactly the evaluator's single-pass prompt. The selector and the oracle share one memory format, as in the evaluator: "Selected conversation messages (in chronological source order)" with each message labelled "Message" plus its source position. `--prompt-format reference` swaps in the oracle's earlier wording ("Dataset reference messages", "Reference message 0, 1, …") for either mode without changing the messages or their order (V2 snippet order is already chronological source order); it is kept to reproduce the label test.

```powershell
python scripts/option_rotation_pilot.py --split val --questions 200 --features data/ranker55/v2/val/features.npz

# Compare another selector, keeping its predictions in a separate label folder.
python scripts/option_rotation_pilot.py --split val --questions 200 --features data/ranker55/v2/val/features.npz --label qwen_graded --model models/qwen_graded_all/ranker55.txt

# Full-history baselines with four rotations on all of V2 test and V1.
python scripts/option_rotation_pilot.py --mode full_history --split test --questions 0
python scripts/option_rotation_pilot.py --mode full_history --dataset v1 --questions 0

# Label test on V2 test: the oracle, the oracle with its earlier wording, and the selection with that wording.
python scripts/option_rotation_pilot.py --mode reference_only --split test --questions 0
python scripts/option_rotation_pilot.py --mode reference_only --split test --questions 0 --prompt-format reference
python scripts/option_rotation_pilot.py --split test --questions 0 --prompt-format reference --features data/ranker55/v2/test/features.npz

# Recompute a saved pilot's summary without loading Qwen.
python scripts/option_rotation_pilot.py --split val --label references_only --summarize-only
```

**Inputs:** Raw V2 validation (default) or test data, or V1 with `--dataset v1` (its only split, `comparison`, is the default there). Selector mode also needs a selector model/sidecar and its matching feature bundle; full-history mode needs neither. `--questions` defaults to 200; 0 uses the entire split. `--seed` defaults to 271 and `--memory-tokens` to 2,048 (selector mode only). As with `reference_hit_check.py`, the default feature path is `data/ranker55/subtlex/{dataset}/{split}/features.npz`; the examples override it explicitly. Questions sharing a history are scored back to back, so the server's prompt cache carries the long shared prefix across questions and rotations.

**Outputs:** `results/option_rotation/{dataset}/{split}/{label}/predictions.jsonl`, `report.json`, and a server log for inference runs. Each prediction row's `predicted_label` and `correct` describe rotation 0, so the folder can also be listed as a one-pass run in `compare_selector_runs.py`. `--output-root` changes the prefix. Existing question IDs are skipped on rerun. Use a new label or output root when changing the model, inputs, seed, or budget: this pilot resumes by question ID and does not apply the main evaluator's full protocol compatibility checks. The default label is `references_only` in selector mode (the reference-trained selector; it does not activate oracle reference-only memory) and the mode name otherwise (`full_history`, `reference_only`), with `_reference_labels` appended for `--prompt-format reference`. Swapped-label runs keep their own mode's server settings (context size, selector budget check), so only the prompt text differs from the native run. `--summarize-only` requires existing predictions and rewrites the report.

### `finetune_embedding_pilot.py`

**Purpose:** Test whether contrastive fine-tuning of EmbeddingGemma 2 on V2 training references improves retrieval. Each training question is pulled toward its annotated reference messages and away from the non-reference messages of its own history that the unchanged model scores highest (read from the training bundle's `cosine` feature), plus the other messages in the batch. Messages whose text equals one of the question's references are never negatives. Inputs use the pipeline's own format (prefixes, masked mean pooling, L2 normalization); training truncates messages to `--max-tokens` (256), evaluation never truncates.

```powershell
python scripts/finetune_embedding_pilot.py --train-questions 6000 --batch-size 12 --evaluate-every 125
python scripts/finetune_embedding_pilot.py --train-questions 0    # unchanged model only
```

**Measurement:** V2 validation, whole-message cosine alone (no ranker): the share of questions with every visible reference in the top 15, with any reference in the top 10 or top 5, reference recall at 15, and the median rank of each question's lowest reference. Validation personas are disjoint from training; the test split is not read. Validation is measured every `--evaluate-every` steps and at the end; nothing is chosen from it (no early stopping).

**Outputs:** `results/embedding_finetune_pilot/{label}/report.json` (config, validation curve, training loss). Weights are kept only with `--save-model <folder>`; a saved model (about 1.2 GB) belongs in a git-ignored folder. One step holds `--batch-size` questions with up to `--positives` references and `--hard-negatives` plus `--random-negatives` other messages each, so GPU memory does not depend on `--train-questions`; batch 12 peaked at 10.3 GiB on the RTX 5080.

### Reader-side experiments (`reader_validation.py`, `evaluate_embedding_ab_cutoff.py`, `evaluate_embedding_ab_mmr.py`)

These change what Qwen reads from the frozen fine-tuned 57-feature ranker's 2,048-token selection; the ranker, its scores and the answer-order policy are unchanged. All three need the saved models and cached features from the persona-disjoint A/B experiment (`models/embedding_ab/`, `data/ranker55/embedding_ab/`) and a local Qwen server.

```powershell
# V2 validation, paired arms on one server (four orders; resumable). Arms: oracle|ranker|filter_<rule> + _current|_clean
python scripts/reader_validation.py --arms ranker_current filter_cut3_current --limit 500
python scripts/reader_validation.py --stage select      # save the ranker's selection per validation question
python scripts/reader_validation.py --stage judge       # Qwen judges each selected message (question + message only)
python scripts/reader_validation.py --stage policies    # reference coverage of each filter rule (CPU)
python scripts/reader_validation.py --arms selector_current --limit 500   # graded selector's tight set (usefulness >= 0.4)

# A whole dataset (v2 test, v1, locomo) with a score cutoff (keep messages within D of the top score), or MMR packing at penalty 0.1
python scripts/evaluate_embedding_ab_cutoff.py --datasets v1 --cutoff 3.0
python scripts/evaluate_embedding_ab_cutoff.py --datasets v1 locomo --cutoff 3.0 --memory-tokens 8192   # cutoff decides the size, 8,192-token cap
python scripts/evaluate_embedding_ab_cutoff.py --datasets v2 v1 locomo --cutoff 3.0 --memory-tokens 8192 --neighbour-margin 5.0   # plus neighbours
python scripts/evaluate_embedding_ab_mmr.py --datasets v1
```

With the default `--memory-tokens 2048` the cutoff filters the usual 2,048-token selection. With a larger cap, packing walks the whole visible pool in rank order and keeps every message within D of the top score that fits the cap, so the cutoff rather than the budget decides how much Qwen reads; the server context grows to the cap plus the selector overhead.

`--neighbour-margin M` (capped path only) adds, after the cutoff packing, each kept message's chronological neighbours (message index ±1) whose own ranker score is within M of the top, then repacks the whole set under the same cap. Only ranker scores are used. Questions whose expanded set equals the plain cutoff set reuse that run's reads from `{dataset}_cut{D}_mem{cap}/predictions.jsonl`, so the comparison has no GPU drift there; the plain cutoff run must exist first. Rows record `neighbours_added`, `neighbour_candidates` and `reused_cutoff_read`. Outputs go to `{dataset}_cut{D}_mem{cap}_nb{M}/`.

`reader_validation.py` reads V2 validation only (never test). Filter rules: `useful` (messages Qwen calls useful, at least 2), `top4`/`top6`/`top8` (Qwen's most useful by log-probability margin) and `cut3` (within 3.0 of the top score, no Qwen judgment). The `selector` memory source (arm `selector_current`) reads the graded selector's tight set instead: usefulness ≥ 0.4 plus the most useful message, packed by usefulness under the same 2,048-token cap. It uses the validation usefulness cache written by `evaluate_graded_selector.py --train`, checked against that run's training record. `_clean` removes the empty `Memories:` heading from the question message; `_current` keeps it. `--limit N` scores the first N questions in persona order, not a random sample. Outputs go to `results/reader_validation/{arm}/predictions.jsonl` (option-rotation row format, so `compare_selector_runs.py --rotation` applies), plus `selections.jsonl`, `judgments.jsonl` and `policies.json`.

The two transfer scripts write `results/embedding_ab/transfer_cutoff/{dataset}_cut{D}/` (`{dataset}_cut{D}_mem{cap}/` with a larger cap) and `results/embedding_ab/transfer_mmr/{dataset}/` with `predictions.jsonl`, `protocol.json` (a changed protocol refuses to resume), `report.json` and the server log; compare them with the unfiltered run in `results/embedding_ab/transfer/{dataset}/`. The cutoff distance and MMR penalty are fixed by flags and defaults, not tuned in these scripts.

### `evaluate_selector.py`

A two-pass yes/no selector replaces the ranker. It is trained only on V2 persona half B (the ranker's 9,192 questions and 57 features); the ranker's scores are not used.

- **Pass 1:** a LightGBM classifier marks each visible message as reference or not.
- **Pass 2:** rescores pass 1's top 30 messages per question. It adds pool-relative features (rank, gaps, share of the pool's softmax mass), the scores of the messages just before and after, and the 57 features again. Its training inputs are out-of-fold pass 1 scores from five persona folds.

```powershell
python scripts/evaluate_selector.py --train                                   # CPU, about 2 minutes; prints V2 validation coverage per threshold
python scripts/evaluate_selector.py --datasets v1 locomo v2 --threshold 0.25  # GPU; V2/V1 four orders, LoCoMo one pass
python scripts/evaluate_selector.py --datasets v1 --threshold 0.25 --summarize-only
```

Scoring keeps every message with pass 2 probability at or above `--threshold`; the most probable message is always kept. Messages are packed by probability under `--memory-tokens` (default 8,192, so the threshold decides the size) and shown chronologically.

0.25 was fixed on V2 validation before any test set was scored:
1. Reader accuracy by selection outcome (every reference kept with 0, 1–2, 3–5 or 6+ extra messages; some references missing; none kept) was estimated from the saved `reader_validation` runs.
2. Each threshold's validation selections were mapped onto those accuracies.
3. The projection was flat from 0.2 to 0.4, and 0.25 sits in that range.

Before scoring a dataset, the script checks its 57 inputs by reproducing the cached ranker scores exactly.

**Outputs:**
- Models: `models/embedding_ab/selector/pass{1,2}.txt` (git-ignored).
- Training record and validation coverage: `results/selector/training.json`.
- Probabilities: `selector_probabilities.npy` next to each dataset's cached features.
- Runs: `results/selector/{dataset}_t{threshold}/`, with the same files as the cutoff runs.
- Comparison lists: `results/selector/runs_{dataset}.json`.

### `evaluate_graded_selector.py`

This is the same two-pass design, but both passes are three-class LightGBM models trained on half B's Qwen-graded labels. It uses `build_graded_training.py`'s `--all-rows` policy:

| Label | Meaning |
|---|---|
| 2 | annotated reference |
| 1 | Qwen judged it useful (one of 20 judged candidates per question) |
| 0 | Qwen judged it not useful, or it was never judged |

Messages are kept and packed by expected usefulness, `u = P(2) + 0.5·P(1)`, which is the ranker's 0/1/2 gain scaled to 1. References come first and Qwen-useful messages next. Pass 2 also sees pass 1's P(1) and P(2).

```powershell
python scripts/evaluate_graded_selector.py --train                                   # CPU, a few minutes
python scripts/evaluate_graded_selector.py --datasets v1 locomo v2 --threshold 0.07  # GPU, about 35 minutes for all three
```

Training prints two coverage tables per threshold:
- **Half B, out of fold:** references fully kept, share of Qwen-useful messages kept, and messages kept. V2 validation has no Qwen grades, so half B is the only place grade-1 recall can be measured.
- **V2 validation:** reference coverage and exact sets.

The threshold 0.07 matches the 3.0 cutoff's V2 validation coverage. Half B's grades are cached as `graded_labels.npy`, read from `data/qwen_labels/v2/train.jsonl`.

```powershell
python scripts/evaluate_graded_selector.py --datasets v2 v1 locomo --threshold 0.4 --route-confidence 0.4   # routed: best V2 setting
```

`--route-confidence c` gives a question the selector's set only when the selector is confident that set holds every reference. Confidence is the product of 1 − P(2) over the messages it leaves out, and must be at least `c`. Otherwise the question gets the frozen ranker's 3.0-cutoff set under the same cap, packed exactly as by `evaluate_embedding_ab_cutoff.py`. Each prediction row records `route`, `route_confidence` and `selector_set_size`; `report.json` adds the share routed to the selector and mean messages per route. The 0.4/0.4 setting was chosen on V2 validation with a CPU projection from the saved `reader_validation` runs. Runs go to `results/selector_graded/{dataset}_routed_t{t}_c{c}/`. Pass 2's class probabilities are cached as `selector_graded_probabilities.npy` next to each dataset's features.

```powershell
python scripts/evaluate_graded_selector.py --datasets v2 --threshold 0.4 --route-confidence 0.4 --two-view   # tested; lost on V2 test
```

`--two-view` reads each cutoff-route question a second time with the selector's tight set. Its answer is the sum of both reads' renormalized letter log-probabilities per answer order, with equal weights. Reads the routed run already made are reused where the route and the selected messages (same order) match, so only the second view uses the GPU. Rows add `two_view`, `reused_routed_read`, `views` (both reads' shifts) and `selector_view`, and `shifts` holds the combined answer. Runs go to `results/selector_graded/{dataset}_twoview_t{t}_c{c}/`. It scored 55.41% on V2 test against the routed rule's 56.48%, so it is not the default.

**Outputs:**
- Models: `models/embedding_ab/selector_graded/` (git-ignored).
- Training record: `results/selector_graded/training.json`.
- Runs: `results/selector_graded/{dataset}_t{threshold}/`.
- Comparisons: `results/selector_graded/{dataset}_comparison_{rotation,one_pass}.json`.

### `neural_ranker.py` and `evaluate_neural_ranker.py`

**Purpose:** Train a new message ranker on the frozen half-A embeddings and the 57 features, using V2 half B's annotated references (variant B) and, in variant C, also the dataset's correct answer. Then test the frozen rankers with Qwen at the same message counts as the 3.0-cutoff baseline. No Qwen call is made during training, checkpoint selection or scoring.

```powershell
python scripts/neural_ranker.py --stage options              # embed V2 B/validation answer options once (frozen encoder)
python scripts/neural_ranker.py --stage check                # Signal B check on half B
python scripts/neural_ranker.py --stage train --variant B    # reference loss
python scripts/neural_ranker.py --stage train --variant C    # reference loss + 0.25 x answer loss
python scripts/neural_ranker.py --stage validate             # A (saved LightGBM) vs B vs C on V2 validation
python scripts/neural_ranker.py --stage score --variant B    # V2 test, V1, LoCoMo scores for the frozen model
python scripts/evaluate_neural_ranker.py --variant B --datasets v2 v1 locomo
```

- **Model:** the 57 features (standardized on B, clipped to ±5) plus, through one shared 768→64 projection of the question and message vectors, their elementwise product and absolute difference; one 128-unit hidden layer gives one score per message. AdamW at 0.001 (weight decay 0.01), 16 complete histories per batch, five epochs, seed 271. The checkpoint with the best V2 validation NDCG@10 is kept (ties: every-reference coverage@15).
- **Reference loss:** softmax over the visible history; the mean of −log p over the question's references; every question weighs the same, `ask_to_forget` 0.5.
- **Answer loss (variant C):** each message's cosine to each answer option, minus that option's mean over the history, is averaged with the ranker's probabilities; a cross-entropy over options at the temperature set by the check (0.0553) rewards evidence favouring the correct option. It skips `ask_to_forget`, whose references favour a wrong option on half B. Answer texts are never ranker inputs.
- **`--stage check`:** compares the correct-versus-best-alternative margin of the annotated references with that of role- and length-matched messages from the same history (20 draws), by persona and question type. It writes `results/neural_ranker/signal_b_check.json`, and training C refuses to run unless the check passed.
- **Qwen test:** for each question, `evaluate_neural_ranker.py` takes the message count of `results/embedding_ab/transfer_cutoff/{dataset}_cut3_mem8192`. It accepts the neural ranker's highest-ranked whole messages that fit the 8,192-token cap until it has that many, and shows them chronologically. A question whose set equals the baseline's reuses the baseline's reads. The Qwen configuration must equal the baseline's.
- **Outputs:**
  - models: `models/neural_ranker/{B,C}.pt`;
  - records: `results/neural_ranker/training_{B,C}.json` and `validation.json`;
  - scores: `neural_{B,C}_scores.npy` next to each dataset's cached features;
  - Qwen runs: `results/neural_ranker/{dataset}_{variant}_matched/`;
  - comparisons: `results/neural_ranker/{dataset}_comparison_*.json`.

**Tuning (`neural_ranker_tune.py`):**

```powershell
python scripts/neural_ranker_tune.py --stage search                  # 21 settings, early stopping on a held-out half-B persona fold
python scripts/neural_ranker_tune.py --stage answer-weight           # best setting, answer-loss weight 0.1/0.25/0.5/1.0
python scripts/neural_ranker_tune.py --stage final                   # retrain both winners on all of B; score V2 validation once
python scripts/neural_ranker_tune.py --stage score --variant C_tuned # V2 test, V1, LoCoMo scores
```

- **Search:** tunes on persona fold 0 of half B (1,835 questions) and trains on the other four. Each setting runs up to 20 epochs and stops after 3 without a better tuning NDCG@10.
- **Final models:** retrained on all of half B for the chosen epoch count, without early stopping. V2 validation is read only by `final`, after every choice is made.
- **Outputs:**
  - tuning records: `results/neural_ranker/tuning_search.json`, `tuning_answer_weight.json`;
  - model records and validation: `training_{B,C}_tuned.json`, `validation_tuned.json`;
  - models: `models/neural_ranker/{B,C}_tuned.pt`.
- **Qwen:** `evaluate_neural_ranker.py` accepts only `B` and `C` today; add the tuned variants to its `--variant` choices before scoring them with Qwen.

## 7. Compare runs and maintain evaluation splits

### `compare_selector_runs.py`

**Purpose:** Read completed Qwen `predictions.jsonl` files and compare paired accuracy differences, gained/lost questions, and 95% bootstrap intervals clustered by persona/conversation. It also reports equal-dataset macro accuracy and pooled question accuracy. It performs no inference.

Create a JSON mapping with the same run labels for each dataset, using actual run directories. For example, save the following as `results/run_comparison.json` after producing the named runs:

```json
{
  "v2": {
    "references_only": "results/eval_refs_v2",
    "qwen_graded": "results/eval_graded_v2"
  },
  "v1": {
    "references_only": "results/eval_refs_v1",
    "qwen_graded": "results/eval_graded_v1"
  }
}
```

```powershell
python scripts/compare_selector_runs.py --runs results/run_comparison.json --output results/run_comparison_report.json --active-only
```

`--runs` and `--output` are required. Every mapped directory must contain predictions for the same question set as the other runs within that dataset. `--active-only` filters out IDs no longer present in the active split files, useful for historical V2 results saved before exclusions. `--draws` defaults to 5,000 and `--seed` to 271. Use at least two consistently named runs to obtain paired comparisons.

**Outputs:** The requested JSON report and printed comparisons. Existing output files are replaced. Paths in the mapping are resolved from the repository root; the command's argument paths follow the current working directory.

### `exclude_eval_questions.py`

**Purpose:** Perform the one-time removal of a question type (default `sensitive_info`) from V2 validation/test IDs and filter feature/score exports under selected roots. It preserves retained feature values and backs up the original NPZ. This is split maintenance, not a normal training step.

```powershell
# Inspect proposed changes on a checkout where exclusion has not been applied.
python scripts/exclude_eval_questions.py

# Apply to splits and the explicitly selected feature root.
python scripts/exclude_eval_questions.py --bundle-root data/ranker55 --apply
```

**Current checkout:** `data/splits/v2/evaluation_exclusions.json` already exists, so the script refuses to run again, including in dry-run mode. The examples describe its use before that exclusion has been applied; no repeat is needed here.

`--question-type` changes the removed type. `--bundle-root` accepts roots containing `v2/{val,test}/features.npz`; its default is **`data/ranker55/subtlex`**, so pass the intended roots explicitly. Without `--apply`, the script only reports planned changes.

**Changes with `--apply`:** Rewrites V2 `val_ids.json`, `test_ids.json`, and `persona_split.json`; writes `evaluation_exclusions.json`; filters matching NPZ/BM25/cosine outputs and updates their reports. Original bundles become `features_before_exclusion.npz`, and removed score rows become `{split}_excluded.jsonl`. Existing backups are not overwritten. Training, raw sources, saved Qwen predictions, and embedding caches are unchanged. Embedding cache provenance can become stale after split changes; rebuild caches as needed, and use `--active-only` when comparing older predictions.

## 8. Shared modules: import them from Python

These four files are helpers for the command-line scripts. Running them directly produces no task output; they have no `main()` entry point. Because sibling imports use the `scripts/` directory, add it to `sys.path` when importing helpers from a root-level Python session or notebook:

```python
import sys
sys.path.insert(0, "scripts")
```

### `raw_datasets.py`

**Purpose:** Read PersonaMem V2, PersonaMem V1, and LoCoMo-MC10 directly from their source files and preserved split IDs. It supplies deterministic answer options, answer targets, original histories, source hashes, and message IDs. Visibility helpers preserve the original V1 cutoff, including negative Python slice indices. Complete nonempty user/assistant messages form retrieval memories.

Example after the import setup above:

```python
from raw_datasets import load_dataset, memories_for_question

questions, contexts, targets, sources = load_dataset("v2", ["val"])
first_question = questions[0]
visible_memories = memories_for_question(first_question, contexts)
print(first_question["example_id"], len(visible_memories))
```

`include_answers=False` avoids parsing answers/options for retrieval; `include_contexts=False` avoids loading histories when they are not needed. This module writes no files. Expected source locations include `data/raw/personamem_v2/benchmark/text/{train,val,benchmark}.csv` and its linked histories, V1 `questions_32k.csv`/`shared_contexts_32k.jsonl`, and the JSONL-formatted `data/raw/locomo/locomo_mc10.json`.

### `message_dataset.py`

**Purpose:** Wrap the raw loader for retrieval, without reading answers. It attaches each question's visible `memory_ids`, deduplicates memory records by source ID, and returns questions, memories, and source hashes. Memory text stays exactly as in the original complete message. It also provides temporary-file-and-replace JSON/JSONL writers used by other scripts.

```python
from message_dataset import load_message_dataset

questions, memories, sources = load_message_dataset("v2", ["val"])
first_memory = memories[questions[0]["memory_ids"][0]]
print(first_memory["role"], first_memory["message_id"])
```

The loader creates no prepared dataset or intermediate manifest. It excludes system/persona, empty, and non-text turns from retrieval memories while preserving original indices and source message metadata.

### `subtlex_mixture.py`

**Purpose:** Validate the pinned prior and lexical configuration, tokenize consistently, and mix external token probabilities with counts from the current question's complete visible history. The mixture generates bounded lexical weights and caches a limited number of history-specific priors. It fits no training-corpus statistics and writes no files.

```python
from subtlex_mixture import ROOT, configuration, load_prior

counts, total, lock = load_prior(
    ROOT / "data/external/subtlex_us/word_frequencies.json",
    ROOT / "configs/subtlex_us.lock.json",
)
recipe = configuration(0.5, lock)
print(recipe["name"], total)
```

The builder uses `Mixture` and `HistoryPrior` to obtain history-local weights. Alpha 0 uses history probabilities; alpha 1 uses the smoothed external distribution. The normal CLI entry point for this functionality is `build_ranker_features.py --alpha VALUE`.

### `feature_contract.py`

**Purpose:** Read and validate a bundle's SUBTLEX recipe, locate embedding caches, and save/check the `.recipe.json` sidecar binding weights to their own hash and exact lexical recipe. Matching column names alone do not establish compatibility.

```python
from feature_contract import bundle_recipe, check_model_recipe

recipe = bundle_recipe("data/ranker55/v2/train/features.npz")
sidecar = check_model_recipe(
    "models/subtlex_mix/alpha_0.5/ranker55.txt", recipe
)
print(sidecar)
```

The check raises an error for a missing/stale sidecar, changed model hash, different alpha/source hashes, or incompatible recipe. Training scripts call `save_model_recipe()` when producing weights; keep those generated sidecars beside their models.

## 9. Supporting files and checks

| File or directory | Role |
| --- | --- |
| `configs/assets.lock.json` | Pinned model/runtime revisions, paths, sizes, and hashes; also records dataset assets |
| `configs/subtlex_us.lock.json` | Exact processed/raw SUBTLEX hashes and normalization contract |
| `configs/qwen_server.json` | Active native Qwen model/runtime paths, context capacities, port, and generation settings |
| `configs/evaluate_qwen.json` | Shared answer prompt, labels, grading policy, bootstrap settings, and model identity |
| `configs/project.json` | Project design/settings overview; individual CLI defaults and active runtime configs govern execution |
| `requirements.lock.txt` | Pinned Python dependencies; install LightGBM separately as described above |
| `data/raw/`, `data/splits/` | Original datasets and active cohort membership |
| `data/ranker55/` | Generated feature bundles, embeddings, and recipe manifests |
| `data/reference_labels/`, `data/label_candidates/`, `data/qwen_labels/` | Exact labels, selected candidate exports, and Qwen judgments respectively |
| `models/` | Pretrained assets, trained rankers, trial models, and recipe sidecars |
| `results/` | Evaluation predictions, diagnostics, and experiment reports |
| `tests/` | Unit tests and synthetic fixtures for loaders, features, labels, packing, runtime interfaces, and evaluation |
| `README.md`, `PROJECT_PLAN.md`, `PLAN_REVIEW.md` | Project overview, research plan, and plan review |

Quick checks without training or loading model weights:

```powershell
python scripts/ranker55.py --describe
python scripts/qwen_llamacpp_server.py --print-settings
python scripts/evaluate_qwen.py --dataset v2 --mode full_history --dry-run
```

To run the existing test suite:

```powershell
python -m unittest discover -s tests
```

For this guide, all 19 Python files were parsed, all 15 CLI `--help` entry points returned successfully, and the documented script flags and Python/JSON examples were checked. The ranker description, server-settings preview, full-history dry run, and selector dry run with the existing SUBTLEX test bundle passed. The default selector dry run exposed the older bundle mismatch documented above. Feature generation, training, downloads, and GPU inference were not executed as part of documentation verification.
