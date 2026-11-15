# Combined mathematical ranker experiment

## Result

All 1,119 active V2 test questions completed, on the RTX 5080. The combined
method improves four-order answer accuracy by **3.31 percentage points**,
with a paired persona-bootstrap 95% interval of **+1.16 to +5.62** (5,000 draws).
It gains 97 answers and loses 60 relative to the saved reference-only baseline,
for 37 more correct answers overall (509 versus 472).

| Scoring | Existing reference-only ranker | Both new features + MMR | Difference (persona 95%) |
| --- | ---: | ---: | --- |
| Single pass | 39.41% | 40.04% | +0.63 (-1.25 to +2.53) |
| Four cyclic answer orders | 42.18% | **45.49%** | **+3.31 (+1.16 to +5.62)** |

Before MMR, the new ranker's non-forget type-macro reference Hit@10 is **80.40%**
against **68.47%** for the existing model on this feature bundle. NDCG@10 is
**48.17%** against **39.19%**. The final combined selection recalls **71.43%** of
visible reference occurrences per question and includes every visible reference
on **53.35%** of questions. It selects 15.03 whole messages and 2,040 memory
tokens per question on average.

Training, including five held-out folds, took **126.5 seconds**. Test preparation,
four-order scoring and the answer report took **20.0 minutes**. These results
do not isolate the two new features from each other or from MMR. The single-pass
gain's interval includes zero; the four-order gain's interval excludes zero.

Four new mathematical/packing tests pass. The full suite runs 180 tests with one
existing failure, `test_deterministic_scoring_errors_are_not_retried` in
`test_qwen_judge.py`; it also fails in isolation (three launches versus one
expected), and that implementation was not modified. Test temporary files were
redirected into the workspace because the sandbox's default temp directory
denies cleanup operations. Evaluation required local-server socket access outside
the sandbox; the initial sandbox attempt scored no questions.

## Design

One fixed configuration, trained on V2 training personas and evaluated only on
the 1,119 active V2 test questions. No validation run, parameter sweep, or
test-based selection is performed. The existing reference-only ranker remains
the comparison baseline. This experiment measures the combined effect; it does
not identify the contribution of each component.

## Components and fixed settings

1. A residual bilinear score on frozen, normalized 768-dimensional cached vectors:
   `s(q,m) = q.T @ m + (q.T @ U) @ (m.T @ V).T`. U and V each have 16 columns.
   Pairwise softplus loss prefers one sampled annotated reference over seven
   samples from the top-30 non-reference cosine results plus one other sampled
   non-reference. Samples exclude every message with the same cached embedding
   identity as a reference. Unlisted messages remain noisy contrast examples,
   not proven irrelevant. AdamW, learning rate 0.003, weight decay 0.01,
   temperature 0.05, batch size 128, three epochs, seed 271. Forget questions
   retain weight 0.5.
2. A background-corrected cosine score: `cos(q,m) - 0.5 * b(m)`, where `b(m)` is
   the mean of the ten highest cosine similarities to a seeded bank of 1,024
   unique training-question embeddings. No encoder is retrained.
3. MMR selection with relevance min-max normalized within the full visible
   pool and a 0.1 penalty on maximum nonnegative cosine similarity to an already
   accepted message. Rejected oversize messages do not alter redundancy.
   Whole messages fill the 2,048-token budget using the existing exact native
   prompt-token accounting, then are presented in chronological order.

Five folds hold out 160 of the 800 training personas each. Both new training
features are computed using an adapter and question bank excluding the held
persona fold. A final adapter/bank uses all training personas for test inference.
The 57-feature LightGBM uses the established setting: 300 rounds, 100 minimum
child samples, 63 leaves, depth 6, L2 10, forget weight 0.5. It fits all 4,273,942
original training rows with binary reference-membership labels. No new feature
changes the complete original-message retrieval unit or input text.

## Reproduction and records

```powershell
.venv/Scripts/python.exe -B scripts/math_ranker_experiment.py
.venv/Scripts/python.exe -B scripts/evaluate_math_ranker.py
# Rebuild reports from completed predictions without inference:
.venv/Scripts/python.exe -B scripts/evaluate_math_ranker.py --summarize-only
```

The training command refuses to overwrite weights once test predictions exist.
Do not retrain or alter parameters after inspecting test performance and then
describe a rerun as an untouched test. Evaluation resumes only with the same
artifact, input, code, and configuration hashes.

- `settings.json`: settings frozen before fitting.
- `training.json`: persona folds, source/cache/model hashes, parameters, training time.
- `protocol.json`: test identities, model hashes, Qwen configuration and scoring code hash.
- `predictions.jsonl`: four cyclic answer orders, selected message identities and token counts.
- `report.json`: answer accuracy and paired persona-bootstrap comparison with the saved baseline.
- `retrieval.json`: post-inference reference diagnostics. References never enter test selection.

Training weights stay locally under `models/math_ranker/`; generated arrays stay
under the ignored `data/ranker55/math_ranker/` directory. Test predictions use
the current Qwen server, prompt format and four-order log-probability aggregation.
Order 0 also supplies the single-pass result. LoCoMo and V1 are not evaluated.
The exact training source matching the recorded hash is archived locally as
`models/math_ranker/training_source.py`; the working script has only a later
introductory-docstring clarification. Check logs are under `checks/`.
