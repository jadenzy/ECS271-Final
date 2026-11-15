# Single-pass reader policy comparison on V2 validation

Completed on November 15, 2026 (America/Los_Angeles): **909 questions, 40
personas, three policies, 2,727 answer evaluations**. Every policy uses exactly
one answer order, as requested. Scoring took **949.1 seconds (15.8 minutes)**.

| Policy | Correct | Answer accuracy | Mean messages | Mean memory tokens |
| --- | ---: | ---: | ---: | ---: |
| Current ranker selection | 394/909 | 43.34% | 16.97 | 2,040.6 |
| Ranker top-8 | 406/909 | **44.66%** | 8.00 | 1,155.0 |
| Qwen-judge top-8 | 382/909 | 42.02% | 8.00 | 1,023.0 |

Paired 95% intervals resample the 40 validation personas with 5,000 draws,
seed 271. Differences below are percentage points.

| Comparison | Difference | Paired 95% interval | Gained / lost answers |
| --- | ---: | --- | ---: |
| Ranker top-8 minus current | +1.32 | -1.11 to +3.77 | 61 / 49 |
| Judge top-8 minus current | -1.32 | -3.75 to +1.12 | 43 / 55 |
| Judge top-8 minus ranker top-8 | -2.64 | -4.67 to -0.67 | 42 / 66 |

The judge does not justify adding a filtering stage in this experiment: its
top-8 answers are worse than the ranker's top-8. Ranker top-8 is a promising
efficiency candidate, using **43.40% fewer memory tokens** than the current
selection, but its answer-accuracy improvement over current selection is
inconclusive. The default selector remains unchanged. These are validation,
single-pass results; they are not the earlier four-order V2 test results.

## Fixed comparison

- Reuse the saved fine-tuned 57-feature ranker's validation selections and the
  existing Qwen per-message judgments for all 909 questions. No retraining or
  new message judging is performed.
- Current selection is the entire saved, score-packed 2,048-token selection.
  Ranker top-8 takes the first eight from that selection, in ranker score order.
  Judge top-8 takes the eight largest useful-minus-not-useful log-probability
  margins from the same selection. If fewer than eight exist, retain all.
- Both top-8 policies have identical message counts per question; they do not
  have identical token counts. Whole messages are shown chronologically with
  their original source labels.
- Frozen Qwen3.5-4B Q4_K_M, existing `current` prompt, original answer order,
  same server, and the same maximum 2,048-token memory budget for every arm.
  No MMR or label-bias correction. Each arm is scored fresh in this run.
- Correct answers and reference annotations do not select messages. Correct
  answers grade model output; reference annotations supply coverage diagnostics.
- `protocol.json` freezes the question/pool content, configuration, saved
  selections, judgments, scores and relevant evaluation source hashes. Resume
  rejects mixed protocols, duplicate questions and different answer-order counts.

## Reference coverage

These diagnostics describe annotated references, not final-answer accuracy.

| Policy | Any reference | Every reference | Mean reference recall | Mean reference share of selection |
| --- | ---: | ---: | ---: | ---: |
| Current selection | 95.71% | 87.79% | 92.37% | 17.19% |
| Ranker top-8 | 91.64% | 77.89% | 85.63% | 32.80% |
| Judge top-8 | 87.35% | 46.53% | 69.13% | 26.23% |

The ranker's eight-message choice preserves more annotated evidence than the
judge's at the same message count. Smaller selections trade coverage for shorter
prompts; higher coverage alone does not establish better answer accuracy.

## Verification and artifacts

Three new runner tests and three existing transfer tests passed. Independent
checks confirmed complete unique cohorts, one score per policy/question, gold
and answer-score agreement, exact frozen ranker scores, policy membership,
native memory budgets, zero prompt truncations, and unchanged protocol hashes.
The full repository test suite was not rerun for this isolated runner.

- `report.json`: answer accuracies, paired intervals and prompt/memory counts.
- `coverage.json`: reference-coverage diagnostics for the same three policies.
- `verification.json`: completed integrity checks.
- `runtime.json`, `run.log`, `qwen-server.log`: timing and live execution records.
- Each policy folder: all 909 predictions with one shift per question.

The older prompt-wording experiment remains incomplete at 255 questions per
arm and is not merged into these results. On those overlapping questions,
the fresh current-selection run preserves all 255 message sets and prompt-token
counts; 240/255 single-pass answers agree. The earlier files lack full prompt
hashes, so this check does not establish the cause of the answer differences.
See `prior_partial_run_check.json`.

```powershell
.venv/Scripts/python.exe -u -B scripts/reader_one_pass.py
.venv/Scripts/python.exe -B scripts/reader_one_pass.py --check-only
.venv/Scripts/python.exe -B scripts/reader_one_pass.py --summarize-only
```
