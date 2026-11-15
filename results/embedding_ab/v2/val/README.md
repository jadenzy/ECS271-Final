# Persona-disjoint embedding/ranker experiment

## Results

All 909 active validation questions were evaluated. Both rankers include the
learned question/message similarity and generic-message correction features.

| Method | Hit@10 | NDCG@10 | Reference recall@15 | Every visible reference@15 |
| --- | ---: | ---: | ---: | ---: |
| Original embedding, cosine only | 45.21% | 25.35% | 37.19% | 22.33% |
| Original embedding + 57-feature ranker | 79.98% | 51.82% | 69.03% | 52.70% |
| Fine-tuned embedding, cosine only | 92.74% | 65.73% | 80.87% | 61.83% |
| **Fine-tuned embedding + 57-feature ranker** | **93.73%** | **77.70%** | **93.28%** | **87.90%** |

Against the matched original-embedding ranker, the fine-tuned ranker gains
**13.75 points Hit@10** (paired persona-bootstrap 95% **+11.02 to +16.54**),
**25.88 points NDCG@10** (+23.44 to +28.26), and **35.20 points all-reference
coverage@15** (+32.06 to +38.33). Bootstrap intervals use 5,000 draws over the
40 validation personas. These are validation retrieval results; this experiment
does not measure downstream answer accuracy or redundancy-aware selection.

The ranker adds relatively little to the fine-tuned embedding's any-reference
Hit@10 (+0.99 points), but substantially improves reference recall@15
(80.87% to 93.28%) and all-reference coverage@15 (61.83% to 87.90%). The old
6,000-question pilot and the current cached-original baseline use slightly
different numerical embedding builds; the table compares the actual bundles
used for this experiment rather than copying the earlier pilot's baseline.

Fine-tuning took **22.3 minutes**, including checkpoint saves. The mathematical
adapters and ranker fits took **66.4 seconds** for the original embedding and
**42.1 seconds** for the fine-tuned embedding, after feature construction.
Five new experiment tests pass; the full suite has 184 passes and the existing
`test_qwen_judge.JudgeTests.test_deterministic_scoring_errors_are_not_retried`
failure (three launches versus one expected). Source, fold, model, feature and
score integrity checks are recorded in `verification.json`.

## Cohorts and training

The V2 training personas are divided once, with seed 271, into A and B:

| Cohort | Personas | Questions | Use |
| --- | ---: | ---: | --- |
| A | 400 | 9,340 | Exactly 9,000 seeded questions fine-tune the encoder |
| B | 400 | 9,192 | Train both comparison rankers |
| Validation | 40 | 909 | Four-way retrieval comparison |

No test evaluation is run. Fine-tuning uses the earlier pilot's settings:
one epoch, batch size 12, learning rate 2e-5, temperature 0.05, two sampled
references, five hard and one random non-reference per question, plus in-batch
contrast examples. Exact reference texts are masked out of the negative pool.
Training inputs may truncate at 256 tokens, as in the earlier pilot; feature
construction never truncates original messages and enforces the 1,200-token cap.
Non-reference annotations are not proof of irrelevance, so contrast examples
remain noisy negatives. There is no early stopping or intermediate validation
evaluation in this run. Optimizer/model/RNG checkpoints are saved every 125 steps.
The final encoder and tokenizer are saved locally with content hashes.

## Requested comparison

1. Original frozen embedding, whole-message cosine only.
2. Original embedding plus a 57-feature ranker fitted on B.
3. A-fine-tuned embedding, whole-message cosine only.
4. A-fine-tuned embedding plus a 57-feature ranker fitted on B.

Both rankers use the same B questions, binary reference-membership targets,
fixed LightGBM settings and two new mathematical features: learned residual
bilinear question/message similarity and generic-message correction. Each
condition trains its small similarity adapter with five persona-held-out folds
within B to generate the ranker's training features. Its final B adapter and B
question bank produce validation features. No A persona or validation persona
enters either B ranker or the B mathematical adapter training. The encoder itself
is fine-tuned only on A. Redundancy-aware selection is deferred, and no MMR or
Qwen answer evaluation is performed in this comparison.

The new features follow the earlier combined pilot's fixed settings: rank 16,
three adapter epochs, pairwise softplus loss, temperature 0.05, learning rate
0.003, a 1,024-question background bank and top-10 background mean with penalty
0.5. LightGBM uses 300 rounds, 100 minimum child samples, 63 leaves, depth 6,
L2 10 and forget weight 0.5. Existing lexical recipes, SUBTLEX alpha 0.5, message
text, chunk windows and all 55 original feature definitions are preserved.

The original B feature rows are copied verbatim from the existing bundle.
Fine-tuned B and validation message, query and chunk embeddings are rebuilt
using the saved local encoder. Encoder identity uses a content hash distinct
from the pinned original model; the default builder remains pinned. Before
comparison, group identities, target rows and every embedding-independent
feature are checked for equality between conditions.

## Reproduce and inspect

```powershell
.venv/Scripts/python.exe -u -B scripts/embedding_ab_experiment.py --stage all
```

Stages `prepare`, `finetune`, `baseline`, `features` and `compare` can be run
separately. Fine-tuning resumes from its checkpoint; embedding builds resume
from completion masks. Completed stages are reused with identity/hash checks.
The prepared partition is never silently replaced by a different one.

- `partition.json`, `a_finetune_ids.json`, `b_ranker_ids.json`: fixed cohorts.
- `encoder.json`: saved encoder identity, file hashes, training command and timing.
- `original_ranker57.json`, `finetuned_ranker57.json`: B folds, mathematical settings, model and feature hashes.
- `report.json`, `REPORT.md`, `per_question.jsonl`: the four-way validation metrics and paired persona intervals.
- `finetune.log`: live fine-tuning output; later stage output is appended here for the visible terminal.
- `tests.log`: the small experiment-specific test suite.

The saved encoder and trained rankers are under ignored `models/embedding_ab/`.
Rebuilt feature bundles are under ignored `data/ranker55/embedding_ab/`.
Mathematical feature and score arrays stay alongside the ignored trained rankers.
An intermediate 55-feature original-B control was fitted before the request to
include both new features; it is not one of the four reported methods.
