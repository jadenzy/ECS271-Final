"""Graded two-pass selector: Qwen 0/1/2 labels, references first, then Qwen-useful messages.

python scripts/evaluate_graded_selector.py --train
python scripts/evaluate_graded_selector.py --datasets v1 locomo v2 --threshold 0.25
python scripts/evaluate_graded_selector.py --datasets v1 --threshold 0.25 --summarize-only
python scripts/evaluate_graded_selector.py --datasets v1 locomo v2 --threshold 0.4 --route-confidence 0.4

With --route-confidence c, each question gets the selector's set only when the selector is confident it holds
every reference: the product of 1 - P(2) over the candidates it leaves out is at least c (an independence
estimate; on V2 validation it understates the true rate). Otherwise the question gets the frozen fine-tuned
ranker's 3.0-cutoff set under the same cap (evaluate_embedding_ab_cutoff.py's packing). Tight sets go only
where the selector is sure; uncertain questions keep the broad set.

python scripts/evaluate_graded_selector.py --datasets v1 locomo v2 --threshold 0.4 --route-confidence 0.4 --two-view

With --two-view, cutoff-route questions are read twice, with the broad cutoff set and with the selector's tight
set, and answered by the sum of both reads' renormalized letter log-probabilities per answer order (equal
weights). Fixed on V2 validation (reader_validation.py's selector_current and filter_cut3_current arms) before
any test run. Reads the routed run already made are reused where the route and selected messages match exactly.
Result: V2 test 55.41% against the routed rule's 56.48% (the tight-set read is weaker on test), so it is not the default.

Same inputs, persona folds, 57 features, pool features and packing as evaluate_selector.py, but
both passes are three-class LightGBM models trained on V2 persona half B's graded labels, with
build_graded_training.py's --all-rows policy:
  2  annotated reference
  1  not a reference, judged useful by Qwen (one of 20 judged candidates per question)
  0  judged not useful, or never judged
Messages are ranked and kept by expected usefulness u = P(2) + 0.5 * P(1): the ranker's 0/1/2
label gain scaled to 1. A likely reference scores up to 1 and a likely Qwen-useful message up to
0.5, so references are kept and packed first and Qwen-useful messages next. Pass 1 scores every
visible message; pass 2 rescores pass 1's top 30 by u with the pool features (computed on
logit(u)), pass 1's P(1) and P(2), and the 57 features again. Pass 2 is trained on out-of-fold
pass 1 outputs; a second out-of-fold pass 2 measures, on B, how many grade 2 and grade 1 messages
each threshold keeps (validation has references but no Qwen grades). Ranker scores are not used.

Models: models/embedding_ab/selector_graded/. Record: results/selector_graded/training.json.
Runs: results/selector_graded/{dataset}_t{threshold}/ with the same files as evaluate_selector.py.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import build_ranker_features as builder
from embedding_ab_experiment import MODELS, OUTPUT as TRAINING_RECORDS
from embedding_ab_math import CachedInputs
from evaluate_embedding_ab_cutoff import CutoffHistories
from evaluate_embedding_ab_transfer import CACHE, DATASETS, MODEL_ROOT, accept_protocol, answer_scores, frozen_models
from evaluate_qwen import (build_messages, evaluation_config, order_for_cache, pack_selector_messages, runtime_metadata,
                           selected_history)
from option_rotation_pilot import summarize
from evaluate_selector import (FEATURES, FOLDS, MEMORY_TOKENS, PASS2_PARAMETERS, POOL_NAMES, SEED, THREADS, TOP,
                               SelectorHistories, copy_bundle, kept_mask, pass1_parameters, pool_features,
                               reference_mask, report)
from math_ranker_experiment import NAMES
from message_dataset import load_message_dataset
from option_rotation_pilot import rotated
import ranker55 as recipe
from raw_datasets import ROOT, load_dataset, sha256
from tune_ranker55 import write_json
import qwen_llamacpp_server as qwen_server

QWEN_LABELS = ROOT / 'data/qwen_labels/v2/train.jsonl'
SELECTOR = MODELS / 'selector_graded'
OUTPUT = ROOT / 'results/selector_graded'
GAIN = np.array([0., .5, 1.])
PASS2_NAMES = POOL_NAMES + ['pass1_p_useful', 'pass1_p_reference'] + NAMES
THRESHOLDS = (.6, .5, .45, .4, .35, .3, .25, .2, .15, .1, .07, .05)
CACHE_NAME = 'selector_graded_usefulness.npy'          # V2 validation usefulness written by --train
PROBABILITY_CACHE = 'selector_graded_probabilities.npy'
ROUTE_CUTOFF = 3.0


def usefulness(P):
    return np.asarray(P) @ GAIN


def logit(u):
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return np.log(u / (1 - u))


def pass1_multiclass():
    return dict(pass1_parameters(), objective='multiclass')


PASS2_MULTICLASS = dict(PASS2_PARAMETERS, objective='multiclass')


def pass2_inputs(P1, X, offsets, indices):
    rows, pool = pool_features(logit(usefulness(P1)), offsets, indices)
    return rows, np.column_stack([pool, P1[rows, 1:], X[rows]]).astype(np.float32)


def class_probabilities(b1, b2, X, offsets, indices):
    """Pass 2 P(0), P(1), P(2) per row; rows outside pass 1's top TOP are (1, 0, 0)."""
    P1 = b1.predict(X, num_threads=THREADS)
    rows, Z = pass2_inputs(P1, X, offsets, indices)
    P = np.zeros((len(X), 3))
    P[:, 0] = 1
    P[rows] = b2.predict(Z, num_threads=THREADS)
    return P


def predict(b1, b2, X, offsets, indices):
    return usefulness(class_probabilities(b1, b2, X, offsets, indices))


def graded_labels(tr, train_path):
    """0/1/2 per half-B row from the Qwen judgments; unjudged messages are 0 (cached)."""
    cache, record_path = train_path.parent / 'graded_labels.npy', train_path.parent / 'graded_labels.json'
    expected = dict(qwen_labels_sha256=sha256(QWEN_LABELS), features_sha256=sha256(train_path),
                    policy='2 reference; 1 Qwen-judged useful; 0 Qwen-judged not useful or unjudged')
    if cache.exists() and record_path.exists():
        saved = json.loads(record_path.read_text())
        if {k: saved.get(k) for k in expected} == expected and saved.get('labels_sha256') == sha256(cache):
            return np.load(cache)
    offsets, mids = tr['offsets'], tr['message_ids']
    position = {str(q): i for i, q in enumerate(tr['question_ids'].tolist())}
    y = np.full(len(mids), -9, dtype=np.int8)
    seen = set()
    with QWEN_LABELS.open('rb') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            i = position.get(row['example_id'])
            if i is None:
                continue                      # persona half A: used to fine-tune the encoder, not here
            a, b = int(offsets[i]), int(offsets[i + 1])
            memories = row['memories']
            if [m['message_id'] for m in memories] != mids[a:b].tolist():
                raise ValueError(f"{row['example_id']}: judged memories differ from the half-B feature rows")
            grades = np.array([m['label'] for m in memories], dtype=np.int8)
            if not np.array_equal(grades == 2, tr['y'][a:b] == 1):
                raise ValueError(f"{row['example_id']}: grade 2 must mark exactly the references")
            y[a:b] = np.maximum(grades, 0)
            seen.add(row['example_id'])
    if len(seen) != len(position) or (y < 0).any():
        raise ValueError(f'{len(position) - len(seen)} half-B questions have no Qwen judgments')
    np.save(cache, y)
    write_json(record_path, dict(expected, labels_sha256=sha256(cache),
                                 counts={str(g): int((y == g).sum()) for g in (0, 1, 2)}))
    return y


def b_coverage(u, y, offsets):
    """Out-of-fold, on half B: references fully kept, share of Qwen-useful messages kept, size."""
    table = []
    for t in THRESHOLDS:
        keep = kept_mask(u, offsets, t)
        every, useful_kept, useful_total, n = [], 0, 0, []
        for i in range(len(offsets) - 1):
            a, b = offsets[i], offsets[i + 1]
            k, g = keep[a:b], y[a:b]
            every.append(bool((k | (g != 2)).all()))
            useful_kept += int((k & (g == 1)).sum())
            useful_total += int((g == 1).sum())
            n.append(int(k.sum()))
        table.append(dict(threshold=t, every_reference=float(np.mean(every)), qwen_useful_kept=useful_kept / useful_total,
                          mean_messages=float(np.mean(n))))
    return table


def validation_coverage(u, ref, offsets):
    table = []
    has = [i for i in range(len(offsets) - 1) if ref[offsets[i]:offsets[i + 1]].any()]
    for t in THRESHOLDS:
        keep = kept_mask(u, offsets, t)
        every, exact, n = [], [], []
        for i in has:
            a, b = offsets[i], offsets[i + 1]
            k, r = keep[a:b], ref[a:b]
            e = bool((k | ~r).all())
            every.append(e)
            exact.append(e and not (k & ~r).any())
            n.append(int(k.sum()))
        table.append(dict(threshold=t, every_reference=float(np.mean(every)), exact_set=float(np.mean(exact)),
                          mean_messages=float(np.mean(n))))
    return table


def train():
    import lightgbm as lgb
    started = time.perf_counter()
    ranker_record = json.loads((TRAINING_RECORDS / 'finetuned_ranker57.json').read_text())
    train_path, val_path = FEATURES / 'train/features.npz', FEATURES / 'val/features.npz'
    if sha256(train_path) != ranker_record['training_features_sha256'] or sha256(val_path) != ranker_record['validation_features_sha256']:
        raise ValueError('B or validation features differ from those the fine-tuned ranker was trained on')
    tr, va = copy_bundle(train_path), copy_bundle(val_path)
    if tr['question_ids'].tolist() != ranker_record['question_ids']:
        raise ValueError('Training questions differ from persona half B')
    if set(tr['persona_ids'].tolist()) & set(va['persona_ids'].tolist()):
        raise ValueError('B and validation personas overlap')
    y = graded_labels(tr, train_path)
    Xtr = np.column_stack([tr['X'], np.load(MODEL_ROOT / 'train_extra.npy')]).astype(np.float32)
    Xva = np.column_stack([va['X'], np.load(MODEL_ROOT / 'validation_extra.npy')]).astype(np.float32)
    questions, _, _ = load_message_dataset('v2', ['train'], root=ROOT)
    qtype = {q['example_id']: q['question_type'] for q in questions}
    weights = recipe.row_weights([.5 if qtype[str(q)] == 'ask_to_forget' else 1. for q in tr['question_ids'].tolist()], tr['offsets'])
    counts = {g: int((y == g).sum()) for g in (0, 1, 2)}
    print(f'B: {len(tr["offsets"]) - 1} questions, {len(y):,} messages; grade 2 {counts[2]:,}, grade 1 {counts[1]:,}', flush=True)

    personas = np.unique(tr['persona_ids'])
    fold_of = dict(zip(np.random.default_rng(SEED).permutation(personas).tolist(), np.arange(len(personas)) % FOLDS))
    rfold = np.repeat(np.array([fold_of[p] for p in tr['persona_ids'].tolist()]), np.diff(tr['offsets']))
    P_oof = np.full((len(y), 3), np.nan)
    for f in range(FOLDS):
        fit = rfold != f
        model = lgb.LGBMClassifier(**pass1_multiclass()).fit(Xtr[fit], y[fit], sample_weight=weights[fit], feature_name=NAMES)
        P_oof[~fit] = model.booster_.predict(Xtr[~fit], num_threads=THREADS)
        print(f'pass 1 fold {f + 1}/{FOLDS} done', flush=True)
    pass1 = lgb.LGBMClassifier(**pass1_multiclass()).fit(Xtr, y, sample_weight=weights, feature_name=NAMES)
    rows, Z = pass2_inputs(P_oof, Xtr, tr['offsets'], tr['message_indices'])
    inside = {g: int((y[rows] == g).sum()) for g in (1, 2)}
    print(f'pass 2 rows: {len(rows):,}; inside pass 1 top {TOP}: grade 2 {inside[2]:,} of {counts[2]:,}, '
          f'grade 1 {inside[1]:,} of {counts[1]:,}', flush=True)
    pass2 = lgb.LGBMClassifier(**PASS2_MULTICLASS).fit(Z, y[rows], sample_weight=weights[rows], feature_name=PASS2_NAMES)
    # Out-of-fold pass 2 on B, only to measure grade 2 / grade 1 coverage per threshold.
    u_oof = np.zeros(len(y))
    zfold = rfold[rows]
    for f in range(FOLDS):
        m = lgb.LGBMClassifier(**PASS2_MULTICLASS).fit(Z[zfold != f], y[rows][zfold != f], sample_weight=weights[rows][zfold != f])
        u_oof[rows[zfold == f]] = usefulness(m.booster_.predict(Z[zfold == f], num_threads=THREADS))
    b_table = b_coverage(u_oof, y, tr['offsets'])

    SELECTOR.mkdir(parents=True, exist_ok=True)
    pass1.booster_.save_model(str(SELECTOR / 'pass1.txt'))
    pass2.booster_.save_model(str(SELECTOR / 'pass2.txt'))
    b1, b2 = lgb.Booster(model_file=str(SELECTOR / 'pass1.txt')), lgb.Booster(model_file=str(SELECTOR / 'pass2.txt'))
    u = predict(b1, b2, Xva, va['offsets'], va['message_indices'])
    np.save(val_path.parent / CACHE_NAME, u)
    v_table = validation_coverage(u, reference_mask('val', va), va['offsets'])
    gain = b2.feature_importance('gain')
    record = dict(pass1_sha256=sha256(SELECTOR / 'pass1.txt'), pass2_sha256=sha256(SELECTOR / 'pass2.txt'),
                  training_features_sha256=sha256(train_path), training_extra_sha256=sha256(MODEL_ROOT / 'train_extra.npy'),
                  graded_labels_sha256=sha256(train_path.parent / 'graded_labels.npy'), qwen_labels_sha256=sha256(QWEN_LABELS),
                  validation_features_sha256=sha256(val_path), validation_extra_sha256=sha256(MODEL_ROOT / 'validation_extra.npy'),
                  validation_usefulness_sha256=sha256(val_path.parent / CACHE_NAME),
                  label_policy='2 reference; 1 Qwen-judged useful; 0 Qwen-judged not useful or unjudged',
                  grade_counts={str(g): c for g, c in counts.items()}, usefulness='P(2) + 0.5 * P(1)',
                  training_questions=len(tr['offsets']) - 1, training_personas=personas.tolist(),
                  folds={p: int(f) for p, f in fold_of.items()}, top=TOP, seed=SEED,
                  pass1_parameters=pass1_multiclass(), pass2_parameters=PASS2_MULTICLASS,
                  pass1_features=NAMES, pass2_features=PASS2_NAMES, ask_to_forget_weight=.5,
                  inside_pass1_top={str(g): c for g, c in inside.items()},
                  pass2_top_features_by_gain={PASS2_NAMES[j]: float(gain[j] / gain.sum()) for j in np.argsort(-gain)[:10]},
                  half_b_out_of_fold_coverage=b_table, validation_coverage=v_table, ranker_scores_used=False,
                  script_sha256=sha256(Path(__file__)), seconds=time.perf_counter() - started)
    write_json(OUTPUT / 'training.json', record)
    print('threshold | half B out-of-fold: every ref, Qwen-useful kept, msgs | V2 validation: every ref, exact set, msgs')
    for b, v in zip(b_table, v_table):
        print(f"  {b['threshold']:<5g}  {b['every_reference']:>6.1%} {b['qwen_useful_kept']:>6.1%} {b['mean_messages']:>5.1f}"
              f"   |  {v['every_reference']:>6.1%} {v['exact_set']:>6.1%} {v['mean_messages']:>5.1f}")


def selector_identity():
    record_path = OUTPUT / 'training.json'
    record = json.loads(record_path.read_text())
    for name, key in [('pass1.txt', 'pass1_sha256'), ('pass2.txt', 'pass2_sha256')]:
        if sha256(SELECTOR / name) != record[key]:
            raise ValueError('Graded selector model changed since training')
    return dict(pass1_sha256=record['pass1_sha256'], pass2_sha256=record['pass2_sha256'],
                training_record_sha256=sha256(record_path), top=TOP, usefulness='P(2) + 0.5 * P(1)', ranker_scores_used=False)


def probabilities(dataset, data, path, frozen):
    """Pass 2 class probabilities (rows x 3) for a dataset, cached; checks the 57 inputs by reproducing the frozen
    ranker's scores."""
    import lightgbm as lgb
    identity = selector_identity()
    cache, record_path = path.parent / PROBABILITY_CACHE, path.parent / 'selector_graded_scoring.json'
    extra_path = path.parent / 'mathematical_features.npy'
    expected = dict(identity, features_sha256=sha256(path), mathematical_features_sha256=sha256(extra_path))
    if cache.exists() and record_path.exists():
        saved = json.loads(record_path.read_text())
        if {k: saved.get(k) for k in expected} == expected and saved.get('probabilities_sha256') == sha256(cache):
            return np.load(cache)
    X = np.column_stack([np.asarray(data.X), np.load(extra_path)]).astype(np.float32)
    if sha256(MODEL_ROOT / 'ranker57.txt') != frozen['ranker_sha256']:
        raise ValueError('Frozen ranker changed')
    check = lgb.Booster(model_file=str(MODEL_ROOT / 'ranker57.txt')).predict(X, num_threads=THREADS)
    error = float(np.max(np.abs(check - np.load(path.parent / 'ranker_scores.npy'))))
    if error > 1e-6:
        raise ValueError(f'57 inputs do not reproduce the cached ranker scores (max error {error})')
    b1, b2 = lgb.Booster(model_file=str(SELECTOR / 'pass1.txt')), lgb.Booster(model_file=str(SELECTOR / 'pass2.txt'))
    P = class_probabilities(b1, b2, X, np.asarray(data.offsets), np.asarray(data.bundle.read('message_indices')))
    np.save(cache, P)
    write_json(record_path, dict(expected, probabilities_sha256=sha256(cache), ranker_reproduction_max_error=error,
                                 uses_evaluation_labels_or_answers=False))
    print(f'{dataset}: graded selector probabilities cached ({len(P):,} messages; ranker inputs reproduced to {error:.1e})', flush=True)
    return P


class RoutedHistories(CutoffHistories):
    """Tight graded-selector set when it is confident it holds every reference, else the ranker's 3.0-cutoff set."""
    def __init__(self, questions, contexts, data, ranker_scores, P, threshold, confidence, budget):
        super().__init__(questions, contexts, data, ranker_scores, ROUTE_CUTOFF, budget)
        if P.shape != (len(ranker_scores), 3) or not np.isfinite(P).all():
            raise ValueError('Invalid selector probabilities')
        self.P, self.threshold, self.confidence = P, threshold, confidence

    def tight(self, question):
        """The selector's tight set (u >= threshold, the most useful always kept) and its confidence."""
        i = self.groups[question['example_id']]
        a, b = map(int, self.data.offsets[i:i+2])
        P = self.P[a:b]
        u = usefulness(P)
        ranked = np.argsort(-u, kind='stable')
        tight = [int(j) for j in ranked if u[j] >= self.threshold or j == ranked[0]]
        outside = np.ones(b - a, dtype=bool)
        outside[tight] = False
        return u, tight, float(np.prod(1 - P[outside, 2]))

    def tight_view(self, question, server, config):
        key = question['history_id'], question['source_end_index']
        members = self.pools[key]
        u, tight, _ = self.tight(question)
        chosen, packing = pack_selector_messages(server, question, config, members, tight, self.budget, blocks=self.blocks[key])
        history = selected_history(members, chosen)
        return history, dict(**packing, selected_memory_count=len(history),
                             selected_messages=[dict(message_id=members[j]['message_id'], message_index=members[j]['message_index'],
                                                     usefulness=float(u[j])) for j in chosen])

    def get(self, question, server, config):
        _, tight, confidence = self.tight(question)
        route = dict(route='selector' if confidence >= self.confidence else 'cutoff', route_confidence=confidence,
                     selector_set_size=len(tight))
        history, selection = (self.get_capped if route['route'] == 'cutoff' else self.tight_view)(question, server, config)
        return history, dict(selection, **route)


def combine_views(views, labels):
    """Two-view answer: per answer order, the sum of each view's renormalized letter log-probabilities (equal weights)."""
    combined = []
    for s in range(len(views[0])):
        shift = views[0][s]['shift']
        conditional = dict.fromkeys(labels, 0.0)
        for view in views:
            x = np.array([view[s]['label_conditional_logprobs'][label] for label in labels], dtype=float)
            for label, value in zip(labels, x - np.logaddexp.reduce(x)):
                conditional[label] += float(value)
        combined.append(dict(shift=shift, predicted_label=max(labels, key=conditional.get), label_conditional_logprobs=conditional,
                             answer_logprobs=answer_scores(conditional, labels, shift),
                             prompt_tokens=sum(view[s]['prompt_tokens'] for view in views),
                             inference_seconds=sum(view[s]['inference_seconds'] for view in views), timings=None))
    return combined


def routed_report(out, dataset, rows, labels, threshold, confidence, budget):
    split, rotations = DATASETS[dataset]
    result = summarize(rows, labels) if rotations > 1 else dict(
        questions=len(rows), accuracy={'one_pass': float(np.mean([r['correct'] for r in rows]))})
    tight = np.array([r['route'] == 'selector' for r in rows])
    sizes = np.array([r['selected_memory_count'] for r in rows])
    result.update(status='complete', dataset=dataset, split=split, rotations=rotations, memory_tokens=budget, threshold=threshold,
                  route_confidence=confidence, score_cutoff=ROUTE_CUTOFF, share_routed_to_selector=float(tight.mean()),
                  mean_selected_messages=float(sizes.mean()),
                  mean_messages_selector_route=float(sizes[tight].mean()) if tight.any() else None,
                  mean_messages_cutoff_route=float(sizes[~tight].mean()) if (~tight).any() else None,
                  mean_memory_tokens=float(np.mean([r['packed_memory_tokens'] for r in rows])),
                  share_questions_capped=float(np.mean([r['budget_skipped_messages'] > 0 for r in rows])))
    if any('two_view' in r for r in rows):
        two = np.array([r['two_view'] for r in rows])
        result.update(share_two_view=float(two.mean()), share_reused_routed_reads=float(np.mean([r['reused_routed_read'] for r in rows])),
                      mean_messages_second_view=float(np.mean([r['selector_view']['selected_memory_count'] for r in rows if r['two_view']]))
                      if two.any() else None)
    write_json(out / 'report.json', result)
    print(f'COMPLETE {dataset}: {result["questions"]} questions; {result["accuracy"]}; '
          f'{tight.mean():.0%} routed to the selector', flush=True)


def score(dataset, threshold, summarize_only=False, budget=MEMORY_TOKENS, confidence=None, two_view=False):
    started = time.perf_counter()
    _, record, frozen = frozen_models()
    split, rotations = DATASETS[dataset]
    suffix = '' if budget == MEMORY_TOKENS else f'_mem{budget}'
    name = (f'{dataset}_t{threshold:g}' if confidence is None
            else f'{dataset}_{"twoview" if two_view else "routed"}_t{threshold:g}_c{confidence:g}')
    out = OUTPUT / f'{name}{suffix}'
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / 'predictions.jsonl'
    rows = [json.loads(line) for line in predictions.open(encoding='utf-8') if line.strip()] if predictions.exists() else []
    config = evaluation_config(dataset, 'selector')
    labels = config['answer_labels']
    if summarize_only:
        if confidence is None:
            return report(out, dataset, rows, labels, threshold, budget)
        return routed_report(out, dataset, rows, labels, threshold, confidence, budget)
    questions, contexts, targets, sources = load_dataset(dataset, [split], root=ROOT)
    questions = order_for_cache(questions)
    path = CACHE / dataset / split / 'features.npz'
    scoring = json.loads((path.parent / 'scoring.json').read_text())
    if scoring['adapter_sha256'] != frozen['adapter_sha256'] or scoring['features_sha256'] != sha256(path):
        raise ValueError('Feature cache differs from the frozen adapter')
    if confidence is not None and (scoring['ranker_sha256'] != frozen['ranker_sha256']
                                   or scoring['scores_sha256'] != sha256(path.parent / 'ranker_scores.npy')):
        raise ValueError('Ranker score cache differs from the frozen ranker')
    data = CachedInputs(path)
    if data.meta['embedding_recipe'] != record['embedding_recipe']:
        raise ValueError('Feature encoder differs from the trained models')
    for source, expected in sources.items():
        if data.meta['source_sha256'].get(source) != expected:
            raise ValueError('Current raw source or split differs from the feature cache')
    P = probabilities(dataset, data, path, frozen)
    if confidence is None:
        histories = SelectorHistories(questions, contexts, data, usefulness(P), threshold, budget)
    else:
        ranker_scores = np.load(path.parent / 'ranker_scores.npy', mmap_mode='r')
        histories = RoutedHistories(questions, contexts, data, ranker_scores, P, threshold, confidence, budget)
    if any([o['label'] for o in q['options']] != labels for q in questions):
        raise ValueError('Raw option labels differ from the configured scorer')
    settings = runtime_metadata(dataset, 'selector', memory_tokens=budget)
    config.update(memory_tokens=budget, input_policy='raw_complete_messages_v1', server_settings=settings,
                  model=settings['Model'], context_tokens=settings['ContextTokens'])
    policy = (f'graded two-pass selector: keep messages with usefulness P(2) + 0.5 P(1) >= {threshold:g} '
              f'(the most useful always kept), packed by usefulness under a {budget:,}-token cap; '
              'whole messages, chronological display; ranker scores not used')
    if confidence is not None:
        policy = (f'routed: the graded selector set (usefulness >= {threshold:g}, the most useful always kept) when the product of '
                  f'1 - P(2) over the messages it leaves out is >= {confidence:g}; otherwise the frozen ranker\'s greedy rank-order '
                  f'packing of messages within {ROUTE_CUTOFF:g} of the top score; {budget:,}-token cap; whole messages, chronological display')
    protocol = dict(dataset=dataset, split=split, rotations=rotations, config=config,
                    frozen_features=dict(encoder=frozen['encoder'], adapter_sha256=frozen['adapter_sha256'], feature_names=NAMES),
                    selector=selector_identity(), threshold=threshold,
                    selector_scoring_sha256=sha256(path.parent / 'selector_graded_scoring.json'), raw_sources=sources,
                    question_ids=data.qids, questions_sha256=builder.fingerprint(questions),
                    targets_sha256=builder.fingerprint(targets), script_sha256=sha256(Path(__file__)),
                    scorer_sha256=sha256(Path(qwen_server.__file__)), selection_policy=policy,
                    uses_evaluation_labels_for_selection=False)
    if confidence is not None:
        protocol.update(route_confidence=confidence, score_cutoff=ROUTE_CUTOFF, ranker_sha256=frozen['ranker_sha256'],
                        ranker_scoring_sha256=sha256(path.parent / 'scoring.json'))
    # Two-view: cutoff-route questions are also read with the selector's tight set, and the answer sums both reads'
    # renormalized letter log-probabilities. Reads already made by the routed run are reused only where its row has
    # the same route and the same selected messages in the same order, so its prompts are identical.
    reuse = {}
    if two_view:
        reuse_path = OUTPUT / f'{dataset}_routed_t{threshold:g}_c{confidence:g}{suffix}' / 'predictions.jsonl'
        protocol['selection_policy'] = policy + ('; two-view: cutoff-route questions are also read with the selector set and '
                                                 'answered by the sum of both reads\' renormalized letter log-probabilities')
        if reuse_path.exists():
            reuse = {r['example_id']: r for r in (json.loads(line) for line in reuse_path.open(encoding='utf-8') if line.strip())}
            protocol.update(reused_routed_predictions=str(reuse_path.relative_to(ROOT)).replace('\\', '/'),
                            reused_routed_predictions_sha256=sha256(reuse_path),
                            reused_routed_protocol_sha256=sha256(reuse_path.parent / 'protocol.json'))
    done = accept_protocol(out / 'protocol.json', protocol, rows, data.qids)

    def read(server, q, history, tag=''):
        shifts = []
        for shift in range(rotations):
            r = server.score(build_messages(rotated(q, shift), config, history),
                             request_id=f'{q["example_id"]}|{tag}shift{shift}', history_messages=len(history))
            conditional = r['label_conditional_logprobs']
            shifts.append(dict(shift=shift, predicted_label=r['predicted_label'], label_conditional_logprobs=conditional,
                               answer_logprobs=answer_scores(conditional, labels, shift), prompt_tokens=r['prompt_tokens'],
                               inference_seconds=r['inference_seconds'], timings=r.get('timings')))
        return shifts

    def reused(q, selection):
        old = reuse.get(q['example_id'])
        same = old is not None and old['route'] == selection['route'] and (
            [m['message_id'] for m in old['selected_messages']] == [m['message_id'] for m in selection['selected_messages']])
        return old['shifts'] if same else None

    todo = [q for q in questions if q['example_id'] not in done]
    print(f'SCORING {dataset}: {name}, {budget:,}-token cap: {len(questions)} questions, '
          f'{len(done)} complete, {rotations} answer order(s)', flush=True)
    if todo:
        with qwen_server.LlamaCppServer(out, config) as server, predictions.open('a', encoding='utf-8', buffering=1) as stream:
            for q in todo:
                history, selection = histories.get(q, server, config)
                shifts, extra = reused(q, selection) if two_view else None, {}
                if two_view:
                    extra.update(reused_routed_read=shifts is not None, two_view=selection['route'] == 'cutoff')
                if shifts is None:
                    shifts = read(server, q, history)
                if extra.get('two_view'):
                    tight_history, tight_selection = histories.tight_view(q, server, config)
                    tight_shifts = read(server, q, tight_history, 'tight|')
                    extra.update(views=dict(cutoff=shifts, selector=tight_shifts), selector_view=tight_selection)
                    shifts = combine_views([shifts, tight_shifts], labels)
                # Targets are read only after memory selection and all answer scoring.
                gold = labels.index(targets[q['example_id']])
                row = dict(example_id=q['example_id'], persona_id=q['persona_id'], question_type=q['question_type'],
                           gold_index=gold, predicted_label=shifts[0]['predicted_label'],
                           correct=shifts[0]['predicted_label'] == labels[gold], shifts=shifts, **selection, **extra)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                rows.append(row)
                if len(rows) % 25 == 0:
                    print(f'{dataset}: {len(rows)}/{len(questions)} complete; {(time.perf_counter() - started) / 60:.1f} min', flush=True)
    if {r['example_id'] for r in rows} != set(data.qids):
        raise ValueError('Incomplete evaluation')
    if confidence is None:
        report(out, dataset, rows, labels, threshold, budget)
    else:
        routed_report(out, dataset, rows, labels, threshold, confidence, budget)
    write_json(out / 'runtime.json', dict(seconds_this_process=time.perf_counter() - started, resumed_questions=len(done)))
    data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='fit both passes on half B graded labels (CPU) and report coverage')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=[])
    parser.add_argument('--threshold', type=float, help='keep messages with usefulness P(2) + 0.5 P(1) at or above this')
    parser.add_argument('--route-confidence', type=float, default=None,
                        help='use the selector set only when the estimated chance it holds every reference is at least this; '
                             'otherwise the ranker\'s 3.0 cutoff set')
    parser.add_argument('--two-view', action='store_true',
                        help='with --route-confidence: also read cutoff-route questions with the selector set and sum both reads')
    parser.add_argument('--memory-tokens', type=int, default=MEMORY_TOKENS)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.datasets and (args.threshold is None or not 0 < args.threshold < 1):
        parser.error('--threshold between 0 and 1 is required with --datasets')
    if args.route_confidence is not None and not 0 <= args.route_confidence <= 1:
        parser.error('--route-confidence must be between 0 and 1')
    if args.two_view and args.route_confidence is None:
        parser.error('--two-view needs --route-confidence')
    if args.memory_tokens < 1:
        parser.error('--memory-tokens must be positive')
    if args.train:
        train()
    for dataset in args.datasets:
        score(dataset, args.threshold, args.summarize_only, args.memory_tokens, args.route_confidence, args.two_view)


if __name__ == '__main__':
    main()
