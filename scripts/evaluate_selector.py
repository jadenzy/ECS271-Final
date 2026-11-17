"""Two-pass yes/no selector on the 57 fine-tuned features, without the ranker.

python scripts/evaluate_selector.py --train
python scripts/evaluate_selector.py --datasets v1 locomo v2 --threshold 0.2
python scripts/evaluate_selector.py --datasets v1 --threshold 0.2 --summarize-only

Training (--train, CPU) uses only V2 persona half B, the same 9,192 questions and 57 features
(55 original + learned bilinear similarity + generic correction) the fine-tuned ranker was fitted
on. Pass 1 is a LightGBM yes/no classifier over every visible message: is it an annotated
reference? Pass 2 rescores pass 1's top 30 messages per question with pass 1's raw score, pool-relative
features (rank, gaps, softmax share of the pool, how many messages sit near the top) and the
scores of the chronologically neighbouring messages, plus the 57 features again. Pass 2 is
trained on out-of-fold pass 1 scores (five persona folds), so its inputs on B look like those on
unseen personas. Messages outside pass 1's top 30 get probability 0. The ranker's scores are not
used anywhere. Models: models/embedding_ab/selector/; record and V2 validation coverage:
results/selector/training.json.

Scoring keeps every message with pass 2 probability >= --threshold (the most probable message is
always kept), packs them by probability under a --memory-tokens cap (default 8,192, so the
threshold, not the cap, decides the size) and shows them in chronological order, with the same
Qwen prompt and answer-order policy as evaluate_embedding_ab_cutoff.py: V2/V1 four cyclic orders,
LoCoMo one pass. The threshold is fixed on V2 validation before any test set is scored; no
evaluation labels are read during selection.

Outputs under results/selector/{dataset}_t{threshold}/: predictions.jsonl (resumable), protocol.json,
report.json, runtime.json and the server log. Probabilities are cached next to each dataset's
features as selector_probabilities.npy.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import build_ranker_features as builder
from embedding_ab_experiment import MODELS, OUTPUT as TRAINING_RECORDS
from embedding_ab_math import CachedInputs
from evaluate_embedding_ab_transfer import (CACHE, DATASETS, MODEL_ROOT, FrozenHistories, accept_protocol, answer_scores,
                                            frozen_models)
from evaluate_qwen import (build_messages, evaluation_config, order_for_cache, pack_selector_messages, runtime_metadata,
                           selected_history)
from label_reference_messages import load_references
from math_ranker_experiment import NAMES
from message_dataset import load_message_dataset
from option_rotation_pilot import rotated, summarize
import ranker55 as recipe
from raw_datasets import ROOT, load_dataset, sha256
from tune_ranker55 import Bundle, write_json
import qwen_llamacpp_server as qwen_server

FEATURES = ROOT / 'data/ranker55/embedding_ab/finetuned/v2'
SELECTOR = MODELS / 'selector'
OUTPUT = ROOT / 'results/selector'
MEMORY_TOKENS = 8192
TOP = 30
SEED, FOLDS, THREADS = 271, 5, 8
POOL_NAMES = ['pass1_score', 'rank', 'gap_to_top', 'gap_to_previous', 'gap_to_next', 'pool_z', 'top_score', 'top_z',
              'top_minus_5th', 'top_minus_10th', 'within_1', 'within_2', 'within_3', 'pool_size', 'cumulative_share', 'share',
              'previous_message_score', 'next_message_score', 'top_minus_previous_message', 'top_minus_next_message',
              'previous_message_rank', 'next_message_rank', 'best_neighbour_minus_score']
PASS2_NAMES = POOL_NAMES + NAMES
THRESHOLDS = (.6, .5, .4, .3, .25, .2, .15, .1, .07, .05, .03, .02)


def pass1_parameters():
    params = recipe.training_parameters(min_child_samples=100, num_leaves=63, max_depth=6, reg_lambda=10.)
    params = {k: v for k, v in params.items() if k not in ('objective', 'label_gain', 'lambdarank_truncation_level')}
    return dict(params, objective='binary', n_jobs=THREADS, verbose=-1)


PASS2_PARAMETERS = dict(objective='binary', n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=6,
                        min_child_samples=50, reg_lambda=5., max_bin=63, random_state=1271, n_jobs=THREADS,
                        deterministic=True, force_col_wise=True, verbose=-1)


def pool_features(scores, offsets, indices):
    """Rows (pass 1's top TOP per question) and their pool-relative and neighbour features."""
    rows, feats = [], []
    for i in range(len(offsets) - 1):
        a, b = int(offsets[i]), int(offsets[i + 1])
        s = np.asarray(scores[a:b], dtype=float)
        order = np.argsort(-s, kind='stable')
        v = s[order]
        mean, sd = s.mean(), s.std() + 1e-9
        rank_of = np.empty(len(s), dtype=int)
        rank_of[order] = np.arange(len(s))
        idx = np.asarray(indices[a:b])
        where = {int(x): j for j, x in enumerate(idx)}
        w = np.exp(v - v[0])
        share = w / w.sum()
        cumulative = np.cumsum(share)
        for r, j in enumerate(order[:TOP]):
            prev_j, next_j = where.get(int(idx[j]) - 1), where.get(int(idx[j]) + 1)
            ps = s[prev_j] if prev_j is not None else np.nan
            ns = s[next_j] if next_j is not None else np.nan
            feats.append([s[j], r, v[0] - s[j], (v[r - 1] - s[j]) if r else 0., (s[j] - v[r + 1]) if r + 1 < len(v) else 0.,
                          (s[j] - mean) / sd, v[0], (v[0] - mean) / sd, v[0] - v[min(4, len(v) - 1)], v[0] - v[min(9, len(v) - 1)],
                          (v >= v[0] - 1).sum(), (v >= v[0] - 2).sum(), (v >= v[0] - 3).sum(), len(s), cumulative[r], share[r],
                          ps, ns, v[0] - ps, v[0] - ns,
                          rank_of[prev_j] if prev_j is not None else np.nan, rank_of[next_j] if next_j is not None else np.nan,
                          np.nanmax([ps, ns]) - s[j] if not (np.isnan(ps) and np.isnan(ns)) else np.nan])
            rows.append(a + j)
    return np.array(rows, dtype=np.int64), np.array(feats, dtype=np.float32)


def predict(pass1, pass2, X, offsets, indices):
    """Pass 2 probability for every row (0 outside pass 1's top TOP)."""
    raw = pass1.predict(X, raw_score=True, num_threads=THREADS)
    rows, pool = pool_features(raw, offsets, indices)
    p = np.zeros(len(raw))
    p[rows] = pass2.predict(np.column_stack([pool, X[rows]]), num_threads=THREADS)
    return p


def kept_mask(p, offsets, threshold):
    best = np.repeat(np.maximum.reduceat(p, offsets[:-1]), np.diff(offsets))
    return (p >= threshold) | (p >= best)


def copy_bundle(path):
    with Bundle(path) as b:
        return {k: np.array(b.read(k), copy=True)
                for k in ('X', 'y', 'offsets', 'question_ids', 'persona_ids', 'message_ids', 'message_indices')}


def reference_mask(split, d):
    """Rows that are an annotated reference visible in the question's pool (coverage reporting only)."""
    questions, memories, sources = load_message_dataset('v2', [split], root=ROOT)
    refs = load_references(questions, sources, ROOT)
    by_id = {q['example_id']: q for q in questions}
    mask = np.zeros(len(d['message_ids']), dtype=bool)
    for i, qid in enumerate(d['question_ids'].tolist()):
        q = by_id[str(qid)]
        keys = {k for k in refs[q['example_id']] if k[1].strip()}
        pool = {(memories[m]['role'], memories[m]['text']): m for m in q['memory_ids']}
        visible = {pool[k] for k in keys if k in pool}
        a, b = d['offsets'][i:i + 2]
        if visible:
            mask[a:b] = np.isin(d['message_ids'][a:b].astype(str), list(visible))
    return mask


def coverage(p, ref, offsets):
    table = []
    has = [i for i in range(len(offsets) - 1) if ref[offsets[i]:offsets[i + 1]].any()]
    for t in THRESHOLDS:
        keep = kept_mask(p, offsets, t)
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
    Xtr = np.column_stack([tr['X'], np.load(MODEL_ROOT / 'train_extra.npy')]).astype(np.float32)
    Xva = np.column_stack([va['X'], np.load(MODEL_ROOT / 'validation_extra.npy')]).astype(np.float32)
    ytr = (tr['y'] > 0).astype(int)
    questions, _, _ = load_message_dataset('v2', ['train'], root=ROOT)
    qtype = {q['example_id']: q['question_type'] for q in questions}
    weights = recipe.row_weights([.5 if qtype[str(q)] == 'ask_to_forget' else 1. for q in tr['question_ids'].tolist()], tr['offsets'])
    print(f'B: {len(tr["offsets"]) - 1} questions, {len(ytr):,} messages, {ytr.sum():,} references', flush=True)

    personas = np.unique(tr['persona_ids'])
    fold_of = dict(zip(np.random.default_rng(SEED).permutation(personas).tolist(), np.arange(len(personas)) % FOLDS))
    rfold = np.repeat(np.array([fold_of[p] for p in tr['persona_ids'].tolist()]), np.diff(tr['offsets']))
    oof = np.full(len(ytr), np.nan)
    for f in range(FOLDS):
        fit = rfold != f
        model = lgb.LGBMClassifier(**pass1_parameters()).fit(Xtr[fit], ytr[fit], sample_weight=weights[fit], feature_name=NAMES)
        oof[~fit] = model.predict(Xtr[~fit], raw_score=True, num_threads=THREADS)
        print(f'pass 1 fold {f + 1}/{FOLDS} done', flush=True)
    pass1 = lgb.LGBMClassifier(**pass1_parameters()).fit(Xtr, ytr, sample_weight=weights, feature_name=NAMES)
    rows, pool = pool_features(oof, tr['offsets'], tr['message_indices'])
    print(f'pass 2 rows: {len(rows):,}; {ytr[rows].sum():,} of {ytr.sum():,} references inside pass 1 top {TOP}', flush=True)
    pass2 = lgb.LGBMClassifier(**PASS2_PARAMETERS).fit(np.column_stack([pool, Xtr[rows]]), ytr[rows], sample_weight=weights[rows],
                                                       feature_name=PASS2_NAMES)
    SELECTOR.mkdir(parents=True, exist_ok=True)
    pass1.booster_.save_model(str(SELECTOR / 'pass1.txt'))
    pass2.booster_.save_model(str(SELECTOR / 'pass2.txt'))

    b1, b2 = lgb.Booster(model_file=str(SELECTOR / 'pass1.txt')), lgb.Booster(model_file=str(SELECTOR / 'pass2.txt'))
    p = predict(b1, b2, Xva, va['offsets'], va['message_indices'])
    np.save(val_path.parent / 'selector_probabilities.npy', p)
    table = coverage(p, reference_mask('val', va), va['offsets'])
    gain = b2.feature_importance('gain')
    record = dict(pass1_sha256=sha256(SELECTOR / 'pass1.txt'), pass2_sha256=sha256(SELECTOR / 'pass2.txt'),
                  training_features_sha256=sha256(train_path), training_extra_sha256=sha256(MODEL_ROOT / 'train_extra.npy'),
                  validation_features_sha256=sha256(val_path), validation_extra_sha256=sha256(MODEL_ROOT / 'validation_extra.npy'),
                  validation_probabilities_sha256=sha256(val_path.parent / 'selector_probabilities.npy'),
                  training_questions=len(tr['offsets']) - 1, training_personas=personas.tolist(),
                  folds={p: int(f) for p, f in fold_of.items()}, top=TOP, seed=SEED,
                  pass1_parameters=pass1_parameters(), pass2_parameters=PASS2_PARAMETERS,
                  pass1_features=NAMES, pass2_features=PASS2_NAMES, ask_to_forget_weight=.5,
                  references_inside_pass1_top=int(ytr[rows].sum()), references=int(ytr.sum()),
                  pass2_top_features_by_gain={PASS2_NAMES[j]: float(gain[j] / gain.sum()) for j in np.argsort(-gain)[:10]},
                  validation_coverage=table, ranker_scores_used=False, script_sha256=sha256(Path(__file__)),
                  seconds=time.perf_counter() - started)
    write_json(OUTPUT / 'training.json', record)
    print('V2 validation coverage (questions with every reference kept / exactly the references / messages kept):')
    for r in table:
        print(f"  threshold {r['threshold']:<5g} every {r['every_reference']:.1%}  exact {r['exact_set']:.1%}  messages {r['mean_messages']:.1f}")


def selector_identity():
    record_path = OUTPUT / 'training.json'
    record = json.loads(record_path.read_text())
    for name, key in [('pass1.txt', 'pass1_sha256'), ('pass2.txt', 'pass2_sha256')]:
        if sha256(SELECTOR / name) != record[key]:
            raise ValueError('Selector model changed since training')
    return dict(pass1_sha256=record['pass1_sha256'], pass2_sha256=record['pass2_sha256'],
                training_record_sha256=sha256(record_path), top=TOP, ranker_scores_used=False)


def probabilities(dataset, data, path, frozen):
    """Pass 2 probabilities for a dataset, cached; checks the 57 inputs by reproducing the frozen ranker's scores."""
    import lightgbm as lgb
    identity = selector_identity()
    cache, record_path = path.parent / 'selector_probabilities.npy', path.parent / 'selector_scoring.json'
    extra_path = path.parent / 'mathematical_features.npy'
    expected = dict(identity, features_sha256=sha256(path), mathematical_features_sha256=sha256(extra_path))
    if cache.exists() and record_path.exists():
        saved = json.loads(record_path.read_text())
        if {k: saved.get(k) for k in expected} == expected and saved.get('probabilities_sha256') == sha256(cache):
            return np.load(cache)
    X = np.column_stack([np.asarray(data.X), np.load(extra_path)]).astype(np.float32)
    ranker = lgb.Booster(model_file=str(MODEL_ROOT / 'ranker57.txt'))
    if sha256(MODEL_ROOT / 'ranker57.txt') != frozen['ranker_sha256']:
        raise ValueError('Frozen ranker changed')
    check = ranker.predict(X, num_threads=THREADS)
    error = float(np.max(np.abs(check - np.load(path.parent / 'ranker_scores.npy'))))
    if error > 1e-6:
        raise ValueError(f'57 inputs do not reproduce the cached ranker scores (max error {error})')
    b1, b2 = lgb.Booster(model_file=str(SELECTOR / 'pass1.txt')), lgb.Booster(model_file=str(SELECTOR / 'pass2.txt'))
    p = predict(b1, b2, X, np.asarray(data.offsets), np.asarray(data.bundle.read('message_indices')))
    np.save(cache, p)
    write_json(record_path, dict(expected, probabilities_sha256=sha256(cache), ranker_reproduction_max_error=error,
                                 uses_evaluation_labels_or_answers=False))
    print(f'{dataset}: selector probabilities cached ({len(p):,} messages; ranker inputs reproduced to {error:.1e})', flush=True)
    return p


class SelectorHistories(FrozenHistories):
    def __init__(self, questions, contexts, data, probabilities, threshold, budget):
        super().__init__(questions, contexts, data, probabilities)
        self.threshold, self.budget = threshold, budget

    def get(self, question, server, config):
        key = question['history_id'], question['source_end_index']
        members = self.pools[key]
        i = self.groups[question['example_id']]
        a, b = map(int, self.data.offsets[i:i+2])
        p = np.asarray(self.scores[a:b], dtype=float)
        ranked = np.argsort(-p, kind='stable')
        kept = [int(j) for j in ranked if p[j] >= self.threshold or j == ranked[0]]
        chosen, packing = pack_selector_messages(server, question, config, members, kept, self.budget, blocks=self.blocks[key])
        history = selected_history(members, chosen)
        return history, dict(**packing, selected_memory_count=len(history), messages_passing_threshold=len(kept),
                             selected_messages=[dict(message_id=members[j]['message_id'], message_index=members[j]['message_index'],
                                                     probability=float(p[j])) for j in chosen])


def report(out, dataset, rows, labels, threshold, budget):
    split, rotations = DATASETS[dataset]
    result = summarize(rows, labels) if rotations > 1 else dict(
        questions=len(rows), accuracy={'one_pass': float(np.mean([r['correct'] for r in rows]))})
    result.update(status='complete', dataset=dataset, split=split, rotations=rotations, memory_tokens=budget, threshold=threshold,
                  mean_selected_messages=float(np.mean([r['selected_memory_count'] for r in rows])),
                  mean_messages_passing_threshold=float(np.mean([r['messages_passing_threshold'] for r in rows])),
                  mean_memory_tokens=float(np.mean([r['packed_memory_tokens'] for r in rows])),
                  share_questions_capped=float(np.mean([r['budget_skipped_messages'] > 0 for r in rows])))
    write_json(out / 'report.json', result)
    print(f'COMPLETE {dataset}: {result["questions"]} questions; {result["accuracy"]}', flush=True)


def score(dataset, threshold, summarize_only=False, budget=MEMORY_TOKENS):
    started = time.perf_counter()
    _, record, frozen = frozen_models()
    split, rotations = DATASETS[dataset]
    out = OUTPUT / (f'{dataset}_t{threshold:g}' if budget == MEMORY_TOKENS else f'{dataset}_t{threshold:g}_mem{budget}')
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / 'predictions.jsonl'
    rows = [json.loads(line) for line in predictions.open(encoding='utf-8') if line.strip()] if predictions.exists() else []
    config = evaluation_config(dataset, 'selector')
    labels = config['answer_labels']
    if summarize_only:
        return report(out, dataset, rows, labels, threshold, budget)
    questions, contexts, targets, sources = load_dataset(dataset, [split], root=ROOT)
    questions = order_for_cache(questions)
    path = CACHE / dataset / split / 'features.npz'
    scoring = json.loads((path.parent / 'scoring.json').read_text())
    if scoring['adapter_sha256'] != frozen['adapter_sha256'] or scoring['features_sha256'] != sha256(path):
        raise ValueError('Feature cache differs from the frozen adapter')
    data = CachedInputs(path)
    if data.meta['embedding_recipe'] != record['embedding_recipe']:
        raise ValueError('Feature encoder differs from the trained models')
    for source, expected in sources.items():
        if data.meta['source_sha256'].get(source) != expected:
            raise ValueError('Current raw source or split differs from the feature cache')
    p = probabilities(dataset, data, path, frozen)
    histories = SelectorHistories(questions, contexts, data, p, threshold, budget)
    if any([o['label'] for o in q['options']] != labels for q in questions):
        raise ValueError('Raw option labels differ from the configured scorer')
    settings = runtime_metadata(dataset, 'selector', memory_tokens=budget)
    config.update(memory_tokens=budget, input_policy='raw_complete_messages_v1', server_settings=settings,
                  model=settings['Model'], context_tokens=settings['ContextTokens'])
    protocol = dict(dataset=dataset, split=split, rotations=rotations, config=config,
                    frozen_features=dict(encoder=frozen['encoder'], adapter_sha256=frozen['adapter_sha256'], feature_names=NAMES),
                    selector=selector_identity(), threshold=threshold,
                    selector_scoring_sha256=sha256(path.parent / 'selector_scoring.json'), raw_sources=sources,
                    question_ids=data.qids, questions_sha256=builder.fingerprint(questions),
                    targets_sha256=builder.fingerprint(targets), script_sha256=sha256(Path(__file__)),
                    scorer_sha256=sha256(Path(qwen_server.__file__)),
                    selection_policy=(f'two-pass selector: keep messages with pass 2 probability >= {threshold:g} (the most probable '
                                      f'always kept), packed by probability under a {budget:,}-token cap; whole messages, '
                                      'chronological display; ranker scores not used'),
                    uses_evaluation_labels_for_selection=False)
    done = accept_protocol(out / 'protocol.json', protocol, rows, data.qids)
    todo = [q for q in questions if q['example_id'] not in done]
    print(f'SCORING {dataset} with the two-pass selector, threshold {threshold:g}, {budget:,}-token cap: {len(questions)} questions, '
          f'{len(done)} complete, {rotations} answer order(s)', flush=True)
    if todo:
        with qwen_server.LlamaCppServer(out, config) as server, predictions.open('a', encoding='utf-8', buffering=1) as stream:
            for q in todo:
                history, selection = histories.get(q, server, config)
                shifts = []
                for shift in range(rotations):
                    r = server.score(build_messages(rotated(q, shift), config, history),
                                     request_id=f'{q["example_id"]}|shift{shift}', history_messages=len(history))
                    conditional = r['label_conditional_logprobs']
                    shifts.append(dict(shift=shift, predicted_label=r['predicted_label'], label_conditional_logprobs=conditional,
                                       answer_logprobs=answer_scores(conditional, labels, shift), prompt_tokens=r['prompt_tokens'],
                                       inference_seconds=r['inference_seconds'], timings=r.get('timings')))
                # Targets are read only after memory selection and all answer scoring.
                gold = labels.index(targets[q['example_id']])
                row = dict(example_id=q['example_id'], persona_id=q['persona_id'], question_type=q['question_type'],
                           gold_index=gold, predicted_label=shifts[0]['predicted_label'],
                           correct=shifts[0]['predicted_label'] == labels[gold], shifts=shifts, **selection)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                rows.append(row)
                if len(rows) % 25 == 0:
                    print(f'{dataset}: {len(rows)}/{len(questions)} complete; {(time.perf_counter() - started) / 60:.1f} min', flush=True)
    if {r['example_id'] for r in rows} != set(data.qids):
        raise ValueError('Incomplete evaluation')
    report(out, dataset, rows, labels, threshold, budget)
    write_json(out / 'runtime.json', dict(seconds_this_process=time.perf_counter() - started, resumed_questions=len(done)))
    data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true', help='fit both passes on persona half B (CPU) and report V2 validation coverage')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=[])
    parser.add_argument('--threshold', type=float, help='keep messages with pass 2 probability at or above this')
    parser.add_argument('--memory-tokens', type=int, default=MEMORY_TOKENS)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.datasets and (args.threshold is None or not 0 < args.threshold < 1):
        parser.error('--threshold between 0 and 1 is required with --datasets')
    if args.memory_tokens < 1:
        parser.error('--memory-tokens must be positive')
    if args.train:
        train()
    for dataset in args.datasets:
        score(dataset, args.threshold, args.summarize_only, args.memory_tokens)


if __name__ == '__main__':
    main()
