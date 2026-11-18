"""Neural message ranker on the frozen half-A embeddings and the 57 features (no Qwen anywhere here).

python scripts/neural_ranker.py --stage options     # embed V2 B/validation answer options once (frozen encoder, GPU)
python scripts/neural_ranker.py --stage check       # Signal B assumption on half B (references vs matched messages)
python scripts/neural_ranker.py --stage train --variant B   # reference loss only
python scripts/neural_ranker.py --stage train --variant C   # reference loss + 0.25 x answer loss
python scripts/neural_ranker.py --stage validate    # A (saved LightGBM) vs B vs C on the 909 V2 validation questions
python scripts/neural_ranker.py --stage score --variant B   # frozen scores for V2 test, V1 and LoCoMo

Training uses the same 9,192 half-B questions as the saved 57-feature LightGBM (out-of-fold learned features)
and checkpoints are chosen on V2 validation NDCG@10 (ties: every-visible-reference coverage@15). The model sees,
per message, the 57 features (standardized on B, clipped to +-5) and, through one shared 768->64 projection of
the question and message vectors, their elementwise product and absolute difference; one 128-unit hidden layer
gives one score per whole message across the full visible history.

Signal A: softmax over the visible history, mean of -log p over the question's annotated references, each question
weighted equally (ask_to_forget 0.5). Signal B (variant C only): the dataset's answer options are embedded once as
documents; each message's similarity to each option, minus that option's mean similarity over the history, is
averaged with the ranker's message probabilities, and a cross-entropy over options (temperature from the check
stage) rewards evidence that favours the correct option. The answer loss skips ask_to_forget questions: on half B
their references favour a wrong option (the forgotten preference), so it would push the ranker away from them.
Answer texts are never ranker inputs.

Outputs: models/neural_ranker/{B,C}.pt and training records; results/neural_ranker/ (check, training, validation
reports); neural_{B,C}_scores.npy next to each evaluation dataset's cached features.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import build_ranker_features as builder
from embedding_ab_experiment import OUTPUT as AB_OUTPUT, ENCODER, verify_encoder
from embedding_ab_math import CachedInputs
from evaluate_embedding_ab_transfer import CACHE, DATASETS, MODEL_ROOT
from math_ranker_experiment import NAMES
from option_rotation_pilot import bootstrap
from raw_datasets import ROOT, load_dataset, sha256
from tune_ranker55 import Bundle, aggregate, score_questions, validation_targets, write_json

FEATURES = ROOT / 'data/ranker55/embedding_ab/finetuned/v2'
TRAIN, VAL = FEATURES / 'train/features.npz', FEATURES / 'val/features.npz'
MODELS = ROOT / 'models/neural_ranker'
OUTPUT = ROOT / 'results/neural_ranker'
OPTIONS = FEATURES / 'train/option_embeddings.npy'
OPTION_INDEX = FEATURES / 'train/option_embeddings.json'
SEED = 271
SETTINGS = dict(projection=64, hidden=128, learning_rate=1e-3, weight_decay=0.01, batch_questions=16, epochs=5,
                clip=5.0, forget_weight=0.5, answer_weight=0.25, controls=20,
                answer_loss_excludes=('ask_to_forget',))  # the check shows forget references favour a wrong option


# ----------------------------------------------------------------------------- data

def inputs(path, extra):
    data = CachedInputs(path)
    X = np.column_stack([data.X, np.load(extra)]).astype(np.float32)
    if X.shape[1] != len(NAMES) or not np.isfinite(X).all():
        raise ValueError('Expected 57 finite features')
    return data, X


def roles(data):
    root = Path(data.meta['embedding_cache_path'])
    with (root / 'messages.jsonl').open(encoding='utf-8') as stream:
        by_id = {r['message_id']: r['role'] for r in map(json.loads, stream)}
    return np.array([by_id[str(m)] == 'user' for m in data.bundle.read('message_ids')])


def option_vectors():
    index = json.loads(OPTION_INDEX.read_text())
    if index['vectors_sha256'] != sha256(OPTIONS):
        raise ValueError('Option vectors differ from their index')
    return np.load(OPTIONS), index


def centered_similarity(M, O):
    """Message-by-option cosine minus each option's mean over the visible history."""
    S = M @ O.T
    return S - S.mean(0, keepdims=True)


# ----------------------------------------------------------------------------- stages

def stage_options():
    info = verify_encoder()
    with Bundle(TRAIN) as b:
        if json.loads(json.dumps(b.metadata['embedding_recipe']))['model'] != info:
            raise ValueError('Training features were not built with the frozen half-A encoder')
        train_ids = b.read('question_ids').tolist()
    with Bundle(VAL) as b:
        val_ids = b.read('question_ids').tolist()
    questions, _, targets, sources = load_dataset('v2', ['train', 'val'], root=ROOT, include_contexts=False)
    by_id = {q['example_id']: q for q in questions}
    texts, rows, entries = {}, [], {}
    for qid in train_ids + val_ids:
        q = by_id[qid]
        labels = [o['label'] for o in q['options']]
        ids = []
        for o in q['options']:
            if o['text'] not in texts:
                texts[o['text']] = len(texts)
            ids.append(texts[o['text']])
        entries[qid] = dict(labels=labels, rows=ids, gold=labels.index(targets[qid]))
    ordered = list(texts)
    encoder = builder.Encoder(ENCODER, 'cuda')
    vectors = np.zeros((len(ordered), 768), dtype=np.float32)
    order = np.argsort([len(t) for t in ordered])
    started = time.perf_counter()
    for s in range(0, len(order), 64):
        batch = order[s:s + 64]
        vectors[batch] = encoder.encode([ordered[i] for i in batch], 'memory')
    np.save(OPTIONS, vectors)
    write_json(OPTION_INDEX, dict(encoder=info, prefix='memory', texts=len(ordered), questions=entries,
                                  raw_sources=sources, vectors_sha256=sha256(OPTIONS), seconds=time.perf_counter() - started))
    print(f'{len(ordered)} unique option texts for {len(entries)} questions embedded', flush=True)


def margin(Sc, members, gold):
    v = Sc[members].mean(0)
    return float(v[gold] - np.delete(v, gold).max()), int(np.argmax(v) == gold)


def stage_check():
    """Do annotated references favour the correct option more than role- and length-matched messages?"""
    data = CachedInputs(TRAIN)
    vectors, index = option_vectors()
    memory = np.asarray(data.memory)
    y = data.bundle.read('y').astype(bool)
    lengths = data.bundle.read('content_token_lengths').astype(float)
    user = roles(data)
    rng = np.random.default_rng(SEED)
    rows, spreads = [], []
    for i, qid in enumerate(data.qids):
        a, b = map(int, data.offsets[i:i + 2])
        ref = np.flatnonzero(y[a:b])
        if not len(ref):
            continue
        e = index['questions'][qid]
        Sc = centered_similarity(memory[data.mi[a:b]], vectors[e['rows']])
        m_ref, win_ref = margin(Sc, ref, e['gold'])
        spreads.append(np.std(Sc[ref].mean(0)))
        others = np.setdiff1d(np.arange(b - a), ref)
        L, U = lengths[a:b], user[a:b]
        controls, wins = [], []
        for _ in range(SETTINGS['controls']):
            used, chosen = set(), []
            for j in ref:
                pool = [k for k in others if U[k] == U[j] and k not in used]
                if not pool:
                    pool = [k for k in others if k not in used]
                pool = np.array(pool)
                ratio = np.maximum(L[pool], 1) / max(L[j], 1)
                close = pool[(ratio >= 2 / 3) & (ratio <= 1.5)]
                k = int(rng.choice(close)) if len(close) else int(pool[np.argmin(np.abs(L[pool] - L[j]))])
                used.add(k); chosen.append(k)
            m, w = margin(Sc, np.array(chosen), e['gold'])
            controls.append(m); wins.append(w)
        rows.append(dict(example_id=qid, persona_id=str(data.personas[i]), question_type=data.types[i], references=len(ref),
                         reference_margin=m_ref, control_margin=float(np.mean(controls)), reference_gold_top=win_ref,
                         control_gold_top=float(np.mean(wins))))
    diff = np.array([r['reference_margin'] - r['control_margin'] for r in rows])
    groups = np.array([r['persona_id'] for r in rows])

    def summary(select):
        d, g = diff[select], groups[select]
        sub = [r for r, s in zip(rows, select) if s]
        return dict(questions=int(select.sum()), reference_margin=float(np.mean([r['reference_margin'] for r in sub])),
                    control_margin=float(np.mean([r['control_margin'] for r in sub])), difference=float(d.mean()),
                    ci95=bootstrap(d, g), reference_gold_top=float(np.mean([r['reference_gold_top'] for r in sub])),
                    control_gold_top=float(np.mean([r['control_gold_top'] for r in sub])),
                    share_reference_above_control=float(np.mean(d > 0)))

    allq = np.ones(len(rows), bool)
    report = dict(status='complete', cohort='V2 persona half B', questions=len(rows), controls_per_question=SETTINGS['controls'],
                  control_policy='per reference, a non-reference message of the same role with content length within x2/3..x1.5 '
                                 '(nearest length if none), without replacement; margin = mean centred similarity to the '
                                 'correct option minus the best alternative',
                  all=summary(allq),
                  by_type={t: summary(np.array([r['question_type'] == t for r in rows])) for t in sorted(set(data.types))},
                  persona_share_positive=float(np.mean([diff[groups == p].mean() > 0 for p in np.unique(groups)])),
                  answer_temperature=float(np.median(spreads)),
                  answer_temperature_rule='median over B questions of the std across options of the reference-averaged '
                                          'centred similarity (logits then have unit scale)')
    report['keep_signal_b'] = bool(report['all']['difference'] > 0 and report['all']['ci95'][0] > 0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / 'signal_b_check.json', report)
    with (OUTPUT / 'signal_b_check_rows.jsonl').open('w', encoding='utf-8') as stream:
        for r in rows:
            stream.write(json.dumps(r) + '\n')
    s = report['all']
    print(f"Signal B check on {len(rows)} B questions: reference margin {s['reference_margin']:+.4f}, matched control "
          f"{s['control_margin']:+.4f}, difference {s['difference']:+.4f} ({s['ci95'][0]:+.4f} to {s['ci95'][1]:+.4f}); "
          f"gold option on top {s['reference_gold_top']:.1%} vs {s['control_gold_top']:.1%}; "
          f"{report['persona_share_positive']:.0%} of personas positive")
    for t, v in report['by_type'].items():
        print(f"  {t:<32} n={v['questions']:>5} diff {v['difference']:+.4f} ({v['ci95'][0]:+.4f} to {v['ci95'][1]:+.4f})  "
              f"gold top {v['reference_gold_top']:.1%} vs {v['control_gold_top']:.1%}")
    print(f"keep Signal B: {report['keep_signal_b']}; answer temperature {report['answer_temperature']:.4f}")
    data.close()


class Standardizer:
    def __init__(self, X=None, state=None):
        if state is None:
            self.mean, self.std = X.mean(0), X.std(0)
            self.std[self.std < 1e-8] = 1.
        else:
            self.mean, self.std = np.asarray(state['mean'], np.float32), np.asarray(state['std'], np.float32)

    def __call__(self, X):
        return np.clip((X - self.mean) / self.std, -SETTINGS['clip'], SETTINGS['clip']).astype(np.float32)

    def state(self):
        return dict(mean=self.mean.tolist(), std=self.std.tolist())


def network():
    import torch
    nn = torch.nn

    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(768, SETTINGS['projection'], bias=False)
            self.score = nn.Sequential(nn.Linear(len(NAMES) + 2 * SETTINGS['projection'], SETTINGS['hidden']), nn.ReLU(),
                                       nn.Linear(SETTINGS['hidden'], 1))

        def forward(self, x, m, q):
            pm, pq = self.project(m), self.project(q)
            return self.score(torch.cat([x, pm * pq, (pm - pq).abs()], 1)).squeeze(1)

    return Ranker()


class Scorer:
    """Score every row of a CachedInputs bundle with a trained network (GPU, no gradients)."""
    def __init__(self, model, standardize):
        self.model, self.standardize = model, standardize

    def __call__(self, data, X):
        import torch
        memory = torch.tensor(np.asarray(data.memory), device='cuda')
        query = torch.tensor(np.asarray(data.query), device='cuda')
        Z = torch.tensor(self.standardize(X), device='cuda')
        mi, qi = torch.tensor(data.mi, device='cuda'), np.asarray(data.qi)
        out = np.zeros(len(X), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for s in range(0, len(data.qids), 64):
                e = min(s + 64, len(data.qids))
                a, b = int(data.offsets[s]), int(data.offsets[e])
                qrow = torch.tensor(np.repeat(qi[s:e], np.diff(data.offsets[s:e + 1])), device='cuda')
                out[a:b] = self.model(Z[a:b], memory[mi[a:b]], query[qrow]).cpu().numpy()
        del memory, query, Z
        torch.cuda.empty_cache()
        return out


def val_targets():
    """V2 validation targets with arrays copied out of the memory-mapped bundle (it is closed afterwards)."""
    with Bundle(VAL) as vb:
        targets = validation_targets(vb, ROOT, 'val')
        return {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in targets.items()}


def val_metrics(scores, targets):
    rows = score_questions(scores, scores, targets)
    for i, r in enumerate(rows):
        a, b = map(int, targets['offsets'][i:i + 2])
        positive = targets['positive'][a:b]
        order = np.argsort(-scores[a:b], kind='stable')
        ranks = np.empty(b - a, int); ranks[order] = np.arange(1, b - a + 1)
        ref = ranks[positive]
        for k in (5, 10, 15):
            r[f'every_reference_at_{k}'] = bool(len(ref) and np.all(ref <= k))
        r['precision_at_5'] = float(positive[order[:5]].mean())
    keys = ['ndcg_at_10', 'hit_at_10', 'every_reference_at_5', 'every_reference_at_10', 'every_reference_at_15', 'precision_at_5']
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}, rows


def stage_train(variant):
    import torch
    started = time.perf_counter()
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    if variant == 'C':
        check = json.loads((OUTPUT / 'signal_b_check.json').read_text())
        if not check['keep_signal_b']:
            raise ValueError('The Signal B check did not support the answer loss')
        temperature = check['answer_temperature']
        vectors, index = option_vectors()
    data, X = inputs(TRAIN, MODEL_ROOT / 'train_extra.npy')
    partition = json.loads((AB_OUTPUT / 'partition.json').read_text())
    if set(data.qids) != set(partition['b_question_ids']):
        raise ValueError('Training must use exactly persona half B')
    standardize = Standardizer(X)
    y = data.bundle.read('y').astype(bool)
    Z = torch.tensor(standardize(X), device='cuda')
    memory = torch.tensor(np.asarray(data.memory), device='cuda')
    query = torch.tensor(np.asarray(data.query), device='cuda')
    mi = torch.tensor(data.mi, device='cuda')
    weights = np.array([SETTINGS['forget_weight'] if t == 'ask_to_forget' else 1. for t in data.types])
    usable = np.array([y[data.offsets[i]:data.offsets[i + 1]].any() for i in range(len(data.qids))])
    if variant == 'C':
        options = [torch.tensor(vectors[index['questions'][q]['rows']], device='cuda') for q in data.qids]
        gold = [index['questions'][q]['gold'] for q in data.qids]
    val, Xv = inputs(VAL, MODEL_ROOT / 'validation_extra.npy')
    targets = val_targets()
    model = network().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['learning_rate'], weight_decay=SETTINGS['weight_decay'])
    rng = np.random.default_rng(SEED)
    questions = np.flatnonzero(usable)
    history, best = [], None
    MODELS.mkdir(parents=True, exist_ok=True)
    for epoch in range(SETTINGS['epochs']):
        model.train()
        order = rng.permutation(questions)
        totals = dict(reference=0., answer=0.)
        for s in range(0, len(order), SETTINGS['batch_questions']):
            batch = np.sort(order[s:s + SETTINGS['batch_questions']])
            spans = [(int(data.offsets[i]), int(data.offsets[i + 1])) for i in batch]
            rows = torch.tensor(np.concatenate([np.arange(a, b) for a, b in spans]), device='cuda')
            qrow = torch.tensor(np.concatenate([np.full(b - a, data.qi[i]) for i, (a, b) in zip(batch, spans)]), device='cuda')
            scores = model(Z[rows], memory[mi[rows]], query[qrow])
            loss_ref, loss_ans, start = [], [], 0
            for i, (a, b) in zip(batch, spans):
                logp = torch.log_softmax(scores[start:start + b - a], 0)
                ref = torch.tensor(np.flatnonzero(y[a:b]), device='cuda')
                loss_ref.append(-logp[ref].mean() * weights[i])
                if variant == 'C' and data.types[i] not in SETTINGS['answer_loss_excludes']:
                    Sc = centered_similarity(memory[mi[a:b]], options[i])
                    evidence = logp.exp() @ Sc / temperature
                    loss_ans.append(torch.nn.functional.cross_entropy(evidence[None], torch.tensor([gold[i]], device='cuda')))
                start += b - a
            loss = torch.stack(loss_ref).mean()
            totals['reference'] += float(loss.detach()) * len(batch)
            if variant == 'C' and loss_ans:
                answer = torch.stack(loss_ans).mean()
                totals['answer'] += float(answer.detach()) * len(batch)
                loss = loss + SETTINGS['answer_weight'] * answer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scores_v = Scorer(model, standardize)(val, Xv)
        metrics, _ = val_metrics(scores_v, targets)
        entry = dict(epoch=epoch + 1, train_reference_loss=totals['reference'] / len(order),
                     train_answer_loss=totals['answer'] / len(order) if variant == 'C' else None, validation=metrics,
                     minutes=(time.perf_counter() - started) / 60)
        history.append(entry)
        print(f"{variant} epoch {epoch + 1}: reference loss {entry['train_reference_loss']:.4f}"
              + (f", answer loss {entry['train_answer_loss']:.4f}" if variant == 'C' else '')
              + f"; val NDCG@10 {metrics['ndcg_at_10']:.2%}, every ref@15 {metrics['every_reference_at_15']:.2%}"
              f" ({entry['minutes']:.1f} min)", flush=True)
        key = (round(metrics['ndcg_at_10'], 6), round(metrics['every_reference_at_15'], 6))
        if best is None or key > best[0]:
            best = (key, epoch + 1)
            torch.save(dict(state=model.state_dict(), standardizer=standardize.state(), settings=SETTINGS, variant=variant),
                       MODELS / f'{variant}.pt')
            np.save(MODELS / f'{variant}_validation_scores.npy', scores_v)
    record = dict(variant=variant, settings=SETTINGS, seed=SEED, selected_epoch=best[1], history=history,
                  answer_temperature=temperature if variant == 'C' else None,
                  model_sha256=sha256(MODELS / f'{variant}.pt'), validation_scores_sha256=sha256(MODELS / f'{variant}_validation_scores.npy'),
                  training_features_sha256=sha256(TRAIN), training_extra_sha256=sha256(MODEL_ROOT / 'train_extra.npy'),
                  validation_features_sha256=sha256(VAL), validation_extra_sha256=sha256(MODEL_ROOT / 'validation_extra.npy'),
                  option_vectors_sha256=sha256(OPTIONS) if variant == 'C' else None, training_questions=int(usable.sum()),
                  training_personas=len(set(data.personas.tolist())), feature_names=NAMES,
                  checkpoint_rule='highest V2 validation NDCG@10, ties by every-visible-reference coverage@15',
                  script_sha256=sha256(Path(__file__)), seconds=time.perf_counter() - started)
    write_json(OUTPUT / f'training_{variant}.json', record)
    print(f'{variant}: selected epoch {best[1]}', flush=True)
    data.close(); val.close()


def load_model(variant):
    import torch
    record = json.loads((OUTPUT / f'training_{variant}.json').read_text())
    if sha256(MODELS / f'{variant}.pt') != record['model_sha256']:
        raise ValueError('Saved neural ranker changed')
    saved = torch.load(MODELS / f'{variant}.pt', map_location='cuda', weights_only=False)
    model = network().cuda()
    model.load_state_dict(saved['state'])
    return Scorer(model, Standardizer(state=saved['standardizer'])), record


def stage_validate():
    targets = val_targets()
    qids = [q['example_id'] for q in targets['questions']]
    runs = {'A': np.load(MODEL_ROOT / 'validation_scores.npy')}
    for v in ('B', 'C'):
        if (OUTPUT / f'training_{v}.json').exists():
            record = json.loads((OUTPUT / f'training_{v}.json').read_text())
            if sha256(MODELS / f'{v}_validation_scores.npy') != record['validation_scores_sha256']:
                raise ValueError('Validation scores changed')
            runs[v] = np.load(MODELS / f'{v}_validation_scores.npy')
    report, rows = dict(questions=len(qids), methods={}, paired={}, by_type={}), {}
    for name, s in runs.items():
        report['methods'][name], rows[name] = val_metrics(s, targets)
    groups = np.array([r['persona_id'] for r in rows['A']])
    types = np.array([r['question_type'] for r in rows['A']])
    keys = ['ndcg_at_10', 'every_reference_at_5', 'every_reference_at_10', 'every_reference_at_15', 'precision_at_5']
    for x, z in (('B', 'A'), ('C', 'A'), ('C', 'B')):
        if x in rows and z in rows:
            out = {}
            for k in keys:
                d = np.array([float(p[k]) - float(q[k]) for p, q in zip(rows[x], rows[z])])
                out[k] = dict(difference=float(d.mean()), ci95=bootstrap(d, groups),
                              by_type={t: float(d[types == t].mean()) for t in sorted(set(types))},
                              persona_share_positive=float(np.mean([d[groups == g].mean() > 0 for g in np.unique(groups)])))
            report['paired'][f'{x} - {z}'] = out
    for name in rows:
        report['by_type'][name] = {t: {k: float(np.mean([r[k] for r, tt in zip(rows[name], types) if tt == t])) for k in keys}
                                   for t in sorted(set(types))}
    if 'B' in runs and 'C' in runs:
        overlap, same = [], []
        for i in range(len(qids)):
            a, b = map(int, targets['offsets'][i:i + 2])
            tb = set(np.argsort(-runs['B'][a:b], kind='stable')[:10].tolist())
            tc = set(np.argsort(-runs['C'][a:b], kind='stable')[:10].tolist())
            overlap.append(len(tb & tc) / 10); same.append(tb == tc)
        report['top10_B_vs_C'] = dict(mean_overlap=float(np.mean(overlap)), identical_sets=float(np.mean(same)))
    write_json(OUTPUT / 'validation.json', report)
    print(f"{'model':<6}" + ''.join(f'{k:>24}' for k in keys))
    for name, m in report['methods'].items():
        print(f'{name:<6}' + ''.join(f'{m[k]:>24.2%}' for k in keys))
    for pair, out in report['paired'].items():
        print(pair + ': ' + '; '.join(f"{k} {v['difference'] * 100:+.2f} ({v['ci95'][0] * 100:+.2f} to {v['ci95'][1] * 100:+.2f})"
                                      for k, v in out.items()))
    if 'top10_B_vs_C' in report:
        print('top-10 B vs C:', report['top10_B_vs_C'])


def stage_score(variant):
    scorer, record = load_model(variant)
    for dataset, (split, _) in DATASETS.items():
        path = CACHE / dataset / split
        scoring = json.loads((path / 'scoring.json').read_text())
        if scoring['features_sha256'] != sha256(path / 'features.npz'):
            raise ValueError(f'{dataset} features changed')
        data, X = inputs(path / 'features.npz', path / 'mathematical_features.npy')
        scores = scorer(data, X)
        out = path / f'neural_{variant}_scores.npy'
        np.save(out, scores)
        write_json(path / f'neural_{variant}_scoring.json',
                   dict(variant=variant, model_sha256=record['model_sha256'], features_sha256=sha256(path / 'features.npz'),
                        mathematical_features_sha256=sha256(path / 'mathematical_features.npy'), scores_sha256=sha256(out),
                        rows=len(scores), questions=len(data.qids)))
        print(f'{dataset}: scored {len(data.qids)} questions ({len(scores)} messages) with {variant}', flush=True)
        data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['options', 'check', 'train', 'validate', 'score'], required=True)
    parser.add_argument('--variant', choices=['B', 'C'])
    args = parser.parse_args()
    if args.stage in ('train', 'score') and not args.variant:
        parser.error('--variant is required for train and score')
    dict(options=stage_options, check=stage_check, validate=stage_validate)[args.stage]() if args.stage in (
        'options', 'check', 'validate') else (stage_train if args.stage == 'train' else stage_score)(args.variant)


if __name__ == '__main__':
    main()
