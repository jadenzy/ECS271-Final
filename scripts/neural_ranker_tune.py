"""Hyperparameter search with early stopping for the neural ranker (scripts/neural_ranker.py), V2 validation untouched.

python scripts/neural_ranker_tune.py --stage search            # random search, reference loss (variant B)
python scripts/neural_ranker_tune.py --stage answer-weight     # best B settings, sweep the answer-loss weight (variant C)
python scripts/neural_ranker_tune.py --stage final             # retrain both winners on all of B, score V2 validation once
python scripts/neural_ranker_tune.py --stage score --variant B_tuned   # V2 test, V1, LoCoMo scores for evaluate_neural_ranker.py

Tuning set: one of five persona folds of half B (seed 271, about 1,840 questions); the other four train. Each
configuration trains up to MAX_EPOCHS and stops after PATIENCE epochs without a better tuning NDCG@10 (ties:
every-reference coverage@15); its best epoch is recorded. The search samples SEARCH configurations at random from
GRID plus the original setting. The final models retrain on all 9,192 half-B questions for the chosen number of
epochs, with no early stopping, and are then scored once on the 909 V2 validation questions, which no tuning step
reads. Models: models/neural_ranker/{B,C}_tuned.pt; records: results/neural_ranker/tuning_*.json,
training_{B,C}_tuned.json, validation_tuned.json.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from embedding_ab_experiment import OUTPUT as AB_OUTPUT
from evaluate_embedding_ab_transfer import CACHE, DATASETS, MODEL_ROOT
from math_ranker_experiment import NAMES, persona_folds
from neural_ranker import (MODELS, OPTIONS, OUTPUT, SETTINGS, TRAIN, VAL, Scorer, Standardizer, centered_similarity, inputs,
                           option_vectors, val_metrics, val_targets)
from option_rotation_pilot import bootstrap
from raw_datasets import sha256
from tune_ranker55 import write_json

SEED, MAX_EPOCHS, PATIENCE, SEARCH = 271, 20, 3, 20
GRID = dict(learning_rate=[3e-4, 1e-3, 3e-3], hidden=[64, 128, 256], layers=[1, 2], projection=[32, 64, 128],
            dropout=[0.0, 0.1, 0.3], weight_decay=[0.01, 0.1], batch_questions=[16, 32])
ORIGINAL = dict(learning_rate=1e-3, hidden=128, layers=1, projection=64, dropout=0.0, weight_decay=0.01, batch_questions=16)
ANSWER_WEIGHTS = [0.1, 0.25, 0.5, 1.0]


def network(cfg):
    import torch
    nn = torch.nn

    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(768, cfg['projection'], bias=False)
            width, layers = len(NAMES) + 2 * cfg['projection'], []
            for _ in range(cfg['layers']):
                layers += [nn.Linear(width, cfg['hidden']), nn.ReLU(), nn.Dropout(cfg['dropout'])]
                width = cfg['hidden']
            self.score = nn.Sequential(*layers, nn.Linear(width, 1))

        def forward(self, x, m, q):
            pm, pq = self.project(m), self.project(q)
            return self.score(torch.cat([x, pm * pq, (pm - pq).abs()], 1)).squeeze(1)

    return Ranker()


class Half:
    """Half B on the GPU, plus the answer options for the answer loss."""
    def __init__(self):
        import torch
        self.data, X = inputs(TRAIN, MODEL_ROOT / 'train_extra.npy')
        if set(self.data.qids) != set(json.loads((AB_OUTPUT / 'partition.json').read_text())['b_question_ids']):
            raise ValueError('Training must use exactly persona half B')
        self.standardize = Standardizer(X)
        d = self.data
        self.y = d.bundle.read('y').astype(bool)
        self.offsets = np.asarray(d.offsets, dtype=np.int64)
        self.Z = torch.tensor(self.standardize(X), device='cuda')
        self.memory = torch.tensor(np.asarray(d.memory), device='cuda')
        self.query = torch.tensor(np.asarray(d.query), device='cuda')
        self.mi = torch.tensor(d.mi, device='cuda')
        self.qi = np.asarray(d.qi)
        self.weights = np.array([SETTINGS['forget_weight'] if t == 'ask_to_forget' else 1. for t in d.types])
        self.usable = np.array([self.y[self.offsets[i]:self.offsets[i + 1]].any() for i in range(len(d.qids))])
        check = json.loads((OUTPUT / 'signal_b_check.json').read_text())
        self.temperature = check['answer_temperature']
        vectors, index = option_vectors()
        self.options = [torch.tensor(vectors[index['questions'][q]['rows']], device='cuda') for q in d.qids]
        self.gold = [index['questions'][q]['gold'] for q in d.qids]
        folds = persona_folds(d.personas, 5, SEED)
        self.tune = np.flatnonzero(folds == 0)
        self.fit = np.flatnonzero((folds != 0) & self.usable)

    def rows(self, batch):
        spans = [(int(self.offsets[i]), int(self.offsets[i + 1])) for i in batch]
        rows = np.concatenate([np.arange(a, b) for a, b in spans])
        qrow = np.concatenate([np.full(b - a, self.qi[i]) for i, (a, b) in zip(batch, spans)])
        return spans, rows, qrow

    def epoch(self, model, optimizer, questions, cfg, rng, answer_weight):
        import torch
        model.train()
        order = rng.permutation(questions)
        for s in range(0, len(order), cfg['batch_questions']):
            batch = np.sort(order[s:s + cfg['batch_questions']])
            spans, rows, qrow = self.rows(batch)
            rows_t = torch.tensor(rows, device='cuda')
            scores = model(self.Z[rows_t], self.memory[self.mi[rows_t]], self.query[torch.tensor(qrow, device='cuda')])
            loss_ref, loss_ans, start = [], [], 0
            for i, (a, b) in zip(batch, spans):
                logp = torch.log_softmax(scores[start:start + b - a], 0)
                loss_ref.append(-logp[torch.tensor(np.flatnonzero(self.y[a:b]), device='cuda')].mean() * self.weights[i])
                if answer_weight and self.data.types[i] not in SETTINGS['answer_loss_excludes']:
                    evidence = logp.exp() @ centered_similarity(self.memory[self.mi[a:b]], self.options[i]) / self.temperature
                    loss_ans.append(torch.nn.functional.cross_entropy(evidence[None], torch.tensor([self.gold[i]], device='cuda')))
                start += b - a
            loss = torch.stack(loss_ref).mean()
            if loss_ans:
                loss = loss + answer_weight * torch.stack(loss_ans).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def evaluate(self, model, questions):
        """Tuning-set NDCG@10 and every-reference coverage@15 (visible references from the bundle labels)."""
        import torch
        model.eval()
        discounts = 1 / np.log2(np.arange(2, 12))
        ndcg, cover = [], []
        with torch.no_grad():
            for s in range(0, len(questions), 64):
                batch = questions[s:s + 64]
                spans, rows, qrow = self.rows(batch)
                rows_t = torch.tensor(rows, device='cuda')
                out = model(self.Z[rows_t], self.memory[self.mi[rows_t]], self.query[torch.tensor(qrow, device='cuda')]).cpu().numpy()
                start = 0
                for a, b in spans:
                    sc, pos = out[start:start + b - a], self.y[a:b]
                    start += b - a
                    if not pos.any():
                        continue
                    order = np.argsort(-sc, kind='stable')
                    ndcg.append(float(np.sum(discounts[:min(10, b - a)] * pos[order[:10]]) / discounts[:min(10, int(pos.sum()))].sum()))
                    ranks = np.empty(b - a, int); ranks[order] = np.arange(1, b - a + 1)
                    cover.append(bool(np.all(ranks[pos] <= 15)))
        return float(np.mean(ndcg)), float(np.mean(cover))

    def train(self, cfg, questions, answer_weight=0.0, epochs=None, label=''):
        """Early stopping on the tuning fold when epochs is None; otherwise a fixed number of epochs."""
        import torch
        torch.manual_seed(SEED)
        model = network(cfg).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        rng = np.random.default_rng(SEED)
        history, best, state = [], None, None
        for e in range(epochs or MAX_EPOCHS):
            self.epoch(model, optimizer, questions, cfg, rng, answer_weight)
            if epochs is not None:
                continue
            ndcg, cover = self.evaluate(model, self.tune)
            history.append(dict(epoch=e + 1, ndcg_at_10=ndcg, every_reference_at_15=cover))
            if best is None or (round(ndcg, 6), round(cover, 6)) > best[0]:
                best = ((round(ndcg, 6), round(cover, 6)), e + 1)
            elif e + 1 - best[1] >= PATIENCE:
                break
        if epochs is not None:
            return model
        return dict(config=cfg, answer_weight=answer_weight, best_epoch=best[1], tuning_ndcg_at_10=best[0][0],
                    tuning_every_reference_at_15=best[0][1], epochs_run=len(history), history=history, label=label)


def sample_configs():
    rng = np.random.default_rng(SEED)
    configs, seen = [dict(ORIGINAL)], {json.dumps(ORIGINAL, sort_keys=True)}
    while len(configs) < SEARCH + 1:
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in GRID.items()}
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key); configs.append(cfg)
    return configs


def stage_search():
    started = time.perf_counter()
    half = Half()
    results = []
    print(f'Tuning fold: {len(half.tune)} questions; training folds: {len(half.fit)} questions', flush=True)
    for n, cfg in enumerate(sample_configs()):
        r = half.train(cfg, half.fit, label='original' if n == 0 else f'search {n}')
        results.append(r)
        print(f"{n:>2} {r['label']:<9} {json.dumps(cfg)}: tuning NDCG@10 {r['tuning_ndcg_at_10']:.2%}, every ref@15 "
              f"{r['tuning_every_reference_at_15']:.2%}, best epoch {r['best_epoch']}/{r['epochs_run']} "
              f"({(time.perf_counter() - started) / 60:.1f} min)", flush=True)
    best = max(results, key=lambda r: (r['tuning_ndcg_at_10'], r['tuning_every_reference_at_15']))
    write_json(OUTPUT / 'tuning_search.json', dict(grid=GRID, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED,
                                                    tuning_questions=len(half.tune), training_questions=len(half.fit),
                                                    results=results, best=best, script_sha256=sha256(Path(__file__)),
                                                    seconds=time.perf_counter() - started))
    print(f"best: {json.dumps(best['config'])}, epoch {best['best_epoch']}, tuning NDCG@10 {best['tuning_ndcg_at_10']:.2%}")


def stage_answer_weight():
    started = time.perf_counter()
    search = json.loads((OUTPUT / 'tuning_search.json').read_text())
    cfg, base = search['best']['config'], search['best']
    half = Half()
    results = [dict(base, answer_weight=0.0, label='B best')]
    for w in ANSWER_WEIGHTS:
        r = half.train(cfg, half.fit, answer_weight=w, label=f'weight {w:g}')
        results.append(r)
        print(f"answer weight {w:g}: tuning NDCG@10 {r['tuning_ndcg_at_10']:.2%}, every ref@15 "
              f"{r['tuning_every_reference_at_15']:.2%}, best epoch {r['best_epoch']}/{r['epochs_run']} "
              f"({(time.perf_counter() - started) / 60:.1f} min)", flush=True)
    best = max(results[1:], key=lambda r: (r['tuning_ndcg_at_10'], r['tuning_every_reference_at_15']))
    write_json(OUTPUT / 'tuning_answer_weight.json', dict(config=cfg, weights=ANSWER_WEIGHTS, results=results, best=best,
                                                           script_sha256=sha256(Path(__file__)), seconds=time.perf_counter() - started))
    print(f"best answer weight {best['answer_weight']:g} (tuning NDCG@10 {best['tuning_ndcg_at_10']:.2%} vs B {base['tuning_ndcg_at_10']:.2%})")


def stage_final():
    import torch
    started = time.perf_counter()
    search = json.loads((OUTPUT / 'tuning_search.json').read_text())
    weight = json.loads((OUTPUT / 'tuning_answer_weight.json').read_text())
    half = Half()
    every = np.flatnonzero(half.usable)
    val, Xv = inputs(VAL, MODEL_ROOT / 'validation_extra.npy')
    targets = val_targets()
    runs = {'A': np.load(MODEL_ROOT / 'validation_scores.npy'), 'B': np.load(MODELS / 'B_validation_scores.npy'),
            'C': np.load(MODELS / 'C_validation_scores.npy')}
    for variant, chosen in (('B_tuned', search['best']), ('C_tuned', weight['best'])):
        cfg, epochs, w = chosen['config'], chosen['best_epoch'], chosen['answer_weight']
        model = half.train(cfg, every, answer_weight=w, epochs=epochs)
        path = MODELS / f'{variant}.pt'
        torch.save(dict(state=model.state_dict(), standardizer=half.standardize.state(), config=cfg, answer_weight=w,
                        epochs=epochs, variant=variant), path)
        scores = Scorer(model, half.standardize)(val, Xv)
        np.save(MODELS / f'{variant}_validation_scores.npy', scores)
        runs[variant] = scores
        write_json(OUTPUT / f'training_{variant}.json',
                   dict(variant=variant, config=cfg, answer_weight=w, epochs=epochs, seed=SEED, model_sha256=sha256(path),
                        validation_scores_sha256=sha256(MODELS / f'{variant}_validation_scores.npy'),
                        answer_temperature=half.temperature if w else None, answer_loss_excludes=SETTINGS['answer_loss_excludes'],
                        training_features_sha256=sha256(TRAIN), training_extra_sha256=sha256(MODEL_ROOT / 'train_extra.npy'),
                        option_vectors_sha256=sha256(OPTIONS) if w else None, training_questions=len(every),
                        selection='tuning fold of half B (persona fold 0); V2 validation not used for any choice',
                        script_sha256=sha256(Path(__file__))))
        print(f'{variant}: trained on all of B for {epochs} epochs ({(time.perf_counter() - started) / 60:.1f} min)', flush=True)
    keys = ['ndcg_at_10', 'every_reference_at_5', 'every_reference_at_10', 'every_reference_at_15', 'precision_at_5']
    metrics, rows = {}, {}
    for name, s in runs.items():
        metrics[name], rows[name] = val_metrics(s, targets)
    groups = np.array([r['persona_id'] for r in rows['A']])
    paired = {}
    for x, z in (('B_tuned', 'A'), ('C_tuned', 'A'), ('B_tuned', 'B'), ('C_tuned', 'C'), ('C_tuned', 'B_tuned')):
        paired[f'{x} - {z}'] = {k: (lambda d: dict(difference=float(d.mean()), ci95=bootstrap(d, groups)))(
            np.array([float(p[k]) - float(q[k]) for p, q in zip(rows[x], rows[z])])) for k in keys}
    write_json(OUTPUT / 'validation_tuned.json', dict(questions=len(rows['A']), methods=metrics, paired=paired))
    print(f"{'model':<9}" + ''.join(f'{k:>24}' for k in keys))
    for name, m in metrics.items():
        print(f'{name:<9}' + ''.join(f'{m[k]:>24.2%}' for k in keys))
    for pair, out in paired.items():
        print(pair + ': ' + '; '.join(f"{k} {v['difference'] * 100:+.2f} ({v['ci95'][0] * 100:+.2f} to {v['ci95'][1] * 100:+.2f})"
                                      for k, v in out.items()))
    val.close()


def stage_score(variant):
    import torch
    record = json.loads((OUTPUT / f'training_{variant}.json').read_text())
    if sha256(MODELS / f'{variant}.pt') != record['model_sha256']:
        raise ValueError('Saved tuned ranker changed')
    saved = torch.load(MODELS / f'{variant}.pt', map_location='cuda', weights_only=False)
    model = network(saved['config']).cuda()
    model.load_state_dict(saved['state'])
    scorer = Scorer(model, Standardizer(state=saved['standardizer']))
    for dataset, (split, _) in DATASETS.items():
        path = CACHE / dataset / split
        data, X = inputs(path / 'features.npz', path / 'mathematical_features.npy')
        scores = scorer(data, X)
        out = path / f'neural_{variant}_scores.npy'
        np.save(out, scores)
        write_json(path / f'neural_{variant}_scoring.json',
                   dict(variant=variant, model_sha256=record['model_sha256'], features_sha256=sha256(path / 'features.npz'),
                        mathematical_features_sha256=sha256(path / 'mathematical_features.npy'), scores_sha256=sha256(out),
                        rows=len(scores), questions=len(data.qids)))
        print(f'{dataset}: scored {len(data.qids)} questions with {variant}', flush=True)
        data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['search', 'answer-weight', 'final', 'score'], required=True)
    parser.add_argument('--variant', choices=['B_tuned', 'C_tuned'])
    args = parser.parse_args()
    if args.stage == 'score':
        if not args.variant:
            parser.error('--variant is required for score')
        stage_score(args.variant)
    else:
        dict(search=stage_search, final=stage_final)[args.stage]() if args.stage != 'answer-weight' else stage_answer_weight()


if __name__ == '__main__':
    main()
