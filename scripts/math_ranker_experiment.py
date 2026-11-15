"""Train the combined mathematical-feature pilot on V2 train only.

python scripts/math_ranker_experiment.py
Then: python scripts/evaluate_math_ranker.py
No validation/test features, labels or metrics are used by training. Outputs are experimental.
"""
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np

import ranker55 as recipe
from raw_datasets import ROOT, sha256
from tune_ranker55 import Bundle, write_json

OUTPUT = ROOT / 'results/math_ranker/v2/test/combined'
MODEL_ROOT = ROOT / 'models/math_ranker'
CACHE = ROOT / 'data/ranker55/math_ranker'
NAMES = recipe.FEATURE_NAMES + ['learned_bilinear_similarity', 'background_corrected_cosine']
SETTINGS = dict(seed=271, folds=5, rank=16, epochs=3, batch_size=128,
                learning_rate=0.003, weight_decay=0.01, temperature=0.05,
                negatives=8, hard_pool=30, bank_size=1024, background_k=10,
                background_penalty=0.5, mmr_penalty=0.1, memory_tokens=2048)


def persona_folds(personas, folds=5, seed=271):
    unique = np.unique(personas)
    shuffled = np.random.default_rng(seed).permutation(unique)
    assignment = {p: i % folds for i, p in enumerate(shuffled)}
    return np.array([assignment[p] for p in personas])


def normalized_scores(scores):
    """Keep a bounded relevance scale so MMR's penalty is interpretable."""
    scores = np.asarray(scores, dtype=float)
    return (scores - scores.min()) / max(float(np.ptp(scores)), 1e-12)


def mmr_choice(relevance, redundancy, available, penalty):
    gains = relevance - penalty * redundancy
    return int(np.argmax(np.where(available, gains, -np.inf)))


class Inputs:
    """Bind complete NPZ groups to the builder's deduplicated vector caches."""
    def __init__(self, split):
        self.path = ROOT / f'data/ranker55/v2/{split}/features.npz'
        self.bundle = Bundle(self.path)
        self.meta = self.bundle.metadata
        if (self.meta['dataset'], self.meta['split'], self.meta['memory_unit']) != ('v2', split, 'single_message'):
            raise ValueError('Wrong feature cohort')
        if self.bundle.read('feature_names').tolist() != recipe.FEATURE_NAMES:
            raise ValueError('Expected original 55-feature order')
        for path, expected in self.meta['source_sha256'].items():
            # Other split IDs/persona_split were changed by evaluation exclusions;
            # they neither contribute to this split's features nor belong in training.
            if path.startswith('data/splits/') and path != f'data/splits/v2/{split}_ids.json':
                continue
            if sha256(ROOT / path) != expected:
                raise ValueError(f'Stale source identity: {path}')
        self.qids = self.bundle.read('question_ids').tolist()
        active = json.loads((ROOT / f'data/splits/v2/{split}_ids.json').read_text())
        if set(active) != set(self.qids) or len(active) != len(self.qids):
            raise ValueError('Feature questions differ from active split')
        self.personas = self.bundle.read('persona_ids')
        self.offsets = self.bundle.read('offsets')
        self.X = self.bundle.read('X')
        root = Path(self.meta['embedding_cache_path'])
        records = [json.loads(line) for line in (root / 'messages.jsonl').open(encoding='utf-8')]
        mapping = {r['message_id']: r['embedding_index'] for r in records}
        self.mi = np.fromiter((mapping[str(mid)] for mid in self.bundle.read('message_ids')), dtype=np.int64)
        qrecords = {r['example_id']: r for r in map(json.loads, (root / 'questions.jsonl').open(encoding='utf-8'))}
        self.qi = np.array([qrecords[q]['embedding_index'] for q in self.qids])
        self.types = [qrecords[q]['question_type'] for q in self.qids]
        for i, qid in enumerate(self.qids):
            a, b = self.offsets[i:i+2]
            if self.bundle.read('message_ids')[a:b].tolist() != qrecords[qid]['memory_ids']:
                raise ValueError(f'{qid}: vector cache pool differs from feature rows')
        self.memory = np.load(root / 'memory_embeddings.npy', mmap_mode='r')
        self.query = np.load(root / 'question_embeddings.npy', mmap_mode='r')
        if not all(np.load(root / f'{kind}_done.npy').all() for kind in ['memory', 'question']):
            raise ValueError('Incomplete embeddings')
        self.cache_hashes = {str(root / f): sha256(root / f) for f in
                             ['memory_embeddings.npy', 'question_embeddings.npy', 'messages.jsonl', 'questions.jsonl']}


def vectors_on_gpu(data):
    import torch
    return (torch.tensor(np.asarray(data.memory), device='cuda'),
            torch.tensor(np.asarray(data.query), device='cuda'))


def validate_cosine(data, memory, query):
    """Check every question/row mapping against the original cosine feature."""
    import torch
    error = 0.
    for i in range(len(data.qids)):
        a, b = map(int, data.offsets[i:i+2])
        with torch.no_grad():
            values = (memory[data.mi[a:b]] @ query[data.qi[i]]).cpu().numpy()
        error = max(error, float(np.max(np.abs(values - data.X[a:b, 0]))))
    if error > 3e-5:
        raise ValueError(f'Cached vector alignment/cosine mismatch: {error}')
    return error


def train_adapter(data, memory, query, selected, cfg):
    """Pairwise reference preference; unlisted messages are noisy contrast examples."""
    import torch
    rng = np.random.default_rng(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    U = torch.nn.Parameter(torch.randn(768, cfg['rank'], device='cuda') * 0.02)
    V = torch.nn.Parameter(torch.zeros(768, cfg['rank'], device='cuda'))
    optimizer = torch.optim.AdamW([U, V], lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    y = data.bundle.read('y')
    pools = {}
    for i in selected:
        a, b = map(int, data.offsets[i:i+2])
        positive = np.unique(data.mi[a:b][y[a:b] == 1])
        eligible = np.flatnonzero((y[a:b] == 0) & ~np.isin(data.mi[a:b], positive))
        hard = eligible[np.argsort(-data.X[a:b, 0][eligible], kind='stable')[:cfg['hard_pool']]]
        if not len(positive) or not len(hard):
            raise ValueError('Training group needs positives and contrast examples')
        pools[int(i)] = positive, data.mi[a:b][hard], data.mi[a:b][eligible]
    losses = []
    for epoch in range(cfg['epochs']):
        order = rng.permutation(selected)
        for start in range(0, len(order), cfg['batch_size']):
            batch = order[start:start + cfg['batch_size']]
            candidates = []
            for i in batch:
                pos, hard, rest = pools[int(i)]
                candidates.append([rng.choice(pos), *rng.choice(hard, cfg['negatives'] - 1), rng.choice(rest)])
            q = query[data.qi[batch]]
            m = memory[torch.tensor(candidates, device='cuda')]
            scores = (torch.einsum('bd,bnd->bn', q, m)
                      + torch.einsum('br,bnr->bn', q @ U, m @ V))
            terms = torch.nn.functional.softplus((scores[:, 1:] - scores[:, :1]) / cfg['temperature']).mean(1)
            weights = torch.tensor([0.5 if data.types[i] == 'ask_to_forget' else 1. for i in batch], device='cuda')
            loss = (terms * weights).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([U, V], 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f'  epoch {epoch + 1}: loss {np.mean(losses[-max(1, len(order)//cfg["batch_size"]):]):.4f}', flush=True)
    return U.detach(), V.detach(), float(np.mean(losses[-50:]))


def background(memory, bank, k):
    import torch
    values = []
    with torch.no_grad():
        for start in range(0, len(memory), 2048):
            similarity = memory[start:start+2048] @ bank.T
            values.append(similarity.topk(min(k, len(bank)), dim=1).values.mean(1).cpu().numpy())
    return np.concatenate(values)


def extra_features(data, memory, query, U, V, bank, selected, cfg, result):
    import torch
    bkg = background(memory, bank, cfg['background_k'])
    with torch.no_grad():
        qm, mm = query @ U, memory @ V
        for i in selected:
            a, b = map(int, data.offsets[i:i+2])
            ids = data.mi[a:b]
            residual = (mm[ids] @ qm[data.qi[i]]).cpu().numpy()
            result[a:b, 0] = data.X[a:b, 0] + residual
            result[a:b, 1] = data.X[a:b, 0] - cfg['background_penalty'] * bkg[ids]


def train():
    import torch
    import lightgbm as lgb
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    for path in [OUTPUT, MODEL_ROOT, CACHE]:
        path.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'predictions.jsonl').exists():
        raise ValueError('A test run exists; do not retrain after looking at test outcomes')
    cfg = SETTINGS.copy()
    write_json(OUTPUT / 'settings.json', cfg)
    data = Inputs('train')
    memory, query = vectors_on_gpu(data)
    alignment = validate_cosine(data, memory, query)
    extra = np.full((len(data.X), 2), np.nan, dtype=np.float32)
    folds = persona_folds(data.personas, cfg['folds'], cfg['seed'])
    fold_records = []
    for f in range(cfg['folds']):
        fit, held = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        print(f'Fold {f + 1}/{cfg["folds"]}: {len(fit)} fit, {len(held)} held-out questions', flush=True)
        U, V, loss = train_adapter(data, memory, query, fit, cfg)
        bank_ids = np.random.default_rng(cfg['seed']).choice(np.unique(data.qi[fit]), cfg['bank_size'], replace=False)
        extra_features(data, memory, query, U, V, query[bank_ids], held, cfg, extra)
        record = dict(fold=f, training_personas=np.unique(data.personas[fit]).tolist(),
                      held_out_personas=np.unique(data.personas[held]).tolist(), final_loss=loss)
        fold_records.append(record)
        np.savez(MODEL_ROOT / f'adapter_fold{f}.npz', U=U.cpu().numpy(), V=V.cpu().numpy(), bank_ids=bank_ids)
    if not np.isfinite(extra).all():
        raise ValueError('Missing/nonfinite OOF scores')
    np.save(CACHE / 'train_extra.npy', extra)
    print('Training final adapter on every training persona', flush=True)
    U, V, loss = train_adapter(data, memory, query, np.arange(len(data.qids)), cfg)
    bank_ids = np.random.default_rng(cfg['seed']).choice(np.unique(data.qi), cfg['bank_size'], replace=False)
    np.savez(MODEL_ROOT / 'adapter.npz', U=U.cpu().numpy(), V=V.cpu().numpy(), bank=np.asarray(data.query)[bank_ids])
    parameters = recipe.training_parameters(min_child_samples=100, num_leaves=63, max_depth=6, reg_lambda=10.)
    X = np.column_stack([data.X, extra])
    model = lgb.LGBMRanker(**parameters)
    weights = recipe.row_weights([0.5 if t == 'ask_to_forget' else 1. for t in data.types], data.offsets)
    print(f'Fitting 57-feature LightGBM on {len(X):,} rows', flush=True)
    model.fit(X, data.bundle.read('y'), group=np.diff(data.offsets), sample_weight=weights, feature_name=NAMES)
    model.booster_.save_model(str(MODEL_ROOT / 'ranker57.txt'))
    record = dict(status='complete', settings=cfg, train_questions=len(data.qids), train_rows=len(X),
                  feature_names=NAMES, feature_recipe=data.meta['feature_recipe'],
                  training_features_sha256=sha256(data.path), embedding_cache_sha256=data.cache_hashes,
                  train_personas=np.unique(data.personas).tolist(), folds=fold_records,
                  cosine_alignment_max_error=alignment, final_adapter_loss=loss,
                  lightgbm_parameters=parameters, seconds=time.perf_counter()-started,
                  model_sha256=sha256(MODEL_ROOT / 'ranker57.txt'),
                  adapter_sha256=sha256(MODEL_ROOT / 'adapter.npz'),
                  script_sha256=sha256(Path(__file__)),
                  evaluation_splits_read=[], validation_used=False, test_used=False,
                  label_policy='Exact reference membership; unlisted messages are noisy negatives, not proven irrelevant.')
    write_json(OUTPUT / 'training.json', record)
    print(f'Training complete in {record["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    train()
