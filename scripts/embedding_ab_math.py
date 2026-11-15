"""Fit the two mathematical features within B, without redundancy-aware selection."""
import json
from pathlib import Path
import time

import numpy as np

from math_ranker_experiment import (NAMES, SETTINGS, extra_features, persona_folds,
                                    train_adapter, validate_cosine, vectors_on_gpu)
import ranker55 as recipe
from raw_datasets import sha256
from tune_ranker55 import Bundle, write_json


class CachedInputs:
    def __init__(self, path):
        self.path = Path(path)
        self.bundle = Bundle(path)
        self.meta = self.bundle.metadata
        self.qids = self.bundle.read('question_ids').tolist()
        self.personas = self.bundle.read('persona_ids')
        self.offsets, self.X = self.bundle.read('offsets'), self.bundle.read('X')
        if self.bundle.read('feature_names').tolist() != recipe.FEATURE_NAMES:
            raise ValueError('Mathematical features require the original 55-column bundle')
        root = Path(self.meta['embedding_cache_path'])
        with (root / 'messages.jsonl').open(encoding='utf-8') as stream:
            messages = {r['message_id']: r['embedding_index'] for r in map(json.loads, stream)}
        self.mi = np.fromiter((messages[str(mid)] for mid in self.bundle.read('message_ids')), dtype=np.int64)
        with (root / 'questions.jsonl').open(encoding='utf-8') as stream:
            questions = {r['example_id']: r for r in map(json.loads, stream)}
        self.qi = np.array([questions[q]['embedding_index'] for q in self.qids])
        self.types = [questions[q]['question_type'] for q in self.qids]
        for i, q in enumerate(self.qids):
            a, b = self.offsets[i:i+2]
            if self.bundle.read('message_ids')[a:b].tolist() != questions[q]['memory_ids']:
                raise ValueError('Vector-cache messages differ from feature groups')
        self.memory = np.load(root / 'memory_embeddings.npy', mmap_mode='r')
        self.query = np.load(root / 'question_embeddings.npy', mmap_mode='r')
        if not all(np.load(root / f'{kind}_done.npy').all() for kind in ['memory', 'question']):
            raise ValueError('Unfinished embedding cache')

    def close(self):
        self.bundle.__exit__(None, None, None)
        for array in [self.memory, self.query]:
            if getattr(array, '_mmap', None) is not None:
                array._mmap.close()


def fit(train_path, val_path, model_root, output, label):
    import torch
    import lightgbm as lgb
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_root, output = Path(model_root), Path(output)
    model_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in SETTINGS.items() if k not in ['mmr_penalty', 'memory_tokens']}
    data = CachedInputs(train_path)
    val = CachedInputs(val_path)
    if set(data.personas) & set(val.personas):
        raise ValueError('B and validation personas overlap')
    if data.meta['embedding_recipe'] != val.meta['embedding_recipe'] or data.meta['feature_recipe'] != val.meta['feature_recipe']:
        raise ValueError('B and validation recipes differ')
    memory, query = vectors_on_gpu(data)
    alignment = validate_cosine(data, memory, query)
    extra = np.full((len(data.X), 2), np.nan, dtype=np.float32)
    folds = persona_folds(data.personas, cfg['folds'], cfg['seed'])
    records = []
    for f in range(cfg['folds']):
        fit_ids, held = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        print(f'{label}: mathematical adapter fold {f+1}/5 on B', flush=True)
        U, V, loss = train_adapter(data, memory, query, fit_ids, cfg)
        bank_ids = np.random.default_rng(cfg['seed']).choice(np.unique(data.qi[fit_ids]), cfg['bank_size'], replace=False)
        extra_features(data, memory, query, U, V, query[bank_ids], held, cfg, extra)
        records.append(dict(fold=f, training_personas=np.unique(data.personas[fit_ids]).tolist(),
                            held_out_personas=np.unique(data.personas[held]).tolist(), final_loss=loss))
    if not np.isfinite(extra).all():
        raise ValueError('Missing OOF mathematical features')
    np.save(model_root / 'train_extra.npy', extra)
    print(f'{label}: final mathematical adapter on all B personas', flush=True)
    U, V, loss = train_adapter(data, memory, query, np.arange(len(data.qids)), cfg)
    bank_ids = np.random.default_rng(cfg['seed']).choice(np.unique(data.qi), cfg['bank_size'], replace=False)
    bank = query[bank_ids]
    np.savez(model_root / 'adapter.npz', U=U.cpu().numpy(), V=V.cpu().numpy(), bank=bank.cpu().numpy())
    vm, vq = vectors_on_gpu(val)
    val_alignment = validate_cosine(val, vm, vq)
    val_extra = np.full((len(val.X), 2), np.nan, dtype=np.float32)
    extra_features(val, vm, vq, U, V, bank, np.arange(len(val.qids)), cfg, val_extra)
    if not np.isfinite(val_extra).all():
        raise ValueError('Nonfinite validation mathematical features')
    np.save(model_root / 'validation_extra.npy', val_extra)
    del memory, query, vm, vq, U, V, bank
    torch.cuda.empty_cache()
    parameters = recipe.training_parameters(min_child_samples=100, num_leaves=63, max_depth=6, reg_lambda=10.)
    model = lgb.LGBMRanker(**parameters)
    print(f'{label}: fitting 57-feature ranker on {len(data.qids)} B questions', flush=True)
    weights = recipe.row_weights([.5 if t == 'ask_to_forget' else 1. for t in data.types], data.offsets)
    model.fit(np.column_stack([data.X, extra]), data.bundle.read('y'), group=np.diff(data.offsets),
              sample_weight=weights, feature_name=NAMES)
    path = model_root / 'ranker57.txt'
    model.booster_.save_model(str(path))
    scores = model.predict(np.column_stack([val.X, val_extra]), num_threads=4)
    np.save(model_root / 'validation_scores.npy', scores)
    record = dict(model_sha256=sha256(path), adapter_sha256=sha256(model_root / 'adapter.npz'),
                  training_features_sha256=sha256(train_path), validation_features_sha256=sha256(val_path),
                  score_file_sha256=sha256(model_root / 'validation_scores.npy'),
                  question_ids=data.qids, validation_question_ids=val.qids,
                  training_personas=np.unique(data.personas).tolist(), embedding_recipe=data.meta['embedding_recipe'],
                  feature_recipe=data.meta['feature_recipe'], feature_names=NAMES, settings=cfg,
                  folds=records, cosine_alignment_max_error=alignment, validation_cosine_alignment_max_error=val_alignment,
                  seconds=time.perf_counter()-started, parameters=parameters, redundancy_selection_used=False,
                  script_sha256=sha256(Path(__file__)))
    write_json(output / f'{label}_ranker57.json', record)
    data.close()
    val.close()
    return record
