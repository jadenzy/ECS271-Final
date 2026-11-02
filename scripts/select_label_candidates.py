"""Select 20 distinct non-reference messages per V2 training question for Qwen.

python scripts/select_label_candidates.py --features data/ranker55/v2/train/features.npz
Loads models/subtlex_mix/alpha_0.5/ranker55.txt; never trains a model or calls Qwen. The CLI defaults to
all cached queries; --limit-questions 100 selects a deterministic pilot.
References retain label 2. Selected and unselected non-references remain unjudged.

Each question gets 12 ranker, 3 weighted BM25, 3 dense chunk and 2 random messages. The lexical and
dense picks find evidence the ranker misses; the random picks give ordinary negatives for ranking the
full history.

The NPZ must contain X, feature_names, offsets, question_ids (one per group), and
message_ids (one per X row), computed over each question's COMPLETE visible pool
with the saved model's feature recipe/statistics. Cached y is deliberately ignored.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from label_reference_messages import REFERENCE_LABEL, label_question, load_references
from message_dataset import SPLIT_ROOT, load_message_dataset
from raw_datasets import ROOT, read_rows, sha256
from feature_contract import bundle_recipe, bundle_embedding_root, check_model_recipe, validate_recipe

# One authoritative feature schema; no copied model weights or training code.
import ranker55 as _recipe
FEATURE_NAMES = _recipe.FEATURE_NAMES
ALLOCATION = {'ranker': 12, 'qweight_bm25': 3, 'dense_chunk': 3, 'random': 2}
BUDGET = sum(ALLOCATION.values())


def stable_seed(seed, example_id):
    """Independent per-question randomness, unaffected by pilot size or row order."""
    return int.from_bytes(hashlib.sha256(f'{seed}\0{example_id}'.encode()).digest()[:8], 'big')


def string_vector(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind != 'U' or any(not s for s in value):
        raise ValueError(f'{name} must be a one-dimensional array of nonempty Unicode strings')
    return value.tolist()


def load_features(path, questions, sources=None, *, dataset='v2', split='train', require_metadata=False):
    """Validate a complete visible-pool cache; annotation defaults remain V2 train."""
    data = np.load(path, allow_pickle=False)
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError('Features must be an NPZ archive')
    required = ('X', 'feature_names', 'offsets', 'question_ids', 'message_ids')
    with data:
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError('Missing feature arrays: ' + ', '.join(sorted(missing)))
        X, names, offsets, qids, mids = (data[key] for key in required)
        if require_metadata and 'metadata_json' not in data.files:
            raise ValueError('Evaluation features require builder metadata_json and source hashes')
        if 'metadata_json' in data.files:
            metadata = json.loads(str(data['metadata_json'].item()))
            validate_recipe(metadata.get('feature_recipe'))
            if require_metadata and metadata.get('memory_unit') != 'single_message':
                raise ValueError('Evaluation features must use the single_message retrieval unit')
            if metadata.get('dataset') != dataset or metadata.get('split') != split:
                raise ValueError(f'Features must be from the {dataset}/{split} split')
            recorded = metadata.get('source_sha256', {})
            if sources is not None and any(recorded.get(name) != digest for name, digest in sources.items()):
                raise ValueError('Feature source hashes differ from the active raw data/splits; rebuild features')
        else:
            raise ValueError('Features require SUBTLEX recipe metadata; rebuild older artifacts')
    names = string_vector(names, 'feature_names')
    if names not in (FEATURE_NAMES, FEATURE_NAMES[:-1]):
        raise ValueError('feature_names must match the ordered 55- or 54-feature recipe')
    if X.ndim != 2 or X.shape[1] != len(names) or not len(X) or X.dtype.kind not in 'fiu':
        raise ValueError('X must be a nonempty real feature matrix')
    X = np.asarray(X, dtype=np.float32)
    if any(not np.isfinite(X[i:i + 65536]).all() for i in range(0, len(X), 65536)):
        raise ValueError('X contains nonfinite features')
    qids, mids = string_vector(qids, 'question_ids'), string_vector(mids, 'message_ids')
    if (offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets) != len(qids) + 1
            or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(X)
            or np.any(offsets[1:] <= offsets[:-1]) or len(mids) != len(X)):
        raise ValueError('offsets must partition every X row into nonempty question groups')
    if len(set(qids)) != len(qids):
        raise ValueError('Duplicate question_ids in feature cache')
    by_id = {q['example_id']: q for q in questions}
    if len(by_id) != len(questions):
        raise ValueError('Duplicate active question IDs')
    for i, qid in enumerate(qids):
        if qid not in by_id or by_id[qid]['split'] != split or by_id[qid].get('dataset') != dataset:
            raise ValueError(f'{qid}: not an active {dataset}/{split} question')
        start, end = int(offsets[i]), int(offsets[i + 1])
        if mids[start:end] != by_id[qid]['memory_ids']:
            raise ValueError(f'{qid}: message_ids must equal the complete visible pool in source order')
    return X, names, offsets, qids


class SavedModels:
    """Load a single saved booster, or route personas to saved held-persona boosters.

    Optional manifest: {"models": [{"path": "fold0.txt", "sha256": "...",
    "training_persona_ids": ["v2:1"], "held_out_persona_ids": ["v2:2"]}, ...]}.
    Paths are relative to the manifest. Hashes bind training declarations to files.
    A single model has unknown training overlap; never call its scores OOF.
    """
    def __init__(self, model_path, fold_manifest, feature_names, *, feature_recipe=None):
        import lightgbm as lgb
        self.models, self.routes, self.records = {}, {}, []
        self.mode = 'single_model_training_overlap_unknown'
        self.input_paths = []
        entries = [{'path': str(Path(model_path).resolve())}] if fold_manifest is None else None
        if fold_manifest is not None:
            manifest = Path(fold_manifest).resolve()
            self.input_paths.append(manifest)
            document = json.loads(manifest.read_text(encoding='utf-8'))
            entries = document.get('models')
            if not isinstance(entries, list) or not entries:
                raise ValueError('Fold manifest must contain a nonempty models list')
            self.mode = 'persona_disjoint_saved_models'
        for number, entry in enumerate(entries):
            path = Path(entry['path'])
            if fold_manifest is not None and not path.is_absolute():
                path = manifest.parent / path
            path = path.resolve()
            digest = sha256(path)
            record = {'path': str(path), 'sha256': digest}
            contract = feature_recipe
            sidecar = check_model_recipe(path, contract)
            record['feature_recipe'] = contract
            if sidecar is not None:
                self.input_paths.append(sidecar)
            if fold_manifest is not None:
                train, held = entry.get('training_persona_ids'), entry.get('held_out_persona_ids')
                if any(not isinstance(v, list) or not v or
                       any(not isinstance(p, str) or not p for p in v) or len(set(v)) != len(v)
                       for v in (train, held)):
                    raise ValueError('Each fold needs distinct training and held-out persona IDs')
                if set(train) & set(held):
                    raise ValueError('Fold training and held-out personas overlap')
                if entry.get('sha256') != digest:
                    raise ValueError(f'Fold model hash mismatch: {path}')
                for persona in held:
                    if persona in self.routes:
                        raise ValueError(f'{persona}: assigned to multiple held-out models')
                    self.routes[persona] = number
                record.update(training_persona_ids=train, held_out_persona_ids=held)
            # Normalize Windows CRLF before handing text to LightGBM; its file
            # reader's tree byte offsets assume the original LF serialization.
            booster = lgb.Booster(model_str=path.read_text(encoding='utf-8'))
            names = booster.feature_name()
            if names not in (FEATURE_NAMES, FEATURE_NAMES[:-1]) or feature_names[:len(names)] != names:
                raise ValueError(f'{path}: saved model feature names/order do not match the cache')
            self.models[number] = booster
            record['feature_count'] = len(names)
            self.records.append(record)
            self.input_paths.append(path)

    def predict(self, question, X):
        number = 0
        if self.mode == 'persona_disjoint_saved_models':
            persona = question['persona_id']
            if persona not in self.routes:
                raise ValueError(f'{persona}: no held-out model in manifest')
            number = self.routes[persona]
        model = self.models[number]
        scores = np.asarray(model.predict(X[:, :model.num_feature()], num_threads=4, device_type='cpu'), dtype=float)
        if scores.shape != (len(X),) or not np.isfinite(scores).all():
            raise ValueError('Invalid saved ranker predictions')
        return scores, number


class MessageEmbeddings:
    """Optional diversity vectors from the existing complete-message embedding cache."""
    def __init__(self, root):
        root = Path(root)
        metadata_path = root / 'memory_embedding_metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if (metadata.get('memory_unit') != 'single_message'
                or metadata.get('model', {}).get('repo') != _recipe.EMBEDDING_RECIPE['repo']
                or metadata.get('model', {}).get('revision') != _recipe.EMBEDDING_RECIPE['revision']
                or metadata.get('prefix') != _recipe.PREFIXES['memory']):
            raise ValueError('Diversity cache must use the pinned single-message EmbeddingGemma recipe')
        rows = read_rows(root / 'messages.jsonl')
        self.rows = {r['message_id']: r for r in rows}
        if len(self.rows) != len(rows):
            raise ValueError('Duplicate message IDs in diversity cache')
        self.vectors = np.load(root / 'memory_embeddings.npy', mmap_mode='r', allow_pickle=False)
        self.done = np.load(root / 'memory_done.npy', mmap_mode='r', allow_pickle=False)
        if (self.vectors.shape != (metadata.get('count'), 768) or self.vectors.dtype != np.float32
                or self.done.shape != (len(self.vectors),) or self.done.dtype != np.bool_):
            raise ValueError('Diversity cache dimensions/completion mask do not match metadata')
        self.provenance = {'root': str(root.resolve()), 'metadata_sha256': sha256(metadata_path),
                           'messages_sha256': sha256(root / 'messages.jsonl')}

    def for_messages(self, messages):
        indices = []
        for message in messages:
            cached = self.rows.get(message['message_id'])
            if cached is None or cached['text'] != message['text'] or cached['role'] != message['role']:
                raise ValueError(f"{message['message_id']}: missing or stale diversity embedding")
            i = cached.get('embedding_index')
            if type(i) is not int or i < 0 or i >= len(self.vectors) or not self.done[i]:
                raise ValueError('Incomplete diversity embedding cache')
            indices.append(i)
        vectors = self.vectors[indices]
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-4):
            raise ValueError('Diversity vectors must be finite and L2 normalized')
        return vectors


def diverse_pick(order, count, selected, similarity, weight):
    """Soft preference within the best 4*count available rows; no hard exclusion."""
    available = [i for i in order if i not in selected]
    if similarity is None or weight == 0:
        return available[:count]
    shortlist = available[:4 * count]
    quality = {i: 1 - rank / max(len(shortlist) - 1, 1) for rank, i in enumerate(shortlist)}
    picked = []
    while shortlist and len(picked) < count:
        compared = list(selected) + picked
        def utility(i):
            penalty = max(0., float(similarity[i, compared].max())) if compared else 0.
            return quality[i] - weight * penalty
        winner = max(shortlist, key=utility)  # stable source ranking resolves ties
        shortlist.remove(winner)
        picked.append(winner)
    return picked


def select_question(question, messages, reference_ids, ranker, weighted_bm25, dense,
                    *, seed=271, embeddings=None, diversity_weight=0):
    """Allocate exactly 12/3/3/2, scoring all messages before reference exclusion."""
    n = len(messages)
    ids = [m['message_id'] for m in messages]
    indices = [m['message_index'] for m in messages]
    if (len(set(ids)) != n or ids != question['memory_ids']
            or any(type(i) is not int or i < 0 for i in indices)
            or any(b <= a for a, b in zip(indices, indices[1:]))):
        raise ValueError('Messages must equal the full visible pool in source order')
    if not reference_ids or not set(reference_ids).issubset(ids):
        raise ValueError('References must be a nonempty subset of visible messages')
    eligible = {i for i, mid in enumerate(ids) if mid not in reference_ids}
    if len(eligible) < BUDGET:
        raise ValueError(f"{question['example_id']}: needs {BUDGET} non-reference messages; only {len(eligible)} available")
    scores = [np.asarray(values, dtype=float) for values in (ranker, weighted_bm25, dense)]
    if any(s.shape != (n,) or not np.isfinite(s).all() for s in scores):
        raise ValueError('Scores must cover every visible message with finite values')
    if not math.isfinite(diversity_weight) or not 0 <= diversity_weight <= 1:
        raise ValueError('diversity_weight must be between 0 and 1')
    similarity = None
    if diversity_weight:
        if embeddings is None or embeddings.shape != (n, 768) or not np.isfinite(embeddings).all():
            raise ValueError('Diversity requires one finite 768-dimensional vector per message')
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=2e-4):
            raise ValueError('Diversity vectors must be L2 normalized')
        similarity = np.clip(embeddings @ embeddings.T, -1., 1.)
    orders = [sorted(range(n), key=lambda i: (-float(s[i]), indices[i])) for s in scores]
    ranks = [{i: rank for rank, i in enumerate(order, 1)} for order in orders]
    selected, records = set(), []

    def add(rows, bucket):
        for i in rows:
            m = messages[i]
            row = {key: m[key] for key in ('message_id', 'message_index', 'role', 'text')}
            row.update(label=None, status='pending_qwen', allocation=bucket,
                       source=bucket, selection_order=len(records) + 1,
                       ranker_score=float(scores[0][i]), ranker_rank=ranks[0][i],
                       qweight_bm25=float(scores[1][i]), qweight_bm25_rank=ranks[1][i],
                       dense_chunk_max=float(scores[2][i]), dense_chunk_rank=ranks[2][i])
            records.append(row)
            selected.add(i)

    add([i for i in orders[0] if i in eligible][:ALLOCATION['ranker']], 'ranker')
    for bucket, order in (('qweight_bm25', orders[1]), ('dense_chunk', orders[2])):
        add(diverse_pick([i for i in order if i in eligible], ALLOCATION[bucket], selected, similarity,
                         diversity_weight), bucket)
    remaining = sorted(eligible - selected)
    add(random.Random(stable_seed(seed, question['example_id'])).sample(remaining, ALLOCATION['random']), 'random')
    if len(selected) != BUDGET or Counter(r['allocation'] for r in records) != ALLOCATION:
        raise AssertionError('Candidate allocation failed')
    return records


def run(features, *, model=None, fold_manifest=None, root=ROOT, split_root=None,
        output_root=None, limit_questions=100, seed=271, embedding_root=None, diversity_weight=0):
    started = time.perf_counter()
    timings = {}
    root = Path(root).resolve()
    if type(limit_questions) is not int or limit_questions < 0:
        raise ValueError('limit_questions must be nonnegative (0 means all cached questions)')
    if not math.isfinite(diversity_weight) or not 0 <= diversity_weight <= 1:
        raise ValueError('diversity_weight must be between 0 and 1')
    if diversity_weight and embedding_root is None:
        embedding_root = bundle_embedding_root(features)
        if not (embedding_root / 'memory_embedding_metadata.json').is_file():
            raise ValueError('Provide --embedding-root when enabling diversity, or build the feature bundle first')
    if model is not None and fold_manifest is not None:
        raise ValueError('Use either a saved model or a fold manifest')
    output = (Path(output_root) if output_root is not None else root / 'data/label_candidates/v2').resolve()
    if not output.is_relative_to((root / 'data/label_candidates').resolve()):
        raise ValueError('Outputs must stay under data/label_candidates/')
    stage = time.perf_counter()
    questions, memories, sources = load_message_dataset('v2', ['train'], root=root, split_root=split_root)
    timings['dataset_load'] = time.perf_counter() - stage
    stage = time.perf_counter()
    X, names, offsets, qids = load_features(features, questions, sources)
    timings['feature_load_and_validation'] = time.perf_counter() - stage
    by_id = {q['example_id']: q for q in questions}
    chosen = sorted(qids, key=lambda qid: (stable_seed(seed, qid), qid))
    if limit_questions:
        chosen = chosen[:limit_questions]
    chosen = sorted(chosen)
    stage = time.perf_counter()
    references = load_references([by_id[qid] for qid in chosen], sources, root)
    timings['reference_load'] = time.perf_counter() - stage
    stage = time.perf_counter()
    models = SavedModels(model or root / 'models/subtlex_mix/alpha_0.5/ranker55.txt', fold_manifest, names,
                         feature_recipe=bundle_recipe(features))
    timings['model_load'] = time.perf_counter() - stage
    stage = time.perf_counter()
    embeddings = MessageEmbeddings(embedding_root) if diversity_weight else None
    timings['embedding_cache_load'] = time.perf_counter() - stage
    timings['prediction'] = timings['selection'] = 0.
    inputs = [Path(features).resolve(), *models.input_paths]
    if any(path == output or path.is_relative_to(output) for path in inputs):
        raise ValueError('Input features/models must be outside the candidate output directory')
    group_index = {qid: i for i, qid in enumerate(qids)}
    lexical_column, dense_column = names.index('qweight_bm25'), names.index('dense_chunk_max')
    output.mkdir(parents=True, exist_ok=True)
    path, report_path = output / 'train.jsonl', output / 'report.json'
    temporary, report_temporary = path.with_suffix('.jsonl.tmp'), report_path.with_suffix('.json.tmp')
    counts, source_counts = Counter(), Counter()
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            for number, qid in enumerate(chosen, 1):
                q = by_id[qid]
                members = [memories[mid] for mid in q['memory_ids']]
                labels = label_question(q, memories, references[qid])
                ref_ids = {m['message_id'] for m in labels['memories'] if m['label'] == REFERENCE_LABEL}
                group = group_index[qid]
                values = X[int(offsets[group]):int(offsets[group + 1])]
                stage = time.perf_counter()
                scores, model_index = models.predict(q, values)
                timings['prediction'] += time.perf_counter() - stage
                stage = time.perf_counter()
                candidates = select_question(q, members, ref_ids, scores,
                    values[:, lexical_column], values[:, dense_column], seed=seed,
                    embeddings=embeddings.for_messages(members) if embeddings is not None else None,
                    diversity_weight=diversity_weight)
                timings['selection'] += time.perf_counter() - stage
                selected_ids = {m['message_id'] for m in candidates}
                reference_rows = [{**{key: m[key] for key in ('message_id', 'message_index', 'role', 'text')},
                                   'label': REFERENCE_LABEL, 'status': 'reference'}
                                  for m in members if m['message_id'] in ref_ids]
                unjudged = [mid for mid in q['memory_ids'] if mid not in ref_ids | selected_ids]
                row = {key: q[key] for key in ('example_id', 'dataset', 'split', 'persona_id', 'history_id', 'question')}
                row.update(schema_version=1, memory_count=len(members), new_label_budget=BUDGET,
                           reference_count=len(reference_rows), candidate_count=len(candidates),
                           model_index=model_index, selection_provenance=models.mode,
                           references=reference_rows, candidates=candidates,
                           unselected_label=None, unjudged_message_ids=unjudged)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                counts.update(questions=1, visible_messages=len(members), references=len(ref_ids),
                              candidates=len(candidates), unselected_unjudged=len(unjudged))
                source_counts.update(m['source'] for m in candidates)
                if number % 100 == 0 or number == len(chosen):
                    print(f'{number:,}/{len(chosen):,} questions; {counts["candidates"]:,} candidates selected', flush=True)
        report = {'schema_version': 1, 'dataset': 'v2', 'split': 'train', 'memory_unit': 'single_message',
                  'training': False, 'qwen_called': False, 'inference_pool': 'complete visible history',
                  'new_candidates_per_question': BUDGET, 'allocation': ALLOCATION, 'counts': dict(counts),
                  'actual_source_counts': dict(source_counts), 'seed': seed, 'limit_questions': limit_questions,
                  'cached_questions': len(qids), 'active_training_questions': len(questions),
                  'feature_names': names, 'features_path': str(Path(features).resolve()),
                  'features_sha256': sha256(Path(features)), 'selection_provenance': models.mode,
                  'models': models.records, 'source_sha256': sources,
                  'selector_sha256': sha256(Path(__file__)),
                  'diversity_weight': diversity_weight,
                  'diversity_cache': embeddings.provenance if embeddings is not None else None,
                  'labels': {'reference': 2, 'pending_qwen': None, 'unselected': None},
                  'ties': 'descending score, then ascending original message index',
                  'files_sha256': {'train.jsonl': sha256(temporary)}}
        timings['total_before_report_publish'] = time.perf_counter() - started
        report['runtime_seconds'] = timings
        report_temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)
        report_temporary.replace(report_path)
    finally:
        temporary.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', type=Path, required=True, help='Complete-pool NPZ with aligned question/message IDs')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--model', type=Path, help='Saved LightGBM text model (default: models/subtlex_mix/alpha_0.5/ranker55.txt)')
    source.add_argument('--fold-manifest', type=Path, help='Saved persona-held-out models with hashes and training/held persona IDs')
    parser.add_argument('--split-root', type=Path, default=SPLIT_ROOT)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data/label_candidates/v2')
    parser.add_argument('--limit-questions', type=int, default=0, help='Pilot size; 0 selects all cached training questions')
    parser.add_argument('--seed', type=int, default=271)
    parser.add_argument('--embedding-root', type=Path, help='Diversity cache; defaults to embeddings/ beside the feature NPZ')
    parser.add_argument('--diversity-weight', type=float, default=0, help='Soft cosine penalty in [0,1]; try 0.1 using the bundle embeddings')
    args = parser.parse_args(argv)
    try:
        report = run(**vars(args))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({'output': str(args.output_root.resolve()), **report['counts'],
                      'selection_provenance': report['selection_provenance']}, indent=2))


if __name__ == '__main__':
    main()
