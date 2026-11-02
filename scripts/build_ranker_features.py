"""Build the complete 55-feature training/labeling bundle from original messages.

python scripts/build_ranker_features.py
python scripts/build_ranker_features.py --splits val test

Default: active V2 training questions, SUBTLEX mixture alpha=0.5,
data/ranker55/v2/train/features.npz.
Embeddings resume from checked caches. Completed exports overwrite existing files.
This builds features and reference targets; it never trains LightGBM or calls Qwen.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np

from label_reference_messages import label_question, load_references
from message_dataset import SPLIT_ROOT, load_message_dataset, write_json, write_rows
from raw_datasets import ROOT, SPLITS, sha256

import ranker55 as recipe
import subtlex_mixture as subtlex
FEATURE_NAMES = recipe.FEATURE_NAMES
DIMENSIONS, MAX_TOKENS = 768, 1200
PREFIXES = recipe.PREFIXES


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verified_model(root=ROOT):
    root = Path(root)
    model_path = root / 'models/embeddinggemma-2'
    info = json.loads((root / 'configs/assets.lock.json').read_text(encoding='utf-8'))['embedding']
    if any(info.get(key) != recipe.EMBEDDING_RECIPE[key] for key in ('repo', 'revision')):
        raise ValueError('The ranker requires the pinned EmbeddingGemma 2 revision')
    if not info.get('files'):
        raise ValueError('Embedding model lock has no verified files')
    for entry in info['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(model_path.resolve()):
            raise ValueError('Embedding model asset escapes its folder')
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Missing or changed EmbeddingGemma 2 asset: {path}')
    return model_path, info


class Encoder:
    def __init__(self, path, device='auto'):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.torch = torch
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable')
        self.dtype = torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        config.vision_config = config.audio_config = None
        self.model = AutoModel.from_pretrained(path, config=config, local_files_only=True,
                                              trust_remote_code=False, dtype=self.dtype,
                                              attn_implementation='sdpa').to(self.device).eval()
        self.model.requires_grad_(False)

    def lengths(self, texts, kind):
        """Content lengths exclude both the prefix and special tokens."""
        lengths = []
        for start in range(0, len(texts), 256):
            batch = texts[start:start + 256]
            formatted = batch if kind == 'content' else [PREFIXES[kind] + text for text in batch]
            rows = self.tokenizer(formatted, add_special_tokens=kind != 'content',
                                  padding=False, truncation=False)['input_ids']
            lengths.extend(len(row) for row in rows)
        return np.asarray(lengths, dtype=np.int32)

    def encode(self, texts, kind):
        torch = self.torch
        formatted = [PREFIXES[kind] + text for text in texts]
        batch = self.tokenizer(formatted, add_special_tokens=True, padding=True,
                               truncation=False, return_tensors='pt').to(self.device)
        if batch['input_ids'].shape[1] > MAX_TOKENS:
            raise ValueError(f'Input exceeds {MAX_TOKENS} tokens; no truncation is allowed')
        with torch.inference_mode():
            states = self.model(**batch).last_hidden_state.float()
            mask = batch['attention_mask'].unsqueeze(-1)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
        validate_vectors(vectors, len(texts))
        return vectors


def validate_vectors(vectors, count):
    if (vectors.shape != (count, DIMENSIONS) or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=2e-4)):
        raise ValueError('Embedding vectors must be finite normalized float32 rows of dimension 768')


def close_array(array):
    mmap = getattr(array, '_mmap', None)
    if mmap is not None:
        mmap.close()


def save_array(path, array):
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cached_embeddings(output, kind, texts, ids, encoder, embedding_recipe,
                      batch_size=16, batch_tokens=4800, rebuild=False, *, lengths=None):
    """Chunk cache uses the memory prefix. Completion is committed after vectors."""
    encode_kind = 'question' if kind == 'question' else 'memory'
    if lengths is None:
        lengths = encoder.lengths(texts, encode_kind)
    lengths = np.asarray(lengths)
    if lengths.shape != (len(texts),) or lengths.dtype.kind not in 'iu' or np.any(lengths < 0):
        raise ValueError('Token lengths must be aligned nonnegative integers')
    too_long = np.flatnonzero(lengths > MAX_TOKENS)
    if len(too_long):
        raise ValueError(f'Complete inputs exceed {MAX_TOKENS} tokens: ' +
                         repr([(ids[int(i)], int(lengths[i])) for i in too_long[:10]]) +
                         '. No splitting or truncation was performed.')
    expected = {**embedding_recipe, 'max_input_tokens': MAX_TOKENS, 'kind': kind,
                'prefix': PREFIXES[encode_kind], 'count': len(texts),
                'inputs_sha256': fingerprint(list(zip(ids, texts)))}
    path, done_path = output / f'{kind}_embeddings.npy', output / f'{kind}_done.npy'
    metadata_path = output / f'{kind}_embedding_metadata.json'
    exists = [p.exists() for p in (path, done_path, metadata_path)]
    vectors = done = None
    try:
        if any(exists) and not rebuild:
            if not all(exists) or json.loads(metadata_path.read_text(encoding='utf-8')) != expected:
                raise ValueError(f'Stale/incomplete {kind} cache; use --rebuild-cache to overwrite it')
            vectors = np.load(path, mmap_mode='r+', allow_pickle=False)
            done = np.load(done_path, mmap_mode='r+', allow_pickle=False)
            if (vectors.shape != (len(texts), DIMENSIONS) or vectors.dtype != np.float32
                    or done.shape != (len(texts),) or done.dtype != np.bool_):
                raise ValueError('Embedding cache arrays differ from their inputs')
            for start in range(0, len(done), 4096):
                finished = vectors[start:start + 4096][done[start:start + 4096]]
                validate_vectors(finished, len(finished))
        else:
            output.mkdir(parents=True, exist_ok=True)
            # Remove the completion manifest first: an interrupted rebuild cannot
            # make partially replaced arrays look like a valid old cache.
            metadata_path.unlink(missing_ok=True)
            vectors = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(texts), DIMENSIONS))
            done = np.lib.format.open_memmap(done_path, mode='w+', dtype=np.bool_, shape=(len(texts),))
            done[:] = False
            done.flush()
            write_json(metadata_path, expected)
        order = [int(i) for i in np.argsort(lengths, kind='stable') if not done[i]]
        position, started, reported = 0, time.perf_counter(), time.monotonic()
        print(f'{kind}: {int(done.sum()):,}/{len(texts):,} embedded', flush=True)
        while position < len(order):
            size = min(batch_size, len(order) - position)
            while size > 1 and size * int(lengths[order[position + size - 1]]) > batch_tokens:
                size -= 1
            rows = order[position:position + size]
            try:
                result = encoder.encode([texts[i] for i in rows], encode_kind)
            except Exception as error:
                oom = getattr(getattr(encoder, 'torch', None), 'OutOfMemoryError', None)
                if oom is None or not isinstance(error, oom) or size == 1:
                    raise
                encoder.torch.cuda.empty_cache()
                batch_size = max(1, size // 2)
                print(f'Reducing embedding batch size to {batch_size}', flush=True)
                continue
            validate_vectors(result, len(rows))
            vectors[rows] = result
            vectors.flush()
            done[rows] = True
            done.flush()
            position += size
            if time.monotonic() - reported >= 20 or position == len(order):
                print(f'{kind}: {int(done.sum()):,}/{len(texts):,}; '
                      f'{position / max(time.perf_counter() - started, .001):.1f} inputs/s', flush=True)
                reported = time.monotonic()
        save_array(output / f'{kind}_input_token_lengths.npy', lengths.astype(np.int32))
        result = vectors
        vectors = None  # caller owns this map
        return result, lengths
    finally:
        close_array(done)
        close_array(vectors)


def cosine_scores(memory_vectors, question_vector):
    return np.clip(memory_vectors @ question_vector, -1.0, 1.0)


def select_ids(questions, ids_path):
    if ids_path is None:
        return questions
    ids = json.loads(Path(ids_path).read_text(encoding='utf-8'))
    if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Question IDs must be a nonempty JSON list of unique strings')
    unknown = set(ids) - {q['example_id'] for q in questions}
    if unknown:
        raise ValueError('Question IDs are outside the requested active split: ' + repr(sorted(unknown)[:5]))
    wanted = set(ids)
    return [q for q in questions if q['example_id'] in wanted]


def prepare_inputs(*, root=ROOT, datasets=('v2',), splits=None, split_root=None,
                   question_ids=None, limit_questions=0):
    """Validate visible source pools and training references before writing."""
    root = Path(root).resolve()
    loaded = []
    for dataset in dict.fromkeys(datasets):
        selected_splits = list(dict.fromkeys(splits or (['train'] if dataset == 'v2' else ['comparison'])))
        print(f'{dataset}: loading original sources for {selected_splits}', flush=True)
        questions, memories, sources = load_message_dataset(dataset, selected_splits, root=root, split_root=split_root)
        print(f'{dataset}: {len(questions):,} questions, {len(memories):,} visible source messages', flush=True)
        loaded.append((dataset, selected_splits, questions, memories, sources))
    tasks = []
    exported = [q for _, _, qs, _, _ in loaded for q in qs]
    wanted = {q['example_id'] for q in select_ids(exported, question_ids)}
    for dataset, selected_splits, questions, memories, sources in loaded:
        for split in selected_splits:
            selected = [q for q in questions if q['split'] == split and q['example_id'] in wanted]
            if limit_questions:
                selected = selected[:limit_questions]
            if not selected:
                continue
            labels = None
            if dataset == 'v2' and split == 'train':
                print(f'{dataset}/{split}: checking original reference labels', flush=True)
                references = load_references(selected, sources, root)
                labels = [label_question(q, memories, references[q['example_id']]) for q in selected]
            # Source cutoffs have already been applied by load_message_dataset,
            # including Python's negative-index semantics for V1.
            for q in selected:
                recipe._validate_messages([memories[mid] for mid in q['memory_ids']])
            tasks.append((dataset, split, selected, memories, sources, labels))
    if not tasks:
        raise ValueError('No questions selected for feature export')
    return tasks


def _init_feature_worker(state, embedding_root):
    global _feature_worker_state
    vectors = {kind: np.load(Path(embedding_root) / f'{kind}_embeddings.npy',
                             mmap_mode='r', allow_pickle=False)
               for kind in ('memory', 'question', 'chunk')}
    _feature_worker_state = (*state, vectors)


def _calculate_feature_group(question, source_memories, content_lengths, text_index,
                             question_index, chunk_index, mixture, vectors):
    def embed(values, kind):
        if kind == 'question':
            return vectors['question'][[question_index[t] for t in values]]
        result = np.empty((len(values), DIMENSIONS), dtype=np.float32)
        for i, text in enumerate(values):
            source, index = ('memory', text_index[text]) if text in text_index else ('chunk', chunk_index[text])
            result[i] = vectors[source][index]
        return result

    messages = [source_memories[mid] for mid in question['memory_ids']]
    lengths = [int(content_lengths[text_index[m['text']]]) for m in messages]
    lexical = (mixture.history(question, messages), mixture.config['alpha'])
    return recipe.feature_matrix(question['question'], messages, lengths, lexical, embed)


def _worker_feature_group(question):
    return _calculate_feature_group(question, *_feature_worker_state)


def feature_groups(questions, state, vectors, embedding_root, workers):
    """Bound pending matrices and yield in source order; only the parent writes."""
    if workers == 1:
        for question in questions:
            yield _calculate_feature_group(question, *state, vectors)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_feature_worker,
                             initargs=(state, embedding_root)) as pool:
        iterator = iter(questions)
        pending = deque()
        for _ in range(workers * 2):
            question = next(iterator, None)
            if question is None:
                break
            pending.append(pool.submit(_worker_feature_group, question))
        while pending:
            yield pending.popleft().result()
            question = next(iterator, None)
            if question is not None:
                pending.append(pool.submit(_worker_feature_group, question))


def readonly_cache_metadata(embedding_root, inputs, model_info, sources):
    """Require a completed, matching export; never repair or mutate its caches."""
    embedding_root = Path(embedding_root)
    report = json.loads((embedding_root.parent / 'report.json').read_text(encoding='utf-8'))
    if report.get('source_sha256') != sources:
        raise ValueError('Embedding cache source hashes differ from active raw data/splits')
    common = None
    lengths = {}
    for kind, values in inputs.items():
        metadata = json.loads((embedding_root / f'{kind}_embedding_metadata.json').read_text(encoding='utf-8'))
        fields = ('model', 'dimensions', 'memory_unit', 'pooling', 'dtype', 'encoder_schema')
        identity = {key: metadata[key] for key in fields}
        if common is None:
            common = identity
        expected = {**common, 'max_input_tokens': MAX_TOKENS, 'kind': kind,
                    'prefix': PREFIXES['question' if kind == 'question' else 'memory'],
                    'count': len(values), 'inputs_sha256': fingerprint(list(enumerate(values)))}
        required_identity = {'model': {k: model_info[k] for k in ('repo', 'revision')},
                             'dimensions': DIMENSIONS, 'memory_unit': 'single_message',
                             'pooling': 'attention_mask_mean_including_prompt_then_L2', 'encoder_schema': 1}
        if metadata != expected or any(common[k] != v for k, v in required_identity.items()):
            raise ValueError(f'Incompatible or stale {kind} embedding cache identity/inputs')
        done = np.load(embedding_root / f'{kind}_done.npy', allow_pickle=False)
        if done.dtype != np.bool_ or done.shape != (len(values),) or not done.all():
            raise ValueError(f'Incomplete {kind} embedding cache; finish the baseline build first')
        lengths[kind] = np.load(embedding_root / f'{kind}_input_token_lengths.npy', allow_pickle=False)
    content = np.load(embedding_root / 'memory_content_token_lengths.npy', allow_pickle=False)
    return common, lengths, content


def build_bundle(task, statistics, encoder, model_info, output, *, batch_size=16,
                 batch_tokens=4800, rebuild_cache=False, feature_workers=1,
                 mixture, embedding_source=None):
    dataset, split, questions, source_memories, sources, labels = task
    output = Path(output)
    embedding_root = Path(embedding_source) if embedding_source is not None else output / 'embeddings'
    mids = list(dict.fromkeys(mid for q in questions for mid in q['memory_ids']))
    members = [source_memories[mid] for mid in mids]
    texts = list(dict.fromkeys(m['text'] for m in members))
    text_index = {text: i for i, text in enumerate(texts)}
    spans = [recipe.dense_spans(text) for text in texts]
    # Short messages use their original whole-message cosine, as in ranker55.
    chunk_texts = list(dict.fromkeys(piece for pieces in spans if len(pieces) > 1 for piece in pieces))
    chunk_index = {text: i for i, text in enumerate(chunk_texts)}
    question_texts = list(dict.fromkeys(q['question'] for q in questions))
    question_index = {text: i for i, text in enumerate(question_texts)}
    inputs = {'memory': texts, 'question': question_texts, 'chunk': chunk_texts}
    print(f'{dataset}/{split}: {len(texts):,} unique messages, {len(question_texts):,} queries, '
          f'{len(chunk_texts):,} chunks', flush=True)
    if embedding_source is not None:
        embedding_recipe, lengths, content_lengths = readonly_cache_metadata(embedding_root, inputs, model_info, sources)
    else:
        lengths = {kind: encoder.lengths(values, 'question' if kind == 'question' else 'memory')
                   for kind, values in inputs.items()}
        content_lengths = encoder.lengths(texts, 'content')
        embedding_recipe = {'model': {k: model_info[k] for k in ('repo', 'revision')},
                            'dimensions': DIMENSIONS, 'memory_unit': 'single_message',
                            'pooling': 'attention_mask_mean_including_prompt_then_L2',
                            'dtype': str(encoder.dtype), 'encoder_schema': 1}
    if content_lengths.shape != (len(texts),) or content_lengths.dtype.kind not in 'iu' or np.any(content_lengths < 0):
        raise ValueError('Invalid bare content token lengths')
    for kind, values in lengths.items():
        if values.shape != (len(inputs[kind]),) or values.dtype.kind not in 'iu' or np.any(values < 0):
            raise ValueError('Invalid input token lengths')
        if np.any(values > MAX_TOKENS):
            raise ValueError(f'{kind} inputs exceed {MAX_TOKENS} tokens. No splitting or truncation was performed.')
    output.mkdir(parents=True, exist_ok=True)
    if embedding_source is None:
        embedding_root.mkdir(exist_ok=True)
    vectors = {}
    staging = {}
    X = None

    def target(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        staging[path] = temporary
        return temporary

    try:
        for kind, values in inputs.items():
            if embedding_source is None:
                vectors[kind], _ = cached_embeddings(embedding_root, kind, values, list(range(len(values))),
                    encoder, embedding_recipe, batch_size, batch_tokens, rebuild_cache, lengths=lengths[kind])
            else:
                vectors[kind] = np.load(embedding_root / f'{kind}_embeddings.npy', mmap_mode='r', allow_pickle=False)
                for start in range(0, max(len(values), 1), 4096):
                    validate_vectors(vectors[kind][start:start + 4096], min(4096, max(len(values) - start, 0)))
                if vectors[kind].shape != (len(values), DIMENSIONS):
                    raise ValueError(f'Invalid {kind} embedding cache shape')
        if embedding_source is None:
            save_array(embedding_root / 'memory_content_token_lengths.npy', content_lengths.astype(np.int32))
            write_rows(embedding_root / 'messages.jsonl', ({**m, 'embedding_index': text_index[m['text']],
                       'content_tokens': int(content_lengths[text_index[m['text']]])} for m in members))
            write_rows(embedding_root / 'unique_texts.jsonl',
                       ({'embedding_index': i, 'text': t} for i, t in enumerate(texts)))
            write_rows(embedding_root / 'chunk_texts.jsonl',
                       ({'embedding_index': i, 'text': t} for i, t in enumerate(chunk_texts)))
            write_rows(embedding_root / 'questions.jsonl',
                       ({**q, 'embedding_index': question_index[q['question']]} for q in questions))
            chunk_offsets, chunk_indices = [0], []
            for pieces in spans:
                chunk_indices.extend(chunk_index[p] for p in pieces if len(pieces) > 1)
                chunk_offsets.append(len(chunk_indices))
            save_array(embedding_root / 'message_chunk_offsets.npy', np.array(chunk_offsets, dtype=np.int64))
            save_array(embedding_root / 'message_chunk_indices.npy', np.array(chunk_indices, dtype=np.int64))

        offsets = np.concatenate(([0], np.cumsum([len(q['memory_ids']) for q in questions]))).astype(np.int64)
        row_ids = [mid for q in questions for mid in q['memory_ids']]
        X_path = output / '.features_work.npy'
        X = np.lib.format.open_memmap(X_path, mode='w+', dtype=np.float32, shape=(len(row_ids), 55))
        reference_labels = np.full(len(row_ids), -1, dtype=np.int8)
        y = np.full(len(row_ids), -1, dtype=np.int8)
        with ExitStack() as stack:
            score_streams = {name: stack.enter_context(target(output / name / 'scores' / f'{split}.jsonl').open(
                             'w', encoding='utf-8', newline='\n')) for name in ('bm25', 'cosine')}
            label_stream = (stack.enter_context(target(output / 'reference_labels' / f'{split}.jsonl').open(
                            'w', encoding='utf-8', newline='\n')) if labels is not None else None)
            state = (source_memories, content_lengths, text_index, question_index, chunk_index, mixture)
            print(f'{dataset}/{split}: calculating feature groups with {feature_workers} worker(s)', flush=True)
            matrices = feature_groups(questions, state, vectors, embedding_root, feature_workers)
            stack.callback(matrices.close)
            for i, (q, matrix) in enumerate(zip(questions, matrices)):
                messages = [source_memories[mid] for mid in q['memory_ids']]
                start, end = int(offsets[i]), int(offsets[i + 1])
                X[start:end] = matrix
                if labels is not None:
                    reference_labels[start:end] = [m['label'] for m in labels[i]['memories']]
                    y[start:end] = reference_labels[start:end] == 2
                    label_stream.write(json.dumps(labels[i], ensure_ascii=False, allow_nan=False) + '\n')
                for name, column, score_key in (('bm25', 'bm25_b0', 'bm25_score'),
                                                ('cosine', 'cosine', 'embeddinggemma2_cosine')):
                    scores = matrix[:, FEATURE_NAMES.index(column)]
                    order = np.argsort(-scores, kind='stable')
                    rows = [{**{key: messages[int(j)][key] for key in ('message_id', 'message_index', 'role')},
                             'rank': rank, score_key: float(scores[j])} for rank, j in enumerate(order, 1)]
                    score_row = {key: q[key] for key in ('example_id', 'dataset', 'split', 'history_id')}
                    score_row.update(memory_count=len(messages), memories=rows)
                    score_streams[name].write(json.dumps(score_row, ensure_ascii=False, allow_nan=False) + '\n')
                if (i + 1) % 100 == 0 or i + 1 == len(questions):
                    print(f'{dataset}/{split}: {i + 1:,}/{len(questions):,} feature groups', flush=True)
        X.flush()
        stats_fingerprint = fingerprint(statistics)
        metadata = {'schema_version': 1, 'dataset': dataset, 'split': split, 'memory_unit': 'single_message',
                    'source_sha256': sources, 'statistics_fingerprint': stats_fingerprint,
                    'feature_recipe': mixture.config,
                    'embedding_cache_path': str(embedding_root.resolve()),
                    'feature_recipe_sha256': sha256(Path(recipe.__file__)),
                    'embedding_recipe': embedding_recipe, 'row_count': len(X), 'questions': len(questions),
                    'y_policy': 'binary exact-reference membership for baseline training; not semantic judgments',
                    'judged_labels_policy': '2 for original references; -1 for unjudged; no Qwen judgments yet'}
        # Preserve scalar metadata and row identities inside the artifact so it
        # stays self-describing when copied away from its sidecar report.
        with target(output / 'features.npz').open('wb') as stream:
            np.savez(stream, X=X, y=y, reference_labels=reference_labels,
                     judged_labels=np.where(reference_labels == 2, 2, -1).astype(np.int8),
                     offsets=offsets, feature_names=np.asarray(FEATURE_NAMES),
                     question_ids=np.asarray([q['example_id'] for q in questions]),
                     persona_ids=np.asarray([q['persona_id'] for q in questions]), message_ids=np.asarray(row_ids),
                     message_indices=np.asarray([source_memories[mid]['message_index'] for mid in row_ids], dtype=np.int64),
                     content_token_lengths=np.asarray([content_lengths[text_index[source_memories[mid]['text']]]
                                                       for mid in row_ids], dtype=np.int32),
                     metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False)))
        target(output / 'statistics.json').write_text(json.dumps(statistics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for name, count_key in (('bm25', 'saved_bm25_scores'), ('cosine', 'saved_cosine_scores'),
                                 ('reference_labels', 'positive_memories')):
            if name == 'reference_labels' and labels is None:
                continue
            parent = {'memory_unit': 'single_message', 'source_sha256': sources,
                      'splits': {split: {'questions': len(questions), count_key: int((y == 1).sum()) if name == 'reference_labels' else len(X)}},
                      'score_definition': 'bm25_b0 (k1=2, b=0, visible-pool IDF)' if name == 'bm25' else name,
                      'files_sha256': {f'{split}.jsonl': sha256(staging[output / name /
                                      (f'{split}.jsonl' if name == 'reference_labels' else f'scores/{split}.jsonl')])}}
            target(output / name / 'report.json').write_text(json.dumps(parent, indent=2) + '\n', encoding='utf-8')
        report = {**metadata, 'feature_names': FEATURE_NAMES, 'embedding_root': str(embedding_root.resolve()),
                  'unique_message_texts': len(texts), 'unique_chunk_texts': len(chunk_texts),
                  'unique_question_texts': len(question_texts), 'reference_count': int((y == 1).sum()),
                  'training': False, 'qwen_called': False, 'builder_sha256': sha256(Path(__file__)),
                  'files_sha256': {p.relative_to(output).as_posix(): sha256(t) for p, t in staging.items()}}
        target(output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for path, temporary in staging.items():
            temporary.replace(path)
        return report
    finally:
        close_array(X)
        for value in vectors.values():
            close_array(value)
        (output / '.features_work.npy').unlink(missing_ok=True)
        for temporary in staging.values():
            temporary.unlink(missing_ok=True)


def run(*, root=ROOT, datasets=('v2',), splits=None, split_root=None, output_root=None,
        question_ids=None, limit_questions=0, device='auto', batch_size=16, batch_tokens=4800,
        rebuild_cache=False, encoder=None, model_info=None, feature_workers=1,
        alpha=.5, subtlex_prior=None, subtlex_manifest=None, embedding_cache_root=None):
    root = Path(root).resolve()
    counts, total, lock = subtlex.load_prior(
        subtlex_prior or root / 'data/external/subtlex_us/word_frequencies.json',
        subtlex_manifest or root / 'configs/subtlex_us.lock.json')
    config = subtlex.configuration(alpha, lock)
    mixture = subtlex.Mixture(counts, total, config)
    statistics = {'schema_version': 2, 'feature_recipe': config}
    output = (Path(output_root) if output_root is not None else root / 'data/ranker55').resolve()
    split_root = Path(split_root).resolve() if split_root is not None else root / 'data/splits'
    if (not output.is_relative_to((root / 'data/ranker55').resolve())
            or output.is_relative_to(split_root) or split_root.is_relative_to(output)):
        raise ValueError('Outputs must stay under data/ranker55/, separate from split ID inputs')
    if embedding_cache_root is not None:
        embedding_cache_root = Path(embedding_cache_root).resolve()
        if embedding_cache_root != output and embedding_cache_root.is_relative_to(output):
            raise ValueError('Read-only embedding cache root cannot be nested inside the output root')
        if rebuild_cache:
            raise ValueError('--rebuild-cache cannot be used with read-only embedding caches')
    if (type(limit_questions) is not int or limit_questions < 0
            or any(type(v) is not int or v < 1 for v in (batch_size, batch_tokens, feature_workers))):
        raise ValueError('Batch sizes must be positive; limit_questions must be nonnegative')
    tasks = prepare_inputs(root=root, datasets=datasets, splits=splits, split_root=split_root,
                           question_ids=question_ids, limit_questions=limit_questions)
    if encoder is None and embedding_cache_root is None:
        path, model_info = verified_model(root)
        encoder = Encoder(path, device)
    elif model_info is None and embedding_cache_root is not None:
        model_info = recipe.EMBEDDING_RECIPE
    if model_info is None or any(model_info.get(k) != recipe.EMBEDDING_RECIPE[k] for k in ('repo', 'revision')):
        raise ValueError('Encoder identity must match the ranker embedding recipe')
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for task in tasks:
        dataset, split = task[:2]
        destination = output / dataset / split
        source = None if embedding_cache_root is None else embedding_cache_root / dataset / split / 'embeddings'
        if source is not None and ((source.is_relative_to(destination) and source != destination / 'embeddings')
                                   or destination.is_relative_to(source)):
            raise ValueError('Read-only embedding caches must be separate from output files')
        reports[f'{dataset}/{split}'] = build_bundle(task, statistics, encoder, model_info, destination,
            batch_size=batch_size, batch_tokens=batch_tokens, rebuild_cache=rebuild_cache,
            feature_workers=feature_workers, mixture=mixture, embedding_source=source)
    write_json(output / 'statistics.json', statistics)
    return reports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', choices=tuple(SPLITS), default=['v2'])
    parser.add_argument('--splits', nargs='+', choices=('train', 'val', 'test', 'comparison'))
    parser.add_argument('--alpha', type=float, default=.5, help='SUBTLEX mixture weight in [0,1] (default: 0.5)')
    parser.add_argument('--subtlex-prior', type=Path, help='Processed word_frequencies.json')
    parser.add_argument('--subtlex-manifest', type=Path, help='Pinned source/hash manifest')
    parser.add_argument('--embedding-cache-root', type=Path,
                        help='Read-only bundle root containing {dataset}/{split}/embeddings; must match complete inputs')
    parser.add_argument('--question-ids', type=Path, help='Optional JSON list selecting output questions; does not change fitting corpus')
    parser.add_argument('--limit-questions', type=int, default=0, help='Limit exported questions per split; 0 means all; fitting corpus is unchanged')
    parser.add_argument('--split-root', type=Path, default=SPLIT_ROOT)
    parser.add_argument('--output-root', type=Path,
                        help='Defaults to data/ranker55')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--batch-tokens', type=int, default=4800)
    parser.add_argument('--feature-workers', type=int, default=4,
                        help='Concurrent CPU feature processes; 2 is a conservative starting point. Embedding caches are shared read-only.')
    parser.add_argument('--rebuild-cache', action='store_true', help='Overwrite stale/interrupted embedding arrays; otherwise resume matching caches')
    args = parser.parse_args(argv)
    for dataset in args.datasets:
        if args.splits and any(split not in SPLITS[dataset] for split in args.splits):
            parser.error(f'{dataset} supports only {SPLITS[dataset]}')
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        reports = run(**vars(args))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({key: {'questions': r['questions'], 'rows': r['row_count'], 'features': 55}
                      for key, r in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
