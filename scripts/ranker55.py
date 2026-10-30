"""Training source for the selected 55-feature LightGBM LambdaRank architecture.

28 base features + 20 neighbor features + 6 dense-chunk features + relative
conversation position. The file contains no fitted model, corpus statistics,
training samples, saved predictions, or encoded data.

Train from an external NPZ with X, y, offsets and feature_names:
  python -B scripts/ranker55.py
Defaults: data/ranker55/v2/train/features.npz -> models/ranker55.txt.
Existing model output files are replaced atomically after successful training.
Inspect the recipe without fitting:
  python -B scripts/ranker55.py --describe

feature_matrix() constructs inputs when given externally fitted statistics,
source messages, token lengths, and a pinned embedding callback. Arrays must be
computed over the complete question-visible candidate pool before training-row
filtering. NumPy and LightGBM are the training dependencies.
"""
import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np

FEATURE_NAMES = ['cosine',
 'cosine_rank_fraction',
 'cosine_zscore',
 'cosine_gap',
 'role_assistant',
 'log_lexical_length',
 'log_content_tokens',
 'bm25_b0',
 'bm25_rank_fraction',
 'bm25_zscore',
 'bm25_gap',
 'winning_fusion',
 'chunk_max',
 'chunk_rank_fraction',
 'chunk_zscore',
 'chunk_gap',
 'chunk_top2mean',
 'log_chunk_count',
 'qweight_bm25',
 'qweight_bm25_rank_fraction',
 'qweight_bm25_zscore',
 'qweight_bm25_gap',
 'qweight_fusion',
 'qweight_chunk_max',
 'qweight_chunk_rank_fraction',
 'qweight_chunk_zscore',
 'qweight_chunk_gap',
 'qweight_chunk_top2mean',
 'neighbor_minus2_cosine',
 'neighbor_minus2_cosine_rank',
 'neighbor_minus2_weighted_bm25',
 'neighbor_minus2_weighted_bm25_rank',
 'neighbor_minus2_available',
 'neighbor_minus1_cosine',
 'neighbor_minus1_cosine_rank',
 'neighbor_minus1_weighted_bm25',
 'neighbor_minus1_weighted_bm25_rank',
 'neighbor_minus1_available',
 'neighbor_plus1_cosine',
 'neighbor_plus1_cosine_rank',
 'neighbor_plus1_weighted_bm25',
 'neighbor_plus1_weighted_bm25_rank',
 'neighbor_plus1_available',
 'neighbor_plus2_cosine',
 'neighbor_plus2_cosine_rank',
 'neighbor_plus2_weighted_bm25',
 'neighbor_plus2_weighted_bm25_rank',
 'neighbor_plus2_available',
 'dense_chunk_max',
 'dense_chunk_rank',
 'dense_chunk_zscore',
 'dense_chunk_gap',
 'dense_chunk_top2mean',
 'dense_chunk_minus_whole',
 'relative_conversation_position']


PARAMETERS = {'objective': 'lambdarank',
 'device_type': 'cpu',
 'label_gain': [0, 1],
 'n_estimators': 300,
 'learning_rate': 0.05,
 'num_leaves': 31,
 'max_depth': 5,
 'min_child_samples': 100,
 'reg_lambda': 5.0,
 'max_bin': 63,
 'lambdarank_truncation_level': 13,
 'random_state': 1271,
 'n_jobs': 4,
 'deterministic': True,
 'force_col_wise': True,
 'verbosity': -1}


STOP = set('a an the and or but if then of in on at for from to with by as is are was were be been being do does did have has had can could would should will may might i me my mine we us our you your yours he him his she her hers it its they them their this that these those what which who how why when where please thanks thank hello hi hey okay ok yes sure help need want like know tell give suggest suggestions suggestion recommend recommendations recommendation ideas idea fun some any something anything about'.split())


PREFIXES = {'memory': 'title: none | text: ', 'question': 'task: search result | query: '}


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def ranks(values):
    order = np.argsort(-np.asarray(values), kind='stable')
    result = np.empty(len(values), dtype=int)
    result[order] = np.arange(1, len(values)+1)
    return result


def score_features(values):
    values = np.asarray(values, dtype=float)
    return [values, ranks(values)/len(values),
            (values-values.mean())/max(float(values.std()), 1e-12), values.max()-values]


def fusion(cosine, lexical):
    cr, lr = ranks(cosine), ranks(lexical)
    votes = 1./(60+lr)
    for value in np.unique(lexical):
        mask = lexical == value
        votes[mask] = float(votes[mask].mean()) if value > 0 else 0.
    return 1./(60+cr) + votes


class _Index:
    def __init__(self, terms, statistics, stop=False):
        self.count = len(terms)
        self.lengths = np.array([len(x) for x in terms], dtype=float)
        self.average = max(float(self.lengths.mean()), 1)
        self.postings = defaultdict(list)
        for i, words in enumerate(terms):
            for term, frequency in Counter(words).items():
                if not stop or term not in STOP:
                    self.postings[term].append((i, frequency))
        self.statistics = statistics

    def score(self, terms, *, global_idf=False, weighted=False, k1=1.5):
        scores = np.zeros(self.count)
        for term in sorted(set(terms)):
            postings = self.postings.get(term, ())
            if not postings:
                continue
            n = self.statistics['n'] if global_idf else self.count
            df = self.statistics['df'].get(term, 0) if global_idf else len(postings)
            idf = math.log1p((n-df+.5)/(df+.5))
            weight = 1.
            if weighted:
                nq = self.statistics['nq']
                weight *= max(.1, min(1., math.log((nq+1)/(self.statistics['qdf'].get(term, 0)+1))/math.log(nq+1)))
            for i, frequency in postings:
                # b=0 for every retained BM25 channel.
                scores[i] += weight*idf*frequency*(k1+1)/(frequency+k1)
        return scores


def dense_spans(text):
    normalized = unicodedata.normalize('NFKC', text).casefold()
    words = list(re.finditer(r'\w+', normalized))
    if len(words) <= 64:
        return [text]
    return [normalized[words[j].start():words[min(j+47, len(words)-1)].end()]
            for j in range(0, len(words), 24)]


def relative_position(indices):
    indices = _integer_vector(indices, 'Source indices')
    if np.any(np.diff(indices) <= 0):
        raise ValueError('Visible source indices must be nonempty, unique and increasing')
    return ((indices-indices[0])/max(int(indices[-1]-indices[0]), 1)).astype(np.float32)


def _integer_vector(values, name):
    raw = np.asarray(values)
    if (raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer)
            or np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max)):
        raise ValueError(f'{name} must be a nonempty vector of nonnegative int64 integers')
    return raw.astype(np.int64, copy=False)


def _validate_messages(messages, source_end_index=None):
    ids = [m['message_id'] for m in messages]
    indices = [m['message_index'] for m in messages]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Expected a nonempty list of unique original messages')
    relative_position(indices)
    if source_end_index is not None:
        if isinstance(source_end_index, bool) or not isinstance(source_end_index, Integral) or source_end_index < 0:
            raise ValueError('source_end_index must be a nonnegative exclusive integer cutoff')
        # raw_datasets.visible_messages uses history[:source_end_index].
        if indices[-1] >= source_end_index:
            raise ValueError('A message reaches or exceeds the exclusive visibility cutoff')
    contexts = {m['context_id'] for m in messages if 'context_id' in m}
    if len(contexts) > 1:
        raise ValueError('All messages must belong to one visible history')
    if any(m['role'] not in ('user', 'assistant') or not isinstance(m['text'], str) or not m['text'].strip() for m in messages):
        raise ValueError('Expected complete nonempty user/assistant message text')


def _validate_statistics(statistics, terms):
    if not isinstance(statistics, Mapping):
        raise ValueError('statistics must provide n, df, nq and qdf')
    for count_name, map_name in [('n', 'df'), ('nq', 'qdf')]:
        count = statistics.get(count_name)
        frequencies = statistics.get(map_name)
        if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
            raise ValueError(f'{count_name} must be a positive integer corpus size')
        if not isinstance(frequencies, Mapping):
            raise ValueError(f'{map_name} must map terms to document frequencies')
        # Only these terms contribute to this query; avoid scanning the full
        # background vocabulary again for each question.
        for term in terms:
            df = frequencies.get(term, 0)
            if isinstance(df, bool) or not isinstance(df, Integral) or not 0 <= df <= count:
                raise ValueError(f'{map_name}[{term!r}] must be an integer between 0 and {count_name}')


def _embedding_rows(embed, texts, kind):
    raw = np.asarray(embed(texts, kind))
    if np.iscomplexobj(raw):
        raise ValueError('Embeddings must be real vectors')
    vectors = np.asarray(raw, dtype=np.float32)
    if vectors.shape != (len(texts), 768) or not np.isfinite(vectors).all():
        raise ValueError(f'Expected {len(texts)} finite 768-dimensional {kind} embeddings')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1., rtol=0., atol=2e-4):
        raise ValueError(f'{kind} embeddings must already be L2 normalized')
    return vectors


def feature_matrix(question, messages, token_lengths, statistics, embed, *, source_end_index=None):
    """Build all 55 inputs over the full chronological question-visible pool.

    token_lengths maps message_id to unprefixed EmbeddingGemma token count, or is
    an aligned list. embed(texts, 'question'|'memory') returns normalized float32
    768-dimensional vectors using the frozen encoder/prefix/pooling recipe.
    statistics is an external dictionary with n, df, nq and qdf fitted on the
    chosen background training corpus. No fitted statistics are stored here.
    Only question text, source text/role/order and supplied statistics enter X.
    source_end_index is exclusive, matching the dataset loader's history[:end].
    """
    _validate_messages(messages, source_end_index)
    n = len(messages)
    lengths = [token_lengths[m['message_id']] for m in messages] if isinstance(token_lengths, Mapping) else token_lengths
    lengths = _integer_vector(lengths, 'Token lengths')
    if len(lengths) != n:
        raise ValueError('Token lengths must align with original messages')
    if not isinstance(question, str):
        raise ValueError('question must be text')
    qt = sorted(set(tokenize(question)))
    stats = statistics
    _validate_statistics(stats, qt)
    query = _embedding_rows(embed, [question], 'question')
    vectors = _embedding_rows(embed, [m['text'] for m in messages], 'memory')
    cosine = np.clip(vectors @ query[0], -1., 1.)
    terms = [tokenize(m['text']) for m in messages]
    plain = _Index(terms, stats)
    bm = plain.score(qt, k1=2.)
    weighted_bm = plain.score(qt, k1=2., weighted=True)
    chunks, parents, counts = [], [], []
    for i, words in enumerate(terms):
        pieces = [words] if len(words) <= 64 else [words[j:j+48] for j in range(0, len(words), 24)]
        counts.append(len(pieces))
        chunks.extend(pieces)
        parents.extend([i]*len(pieces))
    chunk_index = _Index(chunks, stats, stop=True)
    chunk_scores = chunk_index.score(qt, global_idf=True, k1=1.5)
    weighted_chunks = chunk_index.score(qt, global_idf=True, k1=1.5, weighted=True)
    parents = np.array(parents)

    def aggregate(scores):
        best = [np.sort(scores[parents == i])[-2:] for i in range(n)]
        return np.array([v[-1] for v in best]), np.array([v.mean() for v in best])

    cm, ct = aggregate(chunk_scores)
    wm, wt = aggregate(weighted_chunks)
    metadata = [np.array([m['role'] == 'assistant' for m in messages]),
                np.log1p([len(t) for t in terms]), np.log1p(lengths)]
    base = np.column_stack(score_features(cosine)+metadata+score_features(bm)+[fusion(cosine, bm)]
                           +score_features(cm)+[ct, np.log1p(counts)]
                           +score_features(weighted_bm)+[fusion(cosine, weighted_bm)]
                           +score_features(wm)+[wt]).astype(np.float32)
    lookup = {m['message_index']: i for i, m in enumerate(messages)}
    neighbor_values = np.column_stack([cosine, ranks(cosine)/n, weighted_bm, ranks(weighted_bm)/n, np.ones(n)])
    neighbors = []
    for delta in (-2, -1, 1, 2):
        group = np.zeros((n, 5), dtype=np.float32)
        for i, message in enumerate(messages):
            index = message['message_index']
            if all(index+step in lookup for step in range(min(0, delta), max(0, delta)+1)):
                group[i] = neighbor_values[lookup[index+delta]]
        neighbors.append(group)
    spans = [dense_spans(m['text']) for m in messages]
    flat = [text for pieces in spans if len(pieces) > 1 for text in pieces]
    values = np.clip(_embedding_rows(embed, flat, 'memory') @ query[0], -1., 1.) if flat else []
    maximum, mean, position = [], [], 0
    for i, pieces in enumerate(spans):
        if len(pieces) > 1:
            top = np.sort(values[position:position+len(pieces)])[-2:]
            position += len(pieces)
            maximum.append(top[-1])
            mean.append(top.mean())
        else:
            maximum.append(cosine[i])
            mean.append(cosine[i])
    maximum = np.asarray(maximum)
    dense = np.column_stack(score_features(maximum)+[mean, maximum-cosine]).astype(np.float32)
    relative = relative_position([m['message_index'] for m in messages])
    X = np.column_stack([base, *neighbors, dense, relative]).astype(np.float32)
    if X.shape != (n, 55) or not np.isfinite(X).all():
        raise ValueError('Invalid 55-feature matrix')
    return X


FEATURE_GROUPS = {'base': 28, 'neighbors': 20, 'dense_chunks': 6, 'relative_position': 1}
EMBEDDING_RECIPE = {
    'repo': 'google/embeddinggemma-2',
    'revision': '914f7f89142e33e77833254d9c9b90c3cef7303b',
    'dimensions': 768,
    'maximum_complete_input_tokens': 1200,
    'pooling': 'attention_mask_mean_including_prefix_then_L2',
    'prefixes': PREFIXES,
}


def validate_training_data(X, y, offsets, feature_names):
    """Check a contiguous question-grouped matrix against the frozen schema."""
    if np.iscomplexobj(X):
        raise ValueError('Training features must be real values')
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if list(feature_names) != FEATURE_NAMES:
        raise ValueError('feature_names must match all 55 ordered FEATURE_NAMES')
    if X.ndim != 2 or X.shape[1] != 55 or not len(X) or not np.isfinite(X).all():
        raise ValueError('X must be a nonempty finite matrix with 55 columns')
    if y.ndim != 1 or len(y) != len(X) or not np.isin(y, [0, 1]).all():
        raise ValueError('y must contain one binary reference label per row')
    offsets = _integer_vector(offsets, 'Question offsets')
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(X) or np.any(np.diff(offsets) <= 0):
        raise ValueError('offsets must partition all rows into nonempty questions')
    positive_counts = np.add.reduceat(y.astype(np.int64), offsets[:-1])
    if np.any(positive_counts == 0):
        raise ValueError('Each training question must have at least one reference positive')
    if not np.any(positive_counts < np.diff(offsets)):
        raise ValueError('Training requires at least one question with both positive and negative candidates')
    return X, y.astype(np.int8), offsets


def _model_output_path(output_model, features=None):
    if output_model is None:
        return None
    target = Path(output_model).expanduser().absolute()
    protected = [Path(__file__)] + ([Path(features)] if features is not None else [])
    for source in protected:
        if target.resolve() == source.resolve() or (target.exists() and source.exists() and os.path.samefile(target, source)):
            raise ValueError('Model output must differ from the training input and training script')
    if target.is_dir():
        raise ValueError('Model output must name a file, not a directory')
    return target


def _save_model_atomic(booster, target):
    """Replace an existing output only after the complete new model is written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=target.parent, prefix=f'.{target.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(booster.model_to_string())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def training_parameters(*, min_child_samples=100, num_leaves=31, max_depth=5, reg_lambda=5.0):
    """Override only the requested tuning controls; preserve the feature recipe."""
    if (type(min_child_samples) is not int or min_child_samples < 1
            or type(num_leaves) is not int or num_leaves < 2
            or type(max_depth) is not int or max_depth < 1
            or num_leaves > 2 ** max_depth
            or not math.isfinite(reg_lambda) or reg_lambda < 0):
        raise ValueError('Invalid leaf size, leaf/depth combination, or L2 regularization')
    return {**PARAMETERS, 'min_child_samples': min_child_samples, 'num_leaves': num_leaves,
            'max_depth': max_depth, 'reg_lambda': reg_lambda}


def train_ranker(X, y, offsets, *, feature_names=FEATURE_NAMES, output_model=None, **tuning):
    """Fit the fixed 300-tree recipe; optionally save the newly trained model.

    No parameter search, early stopping or held-out split is used here. For
    persona-separated evaluation, call this on each training fold and score its
    held-persona fold externally. Keep all rows of each question together.
    An existing model output is replaced atomically after successful fitting.
    """
    import lightgbm as lgb
    target = _model_output_path(output_model)
    X, y, offsets = validate_training_data(X, y, offsets, feature_names)
    model = lgb.LGBMRanker(**training_parameters(**tuning))
    model.fit(X, y, group=np.diff(offsets), feature_name=FEATURE_NAMES)
    if target is not None:
        _save_model_atomic(model.booster_, target)
    return model


def train_from_npz(features, output_model, **tuning):
    """Load external prepared features and fit; never embeds data in this file."""
    target = _model_output_path(output_model, features)
    if target is None:
        raise ValueError('An output model path is required')
    data = np.load(features, allow_pickle=False)
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError('Expected an NPZ archive containing the training arrays')
    with data:
        required = {'X', 'y', 'offsets', 'feature_names'}
        missing = required - set(data.files)
        if missing:
            raise ValueError('Missing training arrays: '+', '.join(sorted(missing)))
        # Decompress each array once and close the input before fitting/saving.
        X, y, offsets, names = (data[key] for key in ('X', 'y', 'offsets', 'feature_names'))
    model = train_ranker(X, y, offsets, feature_names=names, output_model=target, **tuning)
    return {'features': model.n_features_in_, 'trees': model.booster_.num_trees(),
            'questions': len(offsets)-1, 'candidate_rows': len(y),
            'positive_rows': int(np.sum(y)), 'output_model': str(target.resolve())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true', help='Print schema and fixed training parameters')
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--features', type=Path, default=root / 'data/ranker55/v2/train/features.npz',
                        help='Training NPZ (default: data/ranker55/v2/train/features.npz)')
    parser.add_argument('--output', type=Path, default=root / 'models/selector/ranker55.txt',
                        help='Destination model (default: models/selector/ranker55.txt); overwritten atomically')
    parser.add_argument('--min-child-samples', type=int, default=100)
    parser.add_argument('--num-leaves', type=int, default=31)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--reg-lambda', type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({'feature_names': FEATURE_NAMES, 'feature_groups': FEATURE_GROUPS,
                          'parameters': PARAMETERS, 'embedding_recipe': EMBEDDING_RECIPE}, indent=2))
    elif args.features and args.output:
        try:
            result = train_from_npz(args.features, args.output, min_child_samples=args.min_child_samples,
                                    num_leaves=args.num_leaves, max_depth=args.max_depth, reg_lambda=args.reg_lambda)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
    else:
        parser.error('Use --describe, or --features PATH --output PATH')


if __name__ == '__main__':
    main()
