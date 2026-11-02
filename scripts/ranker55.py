"""Training source for the selected 55-feature LightGBM LambdaRank architecture.

28 base features + 20 neighbor features + 6 dense-chunk features + relative
conversation position. The file contains no fitted model, corpus statistics,
training samples, saved predictions, or encoded data.

Train from an external NPZ with X, y, offsets and feature_names:
  python -B scripts/ranker55.py
Defaults: data/ranker55/v2/train/features.npz -> models/subtlex_mix/alpha_0.5/ranker55.txt.
Existing model output files are replaced atomically after successful training.
Inspect the recipe without fitting:
  python -B scripts/ranker55.py --describe

feature_matrix() constructs inputs when given a SUBTLEX/history lexical prior,
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
from feature_contract import bundle_recipe, save_model_recipe, validate_recipe

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
 'label_gain': [0, 1, 2],
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

    def score(self, terms, *, global_idf=False, weighted=False, k1=1.5, term_weights=None):
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
                if term_weights is None:
                    raise ValueError('Weighted BM25 requires SUBTLEX mixture term weights')
                weight = term_weights[term]
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


def feature_matrix(question, messages, token_lengths, lexical, embed, *, source_end_index=None):
    """Build all 55 inputs over the full chronological question-visible pool.

    token_lengths maps message_id to unprefixed EmbeddingGemma token count, or is
    an aligned list. embed(texts, 'question'|'memory') returns normalized float32
    768-dimensional vectors using the frozen encoder/prefix/pooling recipe.
    lexical=(HistoryPrior, alpha) supplies visible parent-message DF and
    mixed external/history term weights. Only question text, visible source
    text/role/order, and the pinned external prior enter X.
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
    history, alpha = lexical
    if history.n != n:
        raise ValueError('Lexical history must match the complete visible pool')
    stats = {'n': history.n, 'df': history.df}
    term_weights = history.weights(qt, alpha)
    query = _embedding_rows(embed, [question], 'question')
    vectors = _embedding_rows(embed, [m['text'] for m in messages], 'memory')
    cosine = np.clip(vectors @ query[0], -1., 1.)
    terms = [tokenize(m['text']) for m in messages]
    plain = _Index(terms, stats)
    bm = plain.score(qt, k1=2.)
    weighted_bm = plain.score(qt, k1=2., weighted=True, term_weights=term_weights)
    chunks, parents, counts = [], [], []
    for i, words in enumerate(terms):
        pieces = [words] if len(words) <= 64 else [words[j:j+48] for j in range(0, len(words), 24)]
        counts.append(len(pieces))
        chunks.extend(pieces)
        parents.extend([i]*len(pieces))
    chunk_index = _Index(chunks, stats, stop=True)
    chunk_scores = chunk_index.score(qt, global_idf=True, k1=1.5)
    weighted_chunks = chunk_index.score(qt, global_idf=True, k1=1.5, weighted=True, term_weights=term_weights)
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
    if y.ndim != 1 or len(y) != len(X) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError('y must contain one relevance grade (0, 1 or 2) per row; exclude unjudged rows')
    offsets = _integer_vector(offsets, 'Question offsets')
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(X) or np.any(np.diff(offsets) <= 0):
        raise ValueError('offsets must partition all rows into nonempty questions')
    positive_counts = np.add.reduceat((y > 0).astype(np.int64), offsets[:-1])
    if np.any(positive_counts == 0):
        raise ValueError('Each training question must have at least one positive relevance grade')
    if not any(np.any(y[start:end] != y[start]) for start, end in zip(offsets[:-1], offsets[1:])):
        raise ValueError('Training requires at least one question with two distinct relevance grades')
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


# ask_to_forget questions are easy (about 96% validation Hit@10) and about a fifth of training.
# At full weight they pulled the ranker away from preference evidence: weight 0.5 raised
# non-forget type-macro Hit@10 from 54.81% to 56.00% while forget questions went 96.3% -> 93.2%.
FORGET_QUESTION_WEIGHT = 0.5


def forget_question_weights(question_ids, forget_weight=FORGET_QUESTION_WEIGHT, *, root=None):
    """Per-question training weights from the raw question types; 1 for every other type."""
    if not math.isfinite(forget_weight) or forget_weight <= 0:
        raise ValueError('forget_weight must be positive')
    if forget_weight == 1:
        return np.ones(len(question_ids))
    from message_dataset import load_message_dataset
    questions, _, _ = load_message_dataset('v2', ['train'], **({'root': root} if root is not None else {}))
    types = {q['example_id']: q['question_type'] for q in questions}
    if any(qid not in types for qid in question_ids):
        raise ValueError('Every weighted question must be an active V2 training question')
    return np.array([forget_weight if types[qid] == 'ask_to_forget' else 1. for qid in question_ids])


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


def row_weights(question_weights, offsets):
    """Repeat one positive weight per question across that question's rows."""
    weights = np.asarray(question_weights, dtype=float)
    if weights.shape != (len(offsets) - 1,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('question_weights needs one finite positive weight per question')
    return np.repeat(weights, np.diff(offsets))


def train_ranker(X, y, offsets, *, feature_names=FEATURE_NAMES, output_model=None,
                 feature_recipe=None, question_weights=None, **tuning):
    """Fit the fixed 300-tree recipe; optionally save the newly trained model.

    No parameter search, early stopping or held-out split is used here. For
    persona-separated evaluation, call this on each training fold and score its
    held-persona fold externally. Keep all rows of each question together.
    question_weights (one per question, e.g. forget_question_weights) scale whole
    questions; None weights every question equally.
    An existing model output is replaced atomically after successful fitting.
    """
    import lightgbm as lgb
    feature_recipe = validate_recipe(feature_recipe)
    target = _model_output_path(output_model)
    X, y, offsets = validate_training_data(X, y, offsets, feature_names)
    weights = None if question_weights is None else row_weights(question_weights, offsets)
    model = lgb.LGBMRanker(**training_parameters(**tuning))
    model.fit(X, y, group=np.diff(offsets), sample_weight=weights, feature_name=FEATURE_NAMES)
    if target is not None:
        _save_model_atomic(model.booster_, target)
        save_model_recipe(target, feature_recipe)
    return model


def train_from_npz(features, output_model, *, question_weights=None, **tuning):
    """Load external prepared features and fit; never embeds data in this file.

    question_weights follow the NPZ question order; the CLI derives them from
    question_ids with forget_question_weights.
    """
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
    model = train_ranker(X, y, offsets, feature_names=names, output_model=target,
                         feature_recipe=bundle_recipe(features), question_weights=question_weights, **tuning)
    weights = np.ones(len(offsets) - 1) if question_weights is None else np.asarray(question_weights, dtype=float)
    return {'features': model.n_features_in_, 'trees': model.booster_.num_trees(),
            'questions': len(offsets)-1, 'candidate_rows': len(y),
            'positive_rows': int(np.count_nonzero(y > 0)),
            'grade_counts': {str(grade): int(np.count_nonzero(y == grade)) for grade in (0, 1, 2)},
            'question_weight_counts': {str(float(w)): int(n) for w, n in zip(*np.unique(weights, return_counts=True))},
            'label_gain': PARAMETERS['label_gain'], 'output_model': str(target.resolve())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true', help='Print schema and fixed training parameters')
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--features', type=Path, default=root / 'data/ranker55/v2/train/features.npz',
                        help='Training NPZ (default: data/ranker55/v2/train/features.npz)')
    parser.add_argument('--output', type=Path, default=root / 'models/subtlex_mix/alpha_0.5/ranker55.txt',
                        help='Destination model (default: models/subtlex_mix/alpha_0.5/ranker55.txt); overwritten atomically')
    parser.add_argument('--min-child-samples', type=int, default=100)
    parser.add_argument('--num-leaves', type=int, default=31)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--reg-lambda', type=float, default=5.0)
    parser.add_argument('--forget-weight', type=float, default=FORGET_QUESTION_WEIGHT,
                        help=f'Training weight of ask_to_forget questions (default: {FORGET_QUESTION_WEIGHT}); '
                             '1 weights every question equally')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({'feature_names': FEATURE_NAMES, 'feature_groups': FEATURE_GROUPS,
                          'parameters': PARAMETERS, 'forget_question_weight': FORGET_QUESTION_WEIGHT,
                          'embedding_recipe': EMBEDDING_RECIPE}, indent=2))
    elif args.features and args.output:
        try:
            weights = None
            if args.forget_weight != 1:
                with np.load(args.features, allow_pickle=False) as data:
                    if 'question_ids' not in data.files:
                        raise ValueError('--forget-weight needs question_ids in the training NPZ')
                    question_ids = data['question_ids'].tolist()
                weights = forget_question_weights(question_ids, args.forget_weight)
            result = train_from_npz(args.features, args.output, question_weights=weights,
                                    min_child_samples=args.min_child_samples, num_leaves=args.num_leaves,
                                    max_depth=args.max_depth, reg_lambda=args.reg_lambda)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps({**result, 'forget_question_weight': args.forget_weight}, indent=2))
    else:
        parser.error('Use --describe, or --features PATH --output PATH')


if __name__ == '__main__':
    main()
