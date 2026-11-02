"""Pinned external token prior and bounded history-local lexical weighting."""
from collections import Counter, OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
RECIPE = 'subtlex_mix_v1'
TOKENIZATION = 'nfkc_casefold_word_v1'
NORMALIZED_RECIPE = 'subtlex_us_nfkc_casefold_word_token_frequency_v1'
BETA, EPSILON = .5, 1e-9


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be finite and between 0 and 1')


def load_prior(path, manifest):
    """Verify the exact processed artifact and raw-source provenance before use."""
    path = Path(path)
    lock = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if digest(path) != lock.get('processed_sha256'):
        raise ValueError('SUBTLEX source hash differs from the pinned processed artifact')
    document = json.loads(path.read_text(encoding='utf-8'))
    if (document.get('schema_version') != 1 or document.get('recipe') != NORMALIZED_RECIPE
            or document.get('source', {}).get('sha256') != lock.get('raw_sha256')
            or document.get('normalization') != lock.get('normalization')):
        raise ValueError('SUBTLEX raw source hash or normalization differs from the manifest')
    words = document.get('words', {})
    if not words:
        raise ValueError('SUBTLEX prior is empty')
    counts = {}
    for word, value in words.items():
        count = value.get('count')
        if tokenize(word) != [word] or type(count) is not int or count <= 0:
            raise ValueError('SUBTLEX words must be normalized tokens with positive integer counts')
        counts[word] = count
    total = sum(counts.values())
    if (total != document['summary']['total_token_count']
            or len(counts) != document['summary']['vocabulary_size']
            or any(not isinstance(v.get('probability'), (int, float))
                   or not math.isclose(v['probability'], counts[w] / total, rel_tol=1e-12, abs_tol=1e-15)
                   for w, v in words.items())):
        raise ValueError('SUBTLEX counts/probabilities do not normalize consistently')
    return counts, total, lock


def configuration(alpha, lock):
    validate_alpha(alpha)
    return {'name': RECIPE, 'alpha': float(alpha), 'beta': BETA, 'epsilon': EPSILON,
            'minimum_weight': .1, 'tokenization': TOKENIZATION,
            'processed_sha256': lock['processed_sha256'], 'raw_sha256': lock['raw_sha256'],
            'normalization': lock['normalization'],
            'local_unit': 'complete visible source-message token occurrences; no chunks',
            'chunk_idf': 'visible parent-message DF',
            'weight': 'clip(-ln(max(p_mix,epsilon))/ln(1/epsilon),0.1,1)',
            'oov': 'beta per distinct visible OOV token; renormalize external counts on union support'}


def validate_configuration(config):
    if not isinstance(config, dict) or config.get('name') != RECIPE:
        raise ValueError('Unknown SUBTLEX feature recipe')
    for key in ('processed_sha256', 'raw_sha256'):
        if not isinstance(config.get(key), str) or re.fullmatch(r'[0-9a-f]{64}', config[key]) is None:
            raise ValueError('SUBTLEX recipe needs pinned source hashes')
    if not isinstance(config.get('normalization'), dict):
        raise ValueError('SUBTLEX recipe needs normalization metadata')
    expected = configuration(config.get('alpha'), config)
    if config != expected:
        raise ValueError('SUBTLEX recipe parameters or normalization are incompatible')


class HistoryPrior:
    def __init__(self, terms, external_counts, external_total):
        self.counts, self.df = Counter(), Counter()
        for words in terms:
            self.counts.update(words)
            self.df.update(set(words))
        self.n = len(terms)
        self.total = sum(self.counts.values())
        self.external = external_counts
        self.external_total = external_total + BETA * sum(w not in external_counts for w in self.counts)

    def probability(self, word, alpha):
        validate_alpha(alpha)
        if not self.total or (word not in self.external and word not in self.counts):
            return 0.
        external = self.external.get(word, BETA) / self.external_total
        return alpha * external + (1 - alpha) * self.counts[word] / self.total

    def weight(self, word, alpha):
        return max(.1, min(1., -math.log(max(self.probability(word, alpha), EPSILON)) / math.log(1 / EPSILON)))

    def weights(self, words, alpha):
        validate_alpha(alpha)
        return {w: self.weight(w, alpha) for w in set(words)} if self.total else {}


class Mixture:
    """A bounded cache per process, keyed by history identity and exact cutoff/pool."""
    def __init__(self, counts, total, config):
        validate_configuration(config)
        self.counts, self.total, self.config = counts, total, config
        self.histories = OrderedDict()

    def history(self, question, messages):
        key = (question['history_id'], question['source_end_index'], tuple(m['message_id'] for m in messages))
        if key not in self.histories:
            self.histories[key] = HistoryPrior([tokenize(m['text']) for m in messages], self.counts, self.total)
            if len(self.histories) > 32:
                self.histories.popitem(last=False)
        self.histories.move_to_end(key)
        return self.histories[key]
