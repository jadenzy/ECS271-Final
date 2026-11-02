"""External/local token probabilities, inference visibility, and recipe identity."""
import math
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import subtlex_mixture as mixture
from prepare_subtlex_us import prepare


def prior_fixture(root):
    source = root / 'subtlex.dat'
    source.write_text('Word\tFREQcount\tCDcount\nTea\t6\t1\ncoffee\t2\t1\n', encoding='utf-8')
    prior = root / 'word_frequencies.json'
    document = prepare(source, prior, 'utf-8')
    manifest = root / 'subtlex.lock.json'
    lock = {'processed_sha256': mixture.digest(prior), 'raw_sha256': document['source']['sha256'],
            'normalization': document['normalization']}
    manifest.write_text(json.dumps(lock), encoding='utf-8')
    return prior, manifest


class MixtureTests(unittest.TestCase):
    def test_builder_exposes_only_subtlex_and_defaults_alpha_to_half(self):
        import build_ranker_features as builder
        _, args = builder.parse_args([])
        self.assertFalse(hasattr(args, 'lexical_recipe'))
        self.assertFalse(hasattr(args, 'statistics'))
        self.assertEqual(args.alpha, .5)

    def test_probabilities_sum_to_one_and_alpha_endpoints(self):
        history = mixture.HistoryPrior([['tea', 'tea'], ['coffee', 'newword']], {'tea': 6, 'coffee': 2}, 8)
        for alpha in (0, .25, .5, 1):
            values = [history.probability(w, alpha) for w in ('tea', 'coffee', 'newword')]
            self.assertAlmostEqual(math.fsum(values), 1)
        self.assertEqual(history.probability('tea', 0), .5)
        self.assertAlmostEqual(history.probability('tea', 1), 6 / 8.5)
        self.assertEqual(history.probability('missing', .5), 0)
        self.assertTrue(.1 <= history.weight('newword', .5) <= 1)
        for alpha in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                history.probability('tea', alpha)

    def test_normalization_numeric_case_unicode_contractions_and_occurrences(self):
        terms = [mixture.tokenize("ＴＥＡ tea café CAFÉ can't 123"), mixture.tokenize('tea')]
        history = mixture.HistoryPrior(terms, {'tea': 6}, 6)
        self.assertEqual(history.counts['tea'], 3)
        self.assertEqual(history.counts['café'], 2)
        self.assertEqual(history.counts['can'], 1)
        self.assertEqual(history.counts['t'], 1)
        self.assertEqual(history.counts['123'], 1)
        self.assertEqual(history.df['tea'], 2)
        self.assertEqual(history.total, 8)
        self.assertEqual(mixture.HistoryPrior([], {'tea': 6}, 6).weights(['tea'], .5), {})

    def test_counts_are_whole_message_based_not_overlapping_chunks(self):
        history = mixture.HistoryPrior([['tea'] * 100, ['tea'] * 100], {'tea': 6}, 6)
        self.assertEqual(history.counts['tea'], 200)
        self.assertEqual(history.df['tea'], 2)
        self.assertEqual(history.probability('tea', 0), 1)

    def test_source_hashes_and_normalization_are_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            prior, manifest = prior_fixture(Path(directory))
            counts, total, lock = mixture.load_prior(prior, manifest)
            self.assertEqual((counts, total), ({'tea': 6, 'coffee': 2}, 8))
            prior.write_bytes(prior.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                mixture.load_prior(prior, manifest)
            lock['processed_sha256'] = mixture.digest(prior)
            lock['raw_sha256'] = '0' * 64
            manifest.write_text(json.dumps(lock), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'raw source hash'):
                mixture.load_prior(prior, manifest)

    def test_history_cache_ignores_dataset_and_labels_but_respects_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            counts, total, lock = mixture.load_prior(*prior_fixture(Path(directory)))
            engine = mixture.Mixture(counts, total, mixture.configuration(.5, lock))
            messages = [{'message_id': 'a', 'text': "TEA tea can't 123"},
                        {'message_id': 'b', 'text': 'TEA tea'}]
            q = {'history_id': 'history', 'source_end_index': 2}
            first = engine.history(q, messages)
            for dataset in ('v1', 'v2', 'locomo'):
                changed = {**q, 'dataset': dataset, 'answer': 'unseen', 'options': ['unseen'],
                           'reference': 'unseen', 'question': 'different query'}
                self.assertIs(engine.history(changed, messages), first)
                independent = mixture.Mixture(counts, total, mixture.configuration(.5, lock))
                self.assertEqual(independent.history(changed, messages).weights(['tea', '123', 'other'], .5),
                                 first.weights(['tea', '123', 'other'], .5))
            self.assertEqual(first.counts['tea'], 4)  # distinct source occurrences survive
            earlier = engine.history({**q, 'source_end_index': 1}, messages[:1])
            self.assertEqual(earlier.counts['tea'], 2)
            self.assertNotEqual(earlier.probability('tea', .5), first.probability('tea', .5))


if __name__ == '__main__':
    unittest.main()
