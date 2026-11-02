"""Graded training bundles keep judged rows (or every row) and stay usable by the ranker trainer."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_graded_training as graded
import ranker55
from subtlex_fixture import recipe as test_recipe
from tune_ranker55 import Bundle

# Per question: reference, useful candidate, two rejected candidates, two unjudged messages.
GRADES = [2, 1, 0, 0, -1, -1]


def judge(label):
    raw = {'b': -.2, 'a': -2.} if label == 0 else {'b': -2., 'a': -.2}
    norm = math.log(sum(math.exp(v) for v in raw.values()))
    return {'predicted_label': 'a' if label else 'b', 'label_raw_logprobs': raw,
            'label_conditional_logprobs': {k: v - norm for k, v in raw.items()}}


class GradedTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.qids = [f'v2:train:{i:05d}' for i in range(4)]
        self.mids = [[f'v2:32k:persona{q}:message:{j}' for j in range(len(GRADES))] for q in range(4)]
        n = len(GRADES) * len(self.qids)
        rng = np.random.default_rng(7)
        X = rng.normal(size=(n, 55)).astype(np.float32)
        X[:, 0] = np.tile(GRADES, len(self.qids))  # learnable signal for the smoke fit
        reference = np.tile([2, 0, 0, 0, 0, 0], len(self.qids)).astype(np.int8)
        self.features = self.root / 'data/ranker55/v2/train/features.npz'
        self.features.parent.mkdir(parents=True)
        np.savez(self.features, X=X, y=(reference == 2).astype(np.int8), reference_labels=reference,
                 judged_labels=np.where(reference == 2, 2, -1).astype(np.int8),
                 offsets=np.arange(0, n + 1, len(GRADES)), feature_names=np.array(ranker55.FEATURE_NAMES),
                 question_ids=np.array(self.qids), persona_ids=np.array([f'v2:{q}' for q in range(4)]),
                 message_ids=np.array(sum(self.mids, [])), message_indices=np.tile(np.arange(len(GRADES)), 4),
                 content_token_lengths=np.full(n, 10, dtype=np.int32),
                 metadata_json=np.array(json.dumps(dict(dataset='v2', split='train', memory_unit='single_message',
                                                        feature_recipe=test_recipe(), statistics_fingerprint='stats',
                                                        source_sha256={'raw': 'hash'}))))
        self.labels = self.root / 'data/qwen_labels/v2/train.jsonl'
        self.labels.parent.mkdir(parents=True)
        self.rows = [self.judged_row(q) for q in range(4)]
        self.save()

    def judged_row(self, q):
        mids = self.mids[q]
        return {'example_id': self.qids[q],
                'memories': [{'message_id': m, 'label': g} for m, g in zip(mids, GRADES)],
                'candidates': [{'message_id': m, 'label': g, 'judge': judge(g)}
                               for m, g in zip(mids, GRADES) if g in (0, 1)]}

    def save(self):
        self.labels.write_text(''.join(json.dumps(r) + '\n' for r in self.rows), encoding='utf-8')

    def build(self, **options):
        return graded.build(root=self.root, **options)

    def test_only_judged_rows_are_kept_and_the_trainer_accepts_them(self):
        report = self.build()
        self.assertTrue(report['tuner_ready'])
        self.assertEqual(report['grade_counts'], {'0': 8, '1': 4, '2': 4})
        path = self.root / 'data/ranker55/v2/train_graded/features.npz'
        with Bundle(path) as bundle:
            np.testing.assert_array_equal(bundle.read('y'), np.tile([2, 1, 0, 0], 4))
            np.testing.assert_array_equal(bundle.read('offsets'), np.arange(0, 17, 4))
            self.assertEqual(bundle.read('message_ids').tolist(), sum((m[:4] for m in self.mids), []))
            np.testing.assert_array_equal(bundle.read('X')[:, 0], np.tile([2, 1, 0, 0], 4))
            self.assertEqual(bundle.read('question_ids').tolist(), self.qids)
            metadata = bundle.metadata
        self.assertEqual(metadata['row_count'], 16)
        self.assertFalse(metadata['graded_labels']['partial'])
        self.assertEqual((metadata['dataset'], metadata['split'], metadata['statistics_fingerprint']),
                         ('v2', 'train', 'stats'))
        result = ranker55.train_from_npz(path, self.root / 'models/graded.txt',
                                         min_child_samples=1, num_leaves=3, max_depth=2)
        self.assertEqual(result['grade_counts'], {'0': 8, '1': 4, '2': 4})
        self.assertEqual(result['label_gain'], [0, 1, 2])

    def test_all_rows_keeps_every_message_and_labels_unjudged_zero(self):
        report = self.build(all_rows=True)
        self.assertTrue(report['tuner_ready'])
        self.assertEqual(report['grade_counts'], {'0': 16, '1': 4, '2': 4})
        self.assertEqual(report['unjudged_rows'], 8)
        path = self.root / 'data/ranker55/v2/train_graded_all/features.npz'
        with Bundle(path) as bundle:
            np.testing.assert_array_equal(bundle.read('y'), np.tile([2, 1, 0, 0, 0, 0], 4))
            np.testing.assert_array_equal(bundle.read('judged_labels'), np.tile(GRADES, 4))
            np.testing.assert_array_equal(bundle.read('offsets'), np.arange(0, 25, 6))
            self.assertEqual(bundle.read('message_ids').tolist(), sum(self.mids, []))
            with Bundle(self.features) as source:
                np.testing.assert_array_equal(bundle.read('X'), source.read('X'))
            metadata = bundle.metadata
        self.assertIn('all rows', metadata['graded_labels']['row_policy'])
        result = ranker55.train_from_npz(path, self.root / 'models/graded_all.txt',
                                         min_child_samples=1, num_leaves=3, max_depth=2)
        self.assertEqual(result['grade_counts'], {'0': 16, '1': 4, '2': 4})

    def test_unfinished_labeling_needs_explicit_partial_mode(self):
        self.rows.pop(1)
        self.save()
        with self.assertRaisesRegex(ValueError, 'not judged yet'):
            self.build()
        report = self.build(allow_partial=True)
        self.assertFalse(report['tuner_ready'])
        with Bundle(self.root / 'data/ranker55/v2/train_graded/features.npz') as bundle:
            self.assertEqual(bundle.read('question_ids').tolist(), [self.qids[0], *self.qids[2:]])

    def test_misaligned_or_inconsistent_labels_are_rejected(self):
        def mutate(change, message):
            self.rows = [self.judged_row(q) for q in range(4)]
            change(self.rows[0])
            self.save()
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.build()
        mutate(lambda r: r['memories'].reverse(), 'differ from the feature rows')
        mutate(lambda r: r['memories'][1].update(label=2), 'original references')
        mutate(lambda r: r['candidates'][0].update(label=0), 'disagrees')
        mutate(lambda r: r['memories'][4].update(label=0), 'exactly the Qwen-judged candidates')
        mutate(lambda r: r['memories'][4].update(label=None), '-1, 0, 1 or 2')

    def test_judge_sources_must_match_the_bundle_sources_they_share(self):
        report = self.labels.parent / 'report.json'
        report.write_text(json.dumps({'source_sha256': {'raw': 'hash'}}), encoding='utf-8')
        self.build()  # a subset of the bundle's sources is consistent
        report.write_text(json.dumps({'source_sha256': {'raw': 'changed'}}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'different raw sources'):
            self.build()

    def test_outputs_stay_under_ranker55_and_apart_from_the_source(self):
        for output in (self.root / 'models/graded', self.features.parent):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'data/ranker55'):
                self.build(output_root=output)


if __name__ == '__main__':
    unittest.main()
