"""Linear relevance gains 0/1/2, with binary training still accepted."""
import csv
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ranker55
from subtlex_fixture import recipe as test_recipe
from test_message_retrieval import write_fixture


class RelevanceGradeTests(unittest.TestCase):
    def validate(self, y, offsets=None):
        y = np.asarray(y)
        return ranker55.validate_training_data(np.zeros((len(y), 55), dtype=np.float32), y,
                                              np.array([0, len(y)]) if offsets is None else np.asarray(offsets),
                                              ranker55.FEATURE_NAMES)

    def test_grade_two_is_accepted_and_retained(self):
        _, y, _ = self.validate([0, 1, 2])
        np.testing.assert_array_equal(y, [0, 1, 2])
        self.assertEqual(ranker55.training_parameters()['label_gain'], [0, 1, 2])

    def test_one_vs_two_is_an_informative_group_without_zero(self):
        self.validate([1, 2])
        with self.assertRaisesRegex(ValueError, 'distinct relevance grades'):
            self.validate([1, 1, 2, 2], [0, 2, 4])

    def test_binary_labels_still_work(self):
        _, y, _ = self.validate([0, 1, 0])
        np.testing.assert_array_equal(y, [0, 1, 0])

    def test_unknown_and_invalid_grades_are_rejected(self):
        for grade in (-1, 3, .5, float('nan'), float('inf')):
            with self.subTest(grade=grade), self.assertRaises(ValueError):
                self.validate([0, 1, grade])

    def test_question_weights_cover_whole_questions_and_forget_questions_count_less(self):
        offsets = np.array([0, 3, 6])
        np.testing.assert_array_equal(ranker55.row_weights([.5, 1.], offsets), [.5, .5, .5, 1., 1., 1.])
        for bad in ([1.], [0., 1.], [float('nan'), 1.]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ranker55.row_weights(bad, offsets)
        self.assertEqual(ranker55.FORGET_QUESTION_WEIGHT, .5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            path = root / 'data/raw/personamem_v2/benchmark/text/train.csv'
            with path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            rows.append({**rows[0], 'pref_type': 'ask_to_forget'})
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (root / 'data/splits/v2/train_ids.json').write_text(json.dumps(['v2:train:00000', 'v2:train:00001']))
            weights = ranker55.forget_question_weights(['v2:train:00001', 'v2:train:00000'], root=root)
            self.assertEqual(weights.tolist(), [.5, 1.])
            with self.assertRaisesRegex(ValueError, 'active V2 training'):
                ranker55.forget_question_weights(['v2:train:99999'], root=root)

    def test_real_fit_uses_linear_gains_and_preserves_grade_order(self):
        y = np.tile([0, 1, 2], 12).astype(np.int8)
        X = np.zeros((len(y), 55), dtype=np.float32)
        X[:, 0] = y
        model = ranker55.train_ranker(X, y, np.arange(0, len(y) + 1, 3), feature_recipe=test_recipe(),
                                      min_child_samples=1, num_leaves=3, max_depth=2)
        self.assertEqual(model.get_params()['label_gain'], [0, 1, 2])
        scores = model.booster_.predict(X[:3])
        self.assertTrue(scores[0] < scores[1] < scores[2])
        with tempfile.TemporaryDirectory() as directory:
            features, output = Path(directory) / 'features.npz', Path(directory) / 'model.txt'
            np.savez(features, X=X, y=y, offsets=np.arange(0, len(y) + 1, 3),
                     feature_names=np.array(ranker55.FEATURE_NAMES),
                     metadata_json=np.array(json.dumps({'feature_recipe': test_recipe()})))
            report = ranker55.train_from_npz(features, output, min_child_samples=1, num_leaves=3, max_depth=2)
            self.assertEqual(report['positive_rows'], 24)
            self.assertEqual(report['grade_counts'], {'0': 12, '1': 12, '2': 12})
            self.assertEqual(report['label_gain'], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
