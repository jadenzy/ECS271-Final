import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_subtlex_us import prepare


class SubtlexTests(unittest.TestCase):
    def test_external_tokens_handle_numbers_unicode_case_and_contractions(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'raw.dat', Path(directory) / 'words.json'
            source.write_text("Word\tFREQcount\nＣＡＦÉ\t2\ncafé\t3\nCan't\t4\n123\t5\n", encoding='utf-8')
            words = prepare(source, target, 'utf-8')['words']
            self.assertEqual({w: v['count'] for w, v in words.items()},
                             {'café': 5, 'can': 4, 't': 4, '123': 5})
            self.assertAlmostEqual(sum(v['probability'] for v in words.values()), 1)

    def test_normalization_counts_split_words_and_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'raw.dat'
            target = Path(directory) / 'words.json'
            source.write_text('Word\tFREQcount\tCDcount\nTea\t3\t1\ntea\t2\t1\n---\t4\t1\nTea-time\t2\t1\nＴＥＡ\t1\t1\n', encoding='utf-8')
            result = prepare(source, target, 'utf-8')
            self.assertEqual(result['words']['tea']['count'], 8)
            self.assertEqual(result['words']['time']['count'], 2)
            self.assertEqual(result['summary']['total_token_count'], 10)
            self.assertEqual(result['summary']['discarded_rows'], 1)
            self.assertAlmostEqual(sum(v['probability'] for v in result['words'].values()), 1)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), result)


if __name__ == '__main__':
    unittest.main()
