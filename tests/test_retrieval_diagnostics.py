"""Hand-calculated rankings/metrics and strict, read-only streaming diagnostics."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import diagnose_retrieval as diagnostics
from raw_datasets import sha256


def fixture_question(example_id='v2:val:00000', positives=('a',), bm25_order=('a', 'b', 'c'),
                     cosine_order=('c', 'b', 'a'), zero_bm25=False):
    ids = ['a', 'b', 'c']
    base = {'example_id': example_id, 'dataset': 'v2', 'split': 'train',
            'history_id': 'history:' + example_id, 'memory_count': len(ids)}
    labels = {**base, 'memories': [{'message_id': mid, 'label': int(mid in positives),
                                  'message_index': i, 'role': 'user'}
                                 for i, mid in enumerate(ids)]}

    def scores(order, field, zero=False):
        return {**base, 'memories': [{'message_id': mid, 'rank': rank, 'message_index': ids.index(mid),
                                     'role': 'user', field: 0.0 if zero else float(4 - rank)}
                                    for rank, mid in enumerate(order, 1)]}
    return labels, scores(bm25_order, 'bm25_score', zero_bm25), scores(cosine_order, 'embeddinggemma2_cosine')


def write_fixture(root):
    questions = [fixture_question(), fixture_question('v2:train:00001', ('a', 'c'))]

    def save(relative, value, jsonl=False):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((''.join(json.dumps(row) + '\n' for row in value) if jsonl else json.dumps(value)), encoding='utf-8')
        return path

    ids = save('data/splits/v2/train_ids.json', [q[0]['example_id'] for q in questions])
    sources = {'data/splits/v2/train_ids.json': sha256(ids), 'fixture': 'same provenance'}
    labels = save('data/ranker55/v2/train/reference_labels/train.jsonl', [q[0] for q in questions], True)
    save('data/ranker55/v2/train/reference_labels/report.json', {'memory_unit': 'single_message', 'source_sha256': sources,
         'files_sha256': {'train.jsonl': sha256(labels)},
         'splits': {'train': {'questions': 2, 'positive_memories': 3}}})
    for root_name, index, count_key in (('bm25', 1, 'saved_bm25_scores'),
                                        ('cosine', 2, 'saved_cosine_scores')):
        save(f'data/ranker55/v2/train/{root_name}/scores/train.jsonl', [q[index] for q in questions], True)
        save(f'data/ranker55/v2/train/{root_name}/report.json', {'memory_unit': 'single_message', 'source_sha256': sources, 'top_k': None,
             'splits': {'train': {'questions': 2, count_key: 6}}})


class RankingTests(unittest.TestCase):
    def test_grade_two_references_have_identical_metrics_and_reject_utility_labels(self):
        legacy, bm25, cosine = fixture_question()
        current = copy.deepcopy(legacy)
        current['schema_version'] = 3
        for message in current['memories']:
            message['label'] *= 2
        self.assertEqual(diagnostics.fuse_question(current, bm25, cosine),
                         diagnostics.fuse_question(legacy, bm25, cosine))
        current['memories'][1]['label'] = 1
        with self.assertRaisesRegex(ValueError, 'integer 0 or 2'):
            diagnostics.fuse_question(current, bm25, cosine)

    def test_rrf_formula_source_order_ties_and_label_independence(self):
        labels, bm25, cosine = fixture_question(zero_bm25=True)
        fused, positives = diagnostics.fuse_question(labels, bm25, cosine)
        self.assertEqual([m['message_id'] for m in fused['memories']], ['a', 'c', 'b'])
        self.assertAlmostEqual(fused['memories'][0]['rrf_score'], 1 / 61 + 1 / 63)
        self.assertEqual(positives[0]['bm25_rank'], 1)
        self.assertEqual(positives[0]['cosine_rank'], 3)
        self.assertEqual(positives[0]['rrf_rank'], 1)
        changed = copy.deepcopy(labels)
        for m in changed['memories']:
            m['label'] = int(m['message_id'] == 'b')
        self.assertEqual(diagnostics.fuse_question(changed, bm25, cosine)[0], fused)

    def test_macro_metrics_are_per_question_and_percentiles_pool_all_positives(self):
        summary = diagnostics.RankSummary([1, 2, 5])
        summary.add([1, 4])
        summary.add([2])
        report = summary.report()
        self.assertEqual(report['positive_rank_median'], 2)
        self.assertAlmostEqual(report['positive_rank_p90'], 3.6)
        self.assertEqual(report['at_k']['1'], {'recall': .25, 'hit': .5, 'full_reference_coverage': 0})
        self.assertEqual(report['at_k']['2'], {'recall': .75, 'hit': 1, 'full_reference_coverage': .5})
        self.assertEqual(report['at_k']['5'], {'recall': 1, 'hit': 1, 'full_reference_coverage': 1})

    def test_missing_candidates_and_duplicate_ranks_are_rejected(self):
        labels, bm25, cosine = fixture_question()
        missing = copy.deepcopy(bm25)
        missing['memories'].pop()
        with self.assertRaisesRegex(ValueError, 'missing candidates'):
            diagnostics.fuse_question(labels, missing, cosine)
        bm25['memories'][1]['rank'] = 1
        with self.assertRaisesRegex(ValueError, 'one-based'):
            diagnostics.fuse_question(labels, bm25, cosine)
        bm25['memories'][1]['rank'] = 2.0
        with self.assertRaisesRegex(ValueError, 'one-based'):
            diagnostics.fuse_question(labels, bm25, cosine)

    def test_wrong_tie_order_and_nonfinite_scores_are_rejected(self):
        labels, bm25, cosine = fixture_question(zero_bm25=True)
        bm25['memories'][0]['bm25_score'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            diagnostics.fuse_question(labels, bm25, cosine)
        labels, bm25, cosine = fixture_question(bm25_order=('c', 'b', 'a'), zero_bm25=True)
        with self.assertRaisesRegex(ValueError, 'source-order ties'):
            diagnostics.fuse_question(labels, bm25, cosine)

    def test_no_positive_or_mismatched_question_is_never_silently_skipped(self):
        labels, bm25, cosine = fixture_question(positives=())
        with self.assertRaisesRegex(ValueError, 'no positive'):
            diagnostics.fuse_question(labels, bm25, cosine)
        labels, bm25, cosine = fixture_question()
        cosine['example_id'] = 'another-question'
        with self.assertRaisesRegex(ValueError, 'metadata/order mismatch'):
            diagnostics.fuse_question(labels, bm25, cosine)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_fixture(self.root)

    def run_diagnostics(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return diagnostics.run(root=self.root, make_plot=False, **kwargs)

    def test_full_run_records_exact_histograms_current_membership_and_unchanged_inputs(self):
        before = {p: sha256(p) for p in (self.root / 'data').rglob('*.json*')}
        report = self.run_diagnostics()
        self.assertEqual(report['questions'], 2)
        self.assertEqual(report['candidate_memories'], 6)
        self.assertEqual(report['positive_memories'], 3)
        self.assertEqual(report['cutoffs'], [5, 10, 20, 30, 50])
        output = self.root / 'results/selector_diagnostics_single_message/v2/train'
        with (output / 'positive_rank_histogram.csv').open() as stream:
            histogram = list(csv.DictReader(stream))
        for method in diagnostics.METHODS:
            self.assertEqual(sum(int(row[method]) for row in histogram), 3)
            self.assertEqual(report['methods'][method]['at_k']['5']['recall'], 1)
        fused = [json.loads(line) for line in (self.root / 'data/rrf/v2/scores/train.jsonl').read_text().splitlines()]
        self.assertEqual(fused[0]['example_id'], 'v2:val:00000')
        self.assertEqual(fused[0]['split'], 'train')
        self.assertEqual(sum(len(q['memories']) for q in fused), 6)
        for path, digest in before.items():
            self.assertEqual(sha256(path), digest)
        for path, digest in report['files_sha256'].items():
            self.assertEqual(sha256(self.root / path), digest)

    def test_truncated_input_does_not_publish_partial_or_replace_previous_outputs(self):
        self.run_diagnostics()
        output = self.root / 'results/selector_diagnostics_single_message/v2/train'
        before = sha256(output / 'summary.json')
        score_path = self.root / 'data/ranker55/v2/train/cosine/scores/train.jsonl'
        score_path.write_text(score_path.read_text().splitlines()[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'different question counts'):
            self.run_diagnostics()
        self.assertEqual(sha256(output / 'summary.json'), before)
        self.assertEqual(list(output.glob('*.tmp')), [])
        self.assertEqual(list((self.root / 'data/rrf/v2/scores').glob('*.tmp')), [])

    def test_stale_membership_or_tampered_labels_are_rejected(self):
        label_path = self.root / 'data/ranker55/v2/train/reference_labels/train.jsonl'
        original = label_path.read_bytes()
        label_path.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'Label file hash'):
            self.run_diagnostics()
        label_path.write_bytes(original)
        ids = self.root / 'data/splits/v2/train_ids.json'
        ids.write_text('["v2:val:00000"]')
        with self.assertRaisesRegex(ValueError, 'stale training membership'):
            self.run_diagnostics()

    def test_invalid_parameters_and_output_locations_are_rejected(self):
        for constant in (-1, float('nan'), float('inf')):
            with self.subTest(constant=constant), self.assertRaisesRegex(ValueError, 'RRF constant'):
                self.run_diagnostics(constant=constant)
        with self.assertRaisesRegex(ValueError, 'positive integers'):
            self.run_diagnostics(k_values=[0])
        with self.assertRaisesRegex(ValueError, 'Diagnostic outputs'):
            self.run_diagnostics(output_root=self.root / 'data/raw')
        with self.assertRaisesRegex(ValueError, 'RRF scores'):
            self.run_diagnostics(rrf_root=self.root / 'data/messages')

    def test_histogram_plot_is_written(self):
        summaries = {method: diagnostics.RankSummary([5]) for method in diagnostics.METHODS}
        for summary in summaries.values():
            summary.add([1, 2, 5])
        path = self.root / 'histogram.png'
        diagnostics.plot_histogram(path, summaries, self.root)
        self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))


if __name__ == '__main__':
    unittest.main()
