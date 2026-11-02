"""Frozen statistics, complete 55-feature exports, caches and consumer integration."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_ranker_features as builder
import diagnose_retrieval as diagnostics
import select_label_candidates as candidates
from raw_datasets import read_rows, sha256
from test_embeddinggemma2 import FakeEncoder
from test_message_retrieval import write_fixture
from test_subtlex_mixture import prior_fixture


class DistinctEncoder(FakeEncoder):
    def encode(self, texts, kind):
        self.calls.append((list(texts), kind))
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256((kind + '\0' + text).encode()).digest()[:8], 'big')
            row = np.random.default_rng(seed).normal(size=768).astype(np.float32)
            vectors.append(row / np.linalg.norm(row))
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), 768)


class StatisticsTests(unittest.TestCase):
    def test_bm25_requires_explicit_mixture_weights(self):
        index = builder.recipe._Index([['tea', 'tea'], ['coffee']], {'n': 2, 'df': {'tea': 1}})
        expected = np.log(2) * 2 * 3 / (2 + 2)
        self.assertAlmostEqual(index.score(['tea'], k1=2)[0], expected)
        self.assertAlmostEqual(index.score(['tea'], k1=2, weighted=True, term_weights={'tea': .2})[0], .2 * expected)
        with self.assertRaisesRegex(ValueError, 'requires SUBTLEX'):
            index.score(['tea'], weighted=True)
        np.testing.assert_array_equal(index.score(['missing']), [0, 0])


class BundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_fixture(self.root)
        self.encoder = DistinctEncoder()
        self.info = {k: builder.recipe.EMBEDDING_RECIPE[k] for k in ('repo', 'revision')}
        self.prior, self.manifest = prior_fixture(self.root)
        self.output = self.root / 'data/ranker55'
        self.bundle = self.output / 'v2/train'
        self.long_text = ' '.join(f'Term{i},' for i in range(70)) + ' Original ending!'
        history_path = self.root / 'data/raw/personamem_v2/data/chat_history_32k/linked_history.json'
        history = json.loads(history_path.read_text())
        history['chat_history'].extend([{'role': 'user', 'content': self.long_text},
                                        {'role': 'assistant', 'content': 'I prefer tea.'}])
        history['chat_history'].extend({'role': 'user', 'content': f'Extra source message {i}'} for i in range(45))
        history_path.write_text(json.dumps(history), encoding='utf-8')
        csv_path = self.root / 'data/raw/personamem_v2/benchmark/text/train.csv'
        with csv_path.open(encoding='utf-8', newline='') as stream:
            original = next(csv.DictReader(stream))
        original['related_conversation_snippet'] = json.dumps([{'role': 'user', 'content': 'I prefer tea.'}])
        second = {**original, 'user_query': json.dumps({'role': 'user', 'content': 'Another tea idea?'})}
        self.write_csv(csv_path, [original, second])
        (self.root / 'data/splits/v2/train_ids.json').write_text(json.dumps(['v2:train:00000', 'v2:train:00001']))

    @staticmethod
    def write_csv(path, rows):
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def build(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return builder.run(root=self.root, encoder=self.encoder, model_info=self.info, subtlex_prior=self.prior, subtlex_manifest=self.manifest, **kwargs)

    def test_four_workers_match_serial_features_labels_and_score_order(self):
        self.build()
        with np.load(self.bundle / 'features.npz', allow_pickle=False) as data:
            expected = {name: data[name].copy() for name in data.files}
        paths = ['bm25/scores/train.jsonl', 'cosine/scores/train.jsonl',
                 'reference_labels/train.jsonl']
        scores = {name: (self.bundle / name).read_bytes() for name in paths}
        calls = len(self.encoder.calls)
        self.build(feature_workers=4)
        self.assertEqual(len(self.encoder.calls), calls)  # completed embeddings are reused
        with np.load(self.bundle / 'features.npz', allow_pickle=False) as data:
            for name, value in expected.items():
                np.testing.assert_array_equal(data[name], value, err_msg=name)
        for name in paths:
            self.assertEqual((self.bundle / name).read_bytes(), scores[name])

    def test_full_export_matches_recipe_and_keeps_reference_targets_separate(self):
        before = {p: sha256(p) for p in (self.root / 'data/raw').rglob('*') if p.is_file()}
        report = self.build()['v2/train']
        tasks = builder.prepare_inputs(root=self.root)
        counts, total, lock = builder.subtlex.load_prior(self.prior, self.manifest)
        mixture = builder.subtlex.Mixture(counts, total, builder.subtlex.configuration(.5, lock))
        fitted = {'schema_version': 2, 'feature_recipe': mixture.config}
        _, _, qs, memories, _, _ = tasks[0]
        with np.load(self.bundle / 'features.npz', allow_pickle=False) as data:
            self.assertEqual(data['X'].shape, (sum(len(q['memory_ids']) for q in qs), 55))
            self.assertEqual(data['feature_names'].tolist(), builder.FEATURE_NAMES)
            self.assertEqual(data['question_ids'].tolist(), [q['example_id'] for q in qs])
            self.assertEqual(data['message_ids'].tolist(), [mid for q in qs for mid in q['memory_ids']])
            self.assertEqual(set(data['y'].tolist()), {0, 1})
            self.assertEqual(set(data['reference_labels'].tolist()), {0, 2})
            self.assertEqual(set(data['judged_labels'].tolist()), {-1, 2})
            self.assertEqual(data['y'].sum(), 2)  # assistant with same text is not a gold user message
            for i, q in enumerate(qs):
                members = [memories[mid] for mid in q['memory_ids']]
                bare = self.encoder.lengths([m['text'] for m in members], 'content')
                expected = builder.recipe.feature_matrix(q['question'], members, bare, (mixture.history(q, members), .5), self.encoder.encode)
                start, end = data['offsets'][i:i + 2]
                np.testing.assert_allclose(data['X'][start:end], expected, rtol=0, atol=2e-6)
                np.testing.assert_array_equal(data['content_token_lengths'][start:end], bare)
            self.assertEqual(json.loads(str(data['metadata_json'].item()))['statistics_fingerprint'], builder.fingerprint(fitted))
        self.assertGreater(report['unique_chunk_texts'], 0)
        for name, digest in report['files_sha256'].items():
            self.assertEqual(sha256(self.bundle / name), digest)
        for path, digest in before.items():
            self.assertEqual(sha256(path), digest)
        self.assertNotIn('REFERENCE ANSWER', repr(self.encoder.calls))
        self.assertNotIn('DISTRACTOR', repr(self.encoder.calls))
        self.assertNotIn('SECRET PERSONA', repr(self.encoder.calls))
        # Auxiliary chunk embeddings do not replace or truncate original messages.
        exported = read_rows(self.bundle / 'embeddings/messages.jsonl')
        self.assertIn(self.long_text, [m['text'] for m in exported])
        self.assertEqual(len(exported), len(memories))

    def test_repeated_run_reuses_vectors_and_overwrites_exports(self):
        self.build()
        first_calls = len(self.encoder.calls)
        (self.bundle / 'features.npz').write_bytes(b'old output to overwrite')
        self.build()
        self.assertEqual(len(self.encoder.calls), first_calls)
        with np.load(self.bundle / 'features.npz') as data:
            self.assertEqual(data['X'].shape[1], 55)
        self.assertEqual(list(self.bundle.rglob('*.tmp')), [])
        self.assertFalse((self.bundle / '.features_work.npy').exists())

    def test_manifest_is_independent_of_selected_questions(self):
        self.build(limit_questions=1)
        before = json.loads((self.output / 'statistics.json').read_text())
        with np.load(self.bundle / 'features.npz') as data:
            self.assertEqual(len(data['question_ids']), 1)
        self.build(rebuild_cache=True)
        self.assertEqual(json.loads((self.output / 'statistics.json').read_text()), before)
        self.assertFalse({'n', 'df', 'nq', 'qdf'} & before.keys())

    def test_removed_options_are_rejected(self):
        for flag in ('--lexical-recipe', '--statistics', '--refit-statistics', '--fit-question-ids'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                builder.parse_args([flag, 'unused'])

    def test_frozen_statistics_support_v1_cutoffs_and_empty_chunk_cache(self):
        self.build()
        frozen = self.output / 'statistics.json'
        digest = sha256(frozen)
        report = self.build(datasets=['v1', 'locomo'])
        self.assertEqual(set(report), {'v1/comparison', 'locomo/comparison'})
        self.assertEqual(json.loads(frozen.read_text())['feature_recipe']['name'], 'subtlex_mix_v1')
        path = self.output / 'v1/comparison'
        with np.load(path / 'features.npz') as data:
            self.assertEqual(np.diff(data['offsets']).tolist(), [1, 2])
            self.assertEqual(data['message_indices'].tolist(), [1, 1, 2])
            self.assertTrue(np.all(data['y'] == -1))
            self.assertTrue(np.all(data['judged_labels'] == -1))
        chunks = np.load(path / 'embeddings/chunk_embeddings.npy', allow_pickle=False)
        self.assertEqual(chunks.shape, (0, 768))
        self.assertFalse((path / 'reference_labels').exists())

    def test_held_out_build_needs_no_training_statistics(self):
        (self.root / 'data/splits/v2/val_ids.json').write_text('["v2:val:00000"]')
        self.build(splits=['val'])
        stats = json.loads((self.output / 'v2/val/statistics.json').read_text())
        self.assertEqual(stats['feature_recipe']['name'], 'subtlex_mix_v1')
        self.assertFalse({'n', 'df', 'nq', 'qdf'} & stats.keys())

    def test_npz_trains_existing_ranker_and_feeds_candidate_selector_and_diagnostics(self):
        self.build()
        model_path = self.root / 'models/subtlex_mix/alpha_0.5/ranker55.txt'
        result = builder.recipe.train_from_npz(self.bundle / 'features.npz', model_path)
        self.assertEqual(result['features'], 55)
        with contextlib.redirect_stdout(io.StringIO()):
            report = candidates.run(self.bundle / 'features.npz', root=self.root, diversity_weight=.1)
            diagnostic = diagnostics.run(root=self.root, make_plot=False, feature_root=self.bundle)
        self.assertEqual(report['counts']['candidates'], 2 * candidates.BUDGET)
        self.assertEqual(report['counts']['references'], 2)
        self.assertEqual(diagnostic['questions'], 2)
        self.assertIn('k1=2', diagnostic['bm25_definition'])
        self.assertEqual(report['diversity_cache']['root'], str((self.bundle / 'embeddings').resolve()))

    def test_error_preserves_completed_npz_and_closes_temporary_maps(self):
        self.build()
        digest = sha256(self.bundle / 'features.npz')
        with patch.object(builder.recipe, 'feature_matrix', side_effect=RuntimeError('synthetic feature failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic feature'):
                self.build()
        self.assertEqual(sha256(self.bundle / 'features.npz'), digest)
        self.assertFalse((self.bundle / '.features_work.npy').exists())
        self.assertEqual(list(self.bundle.rglob('*.tmp')), [])

    def test_oversized_input_fails_before_any_embeddings_and_unsafe_output_fails(self):
        original = self.encoder.lengths
        def too_long(texts, kind):
            result = original(texts, kind)
            if len(result) and kind == 'memory':
                result[0] = 1201
            return result
        with patch.object(self.encoder, 'lengths', side_effect=too_long):
            with self.assertRaisesRegex(ValueError, 'No splitting or truncation'):
                self.build()
        self.assertFalse(self.encoder.calls)
        self.assertFalse((self.bundle / 'features.npz').exists())
        with self.assertRaisesRegex(ValueError, 'Outputs must stay'):
            self.build(output_root=self.root / 'data/raw')


class CacheTests(unittest.TestCase):
    def test_partial_cache_resumes_only_unfinished_rows_and_invalid_vectors_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = DistinctEncoder()
            original = encoder.encode
            count = 0
            def fail_second(texts, kind):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError('interrupted')
                return original(texts, kind)
            with patch.object(encoder, 'encode', side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    builder.cached_embeddings(root, 'memory', ['one', 'two'], [0, 1], encoder, {}, batch_size=1)
            vectors, _ = builder.cached_embeddings(root, 'memory', ['one', 'two'], [0, 1], encoder, {}, batch_size=1)
            builder.close_array(vectors)
            self.assertEqual([texts for texts, _ in encoder.calls], [['one'], ['two']])
            with self.assertRaisesRegex(ValueError, 'Stale/incomplete'):
                builder.cached_embeddings(root, 'memory', ['changed', 'two'], [0, 1], encoder, {})
            vectors, _ = builder.cached_embeddings(root, 'memory', ['changed', 'two'], [0, 1], encoder, {}, rebuild=True)
            builder.close_array(vectors)
            with patch.object(encoder, 'encode', return_value=np.zeros((2, 768), dtype=np.float32)):
                with self.assertRaisesRegex(ValueError, 'normalized'):
                    builder.cached_embeddings(root, 'memory', ['changed', 'two'], [0, 1], encoder, {}, rebuild=True)
            np.testing.assert_array_equal(np.load(root / 'memory_done.npy'), [False, False])


if __name__ == '__main__':
    unittest.main()
