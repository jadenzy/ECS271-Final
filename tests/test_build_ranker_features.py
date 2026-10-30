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
    def test_statistics_auto_detection_precedence_overrides_and_fresh_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / 'data/ranker55/statistics.json'
            bundled = root / 'data/ranker55/v2/train/statistics.json'
            custom = root / 'data/ranker55/custom/statistics.json'
            self.assertIsNone(builder.resolve_statistics(root=root))
            bundled.parent.mkdir(parents=True)
            bundled.write_text('{}')
            self.assertEqual(builder.resolve_statistics(root=root), bundled)
            shared.write_text('{}')
            self.assertEqual(builder.resolve_statistics(root=root), shared)
            custom.parent.mkdir()
            self.assertEqual(builder.resolve_statistics(root=root, output_root=custom.parent), shared)
            custom.write_text('{}')
            self.assertEqual(builder.resolve_statistics(root=root, output_root=custom.parent), custom)
            override = root / 'explicit.json'
            self.assertEqual(builder.resolve_statistics(root=root, statistics=override), override)
            self.assertIsNone(builder.resolve_statistics(root=root, fit_question_ids=root / 'fold.json'))
            self.assertIsNone(builder.resolve_statistics(root=root, refit_statistics=True))
            with self.assertRaises(ValueError):
                builder.resolve_statistics(root=root, statistics=shared, refit_statistics=True)

    def test_df_counts_unique_source_occurrences_and_qdf_counts_question_presence(self):
        qs = [{'example_id': 'v2:train:0', 'dataset': 'v2', 'split': 'train', 'persona_id': 'v2:1',
               'memory_ids': ['a', 'b'], 'question': 'tea tea ideas'},
              {'example_id': 'v2:train:1', 'dataset': 'v2', 'split': 'train', 'persona_id': 'v2:1',
               'memory_ids': ['a', 'b'], 'question': 'tea coffee'}]
        memories = {'a': {'text': 'tea tea'}, 'b': {'text': 'tea'}}
        stats = builder.fit_statistics(qs, memories, {})
        self.assertEqual(stats['n'], 2)  # not four question-message pairs, not one text
        self.assertEqual(stats['df'], {'tea': 2})
        self.assertEqual(stats['qdf'], {'coffee': 1, 'ideas': 1, 'tea': 2})
        self.assertEqual(stats['nq'], 2)
        with self.assertRaisesRegex(ValueError, 'active V2 training'):
            builder.fit_statistics([{**qs[0], 'split': 'val'}], memories, {})

    def test_retained_bm25_and_question_weight_formula(self):
        stats = {'n': 2, 'df': {'tea': 1}, 'nq': 10, 'qdf': {'tea': 10}}
        index = builder.recipe._Index([['tea', 'tea'], ['coffee']], stats)
        expected = np.log(2) * 2 * 3 / (2 + 2)
        self.assertAlmostEqual(index.score(['tea'], k1=2)[0], expected)
        self.assertAlmostEqual(index.score(['tea'], k1=2, weighted=True)[0], .1 * expected)
        np.testing.assert_array_equal(index.score(['missing']), [0, 0])
        self.assertEqual(builder.recipe.tokenize('CAFÉ ＴＥＡ, 中文!'), ['café', 'tea', '中文'])


class BundleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_fixture(self.root)
        self.encoder = DistinctEncoder()
        self.info = {k: builder.recipe.EMBEDDING_RECIPE[k] for k in ('repo', 'revision')}
        self.bundle = self.root / 'data/ranker55/v2/train'
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
            return builder.run(root=self.root, encoder=self.encoder, model_info=self.info, **kwargs)

    def test_full_export_matches_recipe_and_keeps_reference_targets_separate(self):
        before = {p: sha256(p) for p in (self.root / 'data/raw').rglob('*') if p.is_file()}
        report = self.build()['v2/train']
        fitted, tasks = builder.prepare_inputs(root=self.root)
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
                expected = builder.recipe.feature_matrix(q['question'], members, bare, fitted, self.encoder.encode)
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

    def test_existing_statistics_are_reused_without_rewriting_and_refit_is_explicit(self):
        self.build()
        shared = self.root / 'data/ranker55/statistics.json'
        before = shared.read_bytes(), shared.stat().st_mtime_ns
        with patch.object(builder, 'fit_statistics', side_effect=AssertionError('Unexpected refit')):
            self.build()
        self.assertEqual((shared.read_bytes(), shared.stat().st_mtime_ns), before)
        with patch.object(builder, 'fit_statistics', wraps=builder.fit_statistics) as fit:
            self.build(refit_statistics=True)
        fit.assert_called_once()

    def test_missing_or_invalid_explicit_statistics_do_not_fall_back(self):
        self.build()
        explicit = self.root / 'explicit.json'
        with self.assertRaises(FileNotFoundError):
            builder.prepare_inputs(root=self.root, statistics=explicit)
        explicit.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'frozen training-only'):
            builder.prepare_inputs(root=self.root, statistics=explicit)

    def test_question_limit_does_not_shrink_fitted_statistics_and_fold_ids_do(self):
        self.build(limit_questions=1)
        stats = json.loads((self.root / 'data/ranker55/statistics.json').read_text())
        self.assertEqual(stats['nq'], 2)
        with np.load(self.bundle / 'features.npz') as data:
            self.assertEqual(len(data['question_ids']), 1)
        ids = self.root / 'fit_ids.json'
        ids.write_text('["v2:train:00001"]')
        fitted, tasks = builder.prepare_inputs(root=self.root, fit_question_ids=ids, limit_questions=1)
        self.assertEqual(fitted['nq'], 1)
        self.assertEqual(fitted['fit_question_ids'], ['v2:train:00001'])
        self.assertEqual(tasks[0][2][0]['example_id'], 'v2:train:00000')

    def test_frozen_statistics_support_v1_cutoffs_and_empty_chunk_cache(self):
        self.build()
        frozen = self.root / 'data/ranker55/statistics.json'
        digest = sha256(frozen)
        report = self.build(datasets=['v1', 'locomo'], statistics=frozen)
        self.assertEqual(set(report), {'v1/comparison', 'locomo/comparison'})
        self.assertEqual(sha256(frozen), digest)
        path = self.root / 'data/ranker55/v1/comparison'
        with np.load(path / 'features.npz') as data:
            self.assertEqual(np.diff(data['offsets']).tolist(), [1, 2])
            self.assertEqual(data['message_indices'].tolist(), [1, 1, 2])
            self.assertTrue(np.all(data['y'] == -1))
            self.assertTrue(np.all(data['judged_labels'] == -1))
        chunks = np.load(path / 'embeddings/chunk_embeddings.npy', allow_pickle=False)
        self.assertEqual(chunks.shape, (0, 768))
        self.assertFalse((path / 'reference_labels').exists())

    def test_held_out_requires_statistics_and_does_not_change_question_df(self):
        frozen = self.root / 'data/ranker55/statistics.json'
        csv_path = self.root / 'data/raw/personamem_v2/benchmark/text/val.csv'
        with csv_path.open(encoding='utf-8', newline='') as stream:
            row = next(csv.DictReader(stream))
        row.update(persona_id='8', user_query=json.dumps({'role': 'user', 'content': 'heldoutonlyterm'}))
        self.write_csv(csv_path, [row])
        (self.root / 'data/splits/v2/val_ids.json').write_text('["v2:val:00000"]')
        with self.assertRaisesRegex(ValueError, 'No fitted statistics.json found'):
            self.build(splits=['val'])
        self.build()
        self.build(splits=['val'])
        stats = json.loads((self.root / 'data/ranker55/v2/val/statistics.json').read_text())
        self.assertNotIn('heldoutonlyterm', stats['qdf'])
        self.assertEqual(stats, json.loads(frozen.read_text()))
        self.build(splits=['val'], statistics=frozen)
        row['persona_id'] = '7'
        self.write_csv(csv_path, [row])
        with self.assertRaisesRegex(ValueError, 'personas overlap'):
            self.build(splits=['val'], statistics=frozen)

    def test_npz_trains_existing_ranker_and_feeds_candidate_selector_and_diagnostics(self):
        self.build()
        model_path = self.root / 'models/ranker55.txt'
        result = builder.recipe.train_from_npz(self.bundle / 'features.npz', model_path)
        self.assertEqual(result['features'], 55)
        with contextlib.redirect_stdout(io.StringIO()):
            report = candidates.run(self.bundle / 'features.npz', root=self.root, diversity_weight=.1)
            diagnostic = diagnostics.run(root=self.root, make_plot=False)
        self.assertEqual(report['counts']['candidates'], 80)
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
