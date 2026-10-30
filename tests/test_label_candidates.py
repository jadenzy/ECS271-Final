"""Synthetic candidate selection, saved-model loading, and unknown-label handling."""
from collections import Counter
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import select_label_candidates as candidates
from message_dataset import load_message_dataset
from raw_datasets import read_rows, sha256
from test_message_retrieval import write_fixture


def selection_fixture(n=90, gap=1):
    messages = [{'message_id': f'm{i}', 'message_index': gap * i,
                 'role': 'user' if i % 2 == 0 else 'assistant', 'text': f'Original message {i}'} for i in range(n)]
    question = {'example_id': 'v2:train:00000', 'memory_ids': [m['message_id'] for m in messages]}
    rng = np.random.default_rng(42)
    scores = [rng.normal(size=n) for _ in range(3)]
    return question, messages, {'m0', 'm1', 'm2'}, scores


class SelectionTests(unittest.TestCase):
    def select(self, fixture, **kwargs):
        q, messages, refs, scores = fixture
        return candidates.select_question(q, messages, refs, *scores, **kwargs)

    def test_exact_allocations_exclude_references_and_preserve_unknown_labels(self):
        fixture = selection_fixture()
        rows = self.select(fixture)
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({r['message_id'] for r in rows}), 40)
        self.assertFalse({r['message_id'] for r in rows} & fixture[2])
        self.assertEqual(Counter(r['allocation'] for r in rows), candidates.ALLOCATION)
        self.assertTrue(all(r['label'] is None and r['status'] == 'pending_qwen' for r in rows))
        self.assertEqual([r['selection_order'] for r in rows], list(range(1, 41)))
        self.assertTrue(all(r['text'] == fixture[1][int(r['message_id'][1:])]['text'] for r in rows))

    def test_channel_refill_walks_remaining_rankings_and_ties_follow_source_order(self):
        q, messages, refs, _ = selection_fixture()
        rows = self.select((q, messages, refs, [np.zeros(len(messages))] * 3))
        self.assertEqual([r['message_id'] for r in rows[:20]], [f'm{i}' for i in range(3, 23)])
        self.assertEqual([r['message_id'] for r in rows[20:25]], [f'm{i}' for i in range(23, 28)])
        self.assertEqual([r['message_id'] for r in rows[25:30]], [f'm{i}' for i in range(28, 33)])
        self.assertEqual(rows[0]['ranker_rank'], 4)  # rank computed before gold exclusion

    def test_actual_neighbors_use_source_indices_and_fallback_is_explicit(self):
        rows = self.select(selection_fixture(gap=3))
        self.assertEqual([r['source'] for r in rows if r['allocation'] == 'neighbor'], ['ranker_refill'] * 5)
        q, messages, refs, scores = selection_fixture()
        rows = self.select((q, messages, refs, scores))
        lookup = {m['message_id']: m['message_index'] for m in messages}
        anchors = {r['message_id'] for r in rows[:30]}
        for r in rows:
            if r['source'] == 'neighbor':
                self.assertIn(r['anchor_message_id'], anchors)
                self.assertNotIn(r['anchor_message_id'], refs)
                self.assertEqual(abs(r['message_index'] - lookup[r['anchor_message_id']]), r['neighbor_distance'])
                self.assertIn(r['neighbor_distance'], (1, 2))

    def test_random_selection_is_repeatable_seeded_and_outside_prior_allocations(self):
        fixture = selection_fixture()
        first, second = self.select(fixture, seed=271), self.select(fixture, seed=272)
        self.assertEqual(first, self.select(fixture, seed=271))
        self.assertEqual(first[:35], second[:35])
        self.assertNotEqual(first[35:], second[35:])
        self.assertFalse({r['message_id'] for r in first[:35]} & {r['message_id'] for r in first[35:]})

    def test_exactly_40_eligible_works_and_shortage_raises(self):
        self.assertEqual(len(self.select(selection_fixture(n=43))), 40)
        with self.assertRaisesRegex(ValueError, 'only 39 available'):
            self.select(selection_fixture(n=42))

    def test_missing_references_truncated_pool_and_nonfinite_scores_are_rejected(self):
        q, messages, refs, scores = selection_fixture()
        with self.assertRaisesRegex(ValueError, 'subset'):
            self.select((q, messages, {'unknown'}, scores))
        with self.assertRaisesRegex(ValueError, 'full visible pool'):
            self.select((q, messages[:-1], refs, scores))
        scores[0][2] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.select((q, messages, refs, scores))

    def test_diversity_is_a_soft_preference_and_keeps_identical_messages_eligible(self):
        vectors = np.zeros((22, 768), dtype=np.float32)
        vectors[:, 0] = 1
        vectors[2, 0], vectors[2, 1] = 0, 1
        similarity = vectors @ vectors.T
        self.assertEqual(candidates.diverse_pick(list(range(1, 22)), 5, {0}, None, 0), [1, 2, 3, 4, 5])
        self.assertEqual(candidates.diverse_pick(list(range(1, 22)), 5, {0}, similarity, .1)[0], 2)
        fixture = selection_fixture(n=43)
        all_identical = np.zeros((43, 768), dtype=np.float32)
        all_identical[:, 0] = 1
        rows = self.select(fixture, embeddings=all_identical, diversity_weight=.1)
        self.assertEqual(len(rows), 40)
        self.assertEqual(Counter(r['allocation'] for r in rows), candidates.ALLOCATION)


class SavedSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small real text-model fixtures verify LightGBM serialization/prediction.
        # No production weights, Qwen, GPUs, or real dataset are used.
        import lightgbm as lgb
        cls.model_texts = {}
        for size in (54, 55):
            rng = np.random.default_rng(45)
            X = rng.normal(size=(180, size)).astype(np.float32)
            y = (X[:, 0] > .6).astype(np.int8)
            model = lgb.LGBMRanker(objective='lambdarank', n_estimators=3, num_leaves=4,
                                   min_child_samples=5, n_jobs=1, verbosity=-1)
            model.fit(X, y, group=[60, 60, 60], feature_name=candidates.FEATURE_NAMES[:size])
            cls.model_texts[size] = model.booster_.model_to_string()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_fixture(self.root)
        history_path = self.root / 'data/raw/personamem_v2/data/chat_history_32k/linked_history.json'
        history = json.loads(history_path.read_text())
        history['chat_history'].extend({'role': 'user', 'content': f'Extra message {i}'} for i in range(80))
        history['chat_history'].append({'role': 'user', 'content': 'I prefer tea.'})  # duplicate gold occurrence
        history_path.write_text(json.dumps(history), encoding='utf-8')
        source = self.root / 'data/raw/personamem_v2/benchmark/text/train.csv'
        with source.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['related_conversation_snippet'] = json.dumps([{'role': 'user', 'content': 'I prefer tea.'}])
        with source.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.questions, self.memories, _ = load_message_dataset('v2', ['train'], root=self.root)
        self.feature_path = self.root / 'features.npz'
        n = len(self.questions[0]['memory_ids'])
        self.arrays = dict(X=np.random.default_rng(7).normal(size=(n, 55)).astype(np.float32),
                           offsets=np.array([0, n]), feature_names=np.array(candidates.FEATURE_NAMES),
                           question_ids=np.array([self.questions[0]['example_id']]),
                           message_ids=np.array(self.questions[0]['memory_ids']))
        np.savez(self.feature_path, **self.arrays)
        self.model_path = self.root / 'models/ranker55.txt'
        self.model_path.parent.mkdir()
        self.model_path.write_text(self.model_texts[55], encoding='utf-8')

    def run_selector(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return candidates.run(self.feature_path, root=self.root, **kwargs)

    def test_pipeline_loads_default_saved_model_and_overwrites_only_completed_outputs(self):
        before = {p: sha256(p) for p in (self.root / 'data').rglob('*') if p.is_file()}
        model_hash, feature_hash = sha256(self.model_path), sha256(self.feature_path)
        report = self.run_selector()
        output = self.root / 'data/label_candidates/v2'
        row = read_rows(output / 'train.jsonl')[0]
        self.assertEqual(report['counts']['candidates'], 40)
        self.assertEqual(report['counts']['references'], 2)
        self.assertEqual(report['selection_provenance'], 'single_model_training_overlap_unknown')
        self.assertFalse(report['training'])
        self.assertFalse(report['qwen_called'])
        self.assertGreaterEqual(report['runtime_seconds']['total_before_report_publish'], 0)
        self.assertGreaterEqual(report['runtime_seconds']['prediction'], 0)
        self.assertGreaterEqual(report['runtime_seconds']['selection'], 0)
        self.assertEqual(report['models'][0]['feature_count'], 55)
        self.assertEqual({r['label'] for r in row['references']}, {2})
        self.assertTrue(all(r['label'] is None for r in row['candidates']))
        self.assertIsNone(row['unselected_label'])
        all_ids = [r['message_id'] for r in row['references'] + row['candidates']] + row['unjudged_message_ids']
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(self.questions[0]['memory_ids']))
        self.assertNotIn('REFERENCE ANSWER', json.dumps(row))
        self.assertNotIn('DISTRACTOR', json.dumps(row))
        self.assertEqual(sha256(output / 'train.jsonl'), report['files_sha256']['train.jsonl'])
        first_hash = sha256(output / 'train.jsonl')
        self.run_selector(seed=272)
        self.assertNotEqual(first_hash, sha256(output / 'train.jsonl'))
        self.assertEqual(sha256(self.model_path), model_hash)
        self.assertEqual(sha256(self.feature_path), feature_hash)
        preserved = {p: sha256(p) for p in output.iterdir()}
        self.model_path.write_text(self.model_texts[54], encoding='utf-8')
        self.arrays['X'][0, 0] = np.nan
        np.savez(self.feature_path, **self.arrays)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.run_selector()
        self.assertEqual({p: sha256(p) for p in output.iterdir()}, preserved)
        for path, digest in before.items():
            self.assertEqual(sha256(path), digest)

    def test_54_feature_saved_model_accepts_the_matching_prefix_of_a_55_feature_cache(self):
        self.model_path.write_text(self.model_texts[54], encoding='utf-8')
        self.assertEqual(self.run_selector()['models'][0]['feature_count'], 54)
        np.savez(self.feature_path, **{**self.arrays, 'X': self.arrays['X'][:, :54],
                                      'feature_names': np.array(candidates.FEATURE_NAMES[:54])})
        self.assertEqual(len(self.run_selector()['feature_names']), 54)

    def test_pilot_and_all_queries_keep_selection_stable_across_cache_order(self):
        source = self.root / 'data/raw/personamem_v2/benchmark/text/train.csv'
        with source.open(encoding='utf-8', newline='') as stream:
            original = next(csv.DictReader(stream))
        with source.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(original))
            writer.writeheader()
            writer.writerows({**original, 'user_query': json.dumps({'role': 'user', 'content': f'Question {i}?'})}
                             for i in range(3))
        qids = [f'v2:train:{i:05d}' for i in range(3)]
        (self.root / 'data/splits/v2/train_ids.json').write_text(json.dumps(qids), encoding='utf-8')
        n = len(self.arrays['X'])
        arrays = {**self.arrays, 'X': np.tile(self.arrays['X'], (3, 1)), 'offsets': np.arange(4) * n,
                  'question_ids': np.array(qids), 'message_ids': np.tile(self.arrays['message_ids'], 3),
                  'y': np.full(3 * n, -99)}  # cache labels never drive selection
        np.savez(self.feature_path, **arrays)
        output = self.root / 'data/label_candidates/v2/train.jsonl'
        self.run_selector(limit_questions=1)
        pilot = read_rows(output)[0]
        self.assertEqual(self.run_selector(limit_questions=0)['counts']['candidates'], 120)
        all_rows = read_rows(output)
        self.assertEqual(pilot, next(row for row in all_rows if row['example_id'] == pilot['example_id']))
        digest = sha256(output)
        np.savez(self.feature_path, **{**arrays, 'question_ids': np.array(qids[::-1])})
        self.run_selector(limit_questions=0)
        self.assertEqual(sha256(output), digest)

    def test_existing_message_embedding_cache_supports_diversity_and_rejects_stale_text(self):
        cache_root = self.root / 'data/embeddinggemma2/v2'
        cache_root.mkdir(parents=True)
        members = [self.memories[mid] for mid in self.questions[0]['memory_ids']]
        metadata = {'memory_unit': 'single_message', 'model': candidates._recipe.EMBEDDING_RECIPE,
                    'prefix': candidates._recipe.PREFIXES['memory'], 'count': len(members)}
        (cache_root / 'memory_embedding_metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        (cache_root / 'messages.jsonl').write_text(''.join(json.dumps({**m, 'embedding_index': i}) + '\n'
                                                         for i, m in enumerate(members)), encoding='utf-8')
        vectors = np.random.default_rng(4).normal(size=(len(members), 768)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        np.save(cache_root / 'memory_embeddings.npy', vectors)
        np.save(cache_root / 'memory_done.npy', np.ones(len(members), dtype=np.bool_))
        self.assertEqual(self.run_selector(embedding_root=cache_root, diversity_weight=.1)['counts']['candidates'], 40)
        cache = candidates.MessageEmbeddings(cache_root)
        self.addCleanup(cache.vectors._mmap.close)
        self.addCleanup(cache.done._mmap.close)
        np.testing.assert_allclose(cache.for_messages(members), vectors)
        cache.rows[members[0]['message_id']]['text'] = 'Changed text'
        with self.assertRaisesRegex(ValueError, 'stale diversity'):
            cache.for_messages(members)
        metadata['model'] = {'repo': 'wrong-model', 'revision': 'wrong'}
        (cache_root / 'memory_embedding_metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'pinned single-message'):
            candidates.MessageEmbeddings(cache_root)

    def test_feature_alignment_and_active_split_are_required(self):
        wrongs = [dict(message_ids=self.arrays['message_ids'][::-1]),
                  dict(offsets=np.array([0., len(self.arrays['X'])])),
                  dict(question_ids=np.array(['v2:test:00000'])),
                  dict(feature_names=np.array(candidates.FEATURE_NAMES[::-1]))]
        for changes in wrongs:
            with self.subTest(changes=list(changes)):
                np.savez(self.feature_path, **{**self.arrays, **changes})
                with self.assertRaises(ValueError):
                    self.run_selector()
        self.assertFalse((self.root / 'data/label_candidates').exists())

    def test_saved_fold_models_require_disjoint_personas_and_matching_hashes(self):
        manifest = self.root / 'models/folds.json'
        entry = {'path': 'ranker55.txt', 'sha256': sha256(self.model_path),
                 'training_persona_ids': ['v2:99'], 'held_out_persona_ids': ['v2:7']}
        manifest.write_text(json.dumps({'models': [entry]}), encoding='utf-8')
        report = self.run_selector(fold_manifest=manifest)
        self.assertEqual(report['selection_provenance'], 'persona_disjoint_saved_models')
        for change, message in (({'training_persona_ids': ['v2:7']}, 'overlap'),
                                ({'sha256': 'bad'}, 'hash mismatch'),
                                ({'held_out_persona_ids': ['v2:8']}, 'no held-out model')):
            manifest.write_text(json.dumps({'models': [{**entry, **change}]}), encoding='utf-8')
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.run_selector(fold_manifest=manifest)
        self.assertEqual(list((self.root / 'data/label_candidates/v2').glob('*.tmp')), [])

    def test_wrong_saved_feature_schema_and_unsafe_output_are_rejected(self):
        # Keep header byte lengths intact: LightGBM's text loader uses offsets.
        self.model_path.write_text(self.model_texts[55].replace('feature_names=cosine ', 'feature_names=wrongx '), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'feature names/order'):
            self.run_selector()
        with self.assertRaisesRegex(ValueError, 'Outputs must stay'):
            self.run_selector(output_root=self.root / 'data/raw')
        with self.assertRaisesRegex(ValueError, 'embedding-root'):
            self.run_selector(diversity_weight=.1)


if __name__ == '__main__':
    unittest.main()
