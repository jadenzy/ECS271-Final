"""Mixture feature bundles, read-only cache reuse and consumer compatibility."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_ranker_features as builder
import feature_contract as contract
import select_label_candidates as candidates
import evaluate_qwen as evaluator
from raw_datasets import load_dataset
import test_build_ranker_features as fixtures
from test_subtlex_mixture import prior_fixture


class MixtureBundleTests(unittest.TestCase):
    setUp = fixtures.BundleTests.setUp
    write_csv = staticmethod(fixtures.BundleTests.write_csv)
    build = fixtures.BundleTests.build

    def setup_mixture(self):
        self.output = self.root / 'data/ranker55/reused/alpha_0.5'
        self.options = dict(alpha=.5, subtlex_prior=self.prior,
                            subtlex_manifest=self.manifest, output_root=self.output,
                            embedding_cache_root=self.bundle.parent.parent)

    def mixture_build(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return builder.run(root=self.root, **{**self.options, **kwargs})

    def test_readonly_caches_no_encoder_and_four_workers_equal_serial(self):
        # Four distinct queries exercise all four process slots, sharing one history.
        csv_path = self.root / 'data/raw/personamem_v2/benchmark/text/train.csv'
        with csv_path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows.extend({**rows[0], 'user_query': json.dumps({'role': 'user', 'content': text})}
                    for text in ('Tea or coffee?', 'Any other tea idea?'))
        self.write_csv(csv_path, rows)
        (self.root / 'data/splits/v2/train_ids.json').write_text(json.dumps(
            [f'v2:train:{i:05d}' for i in range(4)]))
        self.build()
        self.setup_mixture()
        cache = self.bundle / 'embeddings'
        before = {p: (builder.sha256(p), p.stat().st_mtime_ns) for p in cache.iterdir()}
        baseline = builder.sha256(self.bundle / 'features.npz')
        with patch.object(builder, 'Encoder', side_effect=AssertionError('No encoder needed')):
            self.mixture_build(feature_workers=1)
            path = self.output / 'v2/train/features.npz'
            with np.load(path, allow_pickle=False) as data:
                expected = {key: data[key].copy() for key in data.files}
            score = (self.output / 'v2/train/bm25/scores/train.jsonl').read_bytes()
            self.mixture_build(feature_workers=4)
        with np.load(path, allow_pickle=False) as data:
            for key in expected:
                np.testing.assert_array_equal(expected[key], data[key], err_msg=key)
        self.assertEqual(score, (self.output / 'v2/train/bm25/scores/train.jsonl').read_bytes())
        self.assertEqual(before, {p: (builder.sha256(p), p.stat().st_mtime_ns) for p in cache.iterdir()})
        self.assertEqual(builder.sha256(self.bundle / 'features.npz'), baseline)
        self.assertFalse((self.output / 'v2/train/embeddings').exists())
        stats = json.loads((self.output / 'statistics.json').read_text())
        self.assertEqual(stats['feature_recipe']['alpha'], .5)
        self.assertFalse({'n', 'df', 'nq', 'qdf'} & stats.keys())

    def test_new_recipe_has_no_v2_statistics_dependency_and_keeps_dense_features(self):
        self.build()
        self.build(datasets=['v1', 'locomo'])
        self.setup_mixture()
        for file in (self.bundle.parent.parent / 'statistics.json', self.bundle / 'statistics.json'):
            file.unlink()
        self.mixture_build(datasets=['v1', 'locomo'])
        for dataset in ('v1', 'locomo'):
            with np.load(self.bundle.parent.parent / dataset / 'comparison/features.npz') as original, np.load(
                    self.output / dataset / 'comparison/features.npz') as new:
                # Only lexical-derived columns change; embeddings/role/length/position retain definitions.
                columns = list(range(7)) + [28, 29, 32, 33, 34, 37, 38, 39, 42, 43, 44, 47] + list(range(48, 55))
                np.testing.assert_array_equal(original['X'][:, columns], new['X'][:, columns])
                self.assertTrue(np.all(new['y'] == -1))

    def test_bm25_uses_local_parent_df_and_mixed_weight(self):
        self.build()
        self.setup_mixture()
        self.mixture_build()
        counts, total, lock = builder.subtlex.load_prior(self.prior, self.manifest)
        engine = builder.subtlex.Mixture(counts, total, builder.subtlex.configuration(.5, lock))
        tasks = builder.prepare_inputs(root=self.root)
        q = tasks[0][2][0]
        messages = [tasks[0][3][mid] for mid in q['memory_ids']]
        h = engine.history(q, messages)
        self.assertEqual(h.total, sum(len(builder.recipe.tokenize(m['text'])) for m in messages))
        with np.load(self.output / 'v2/train/features.npz') as data:
            n = len(messages)
            # The query fixture has a single matching term (tea).
            row = next(i for i, m in enumerate(messages) if m['text'] == 'I prefer tea.')
            bm = data['X'][row, builder.FEATURE_NAMES.index('bm25_b0')]
            weighted = data['X'][row, builder.FEATURE_NAMES.index('qweight_bm25')]
            self.assertAlmostEqual(weighted, bm * h.weight('tea', .5), places=6)
            self.assertEqual(data['X'][:n].shape, (n, 55))

    def test_stale_cache_inputs_sources_and_incomplete_masks_fail_without_writes(self):
        self.build()
        self.setup_mixture()
        metadata = self.bundle / 'embeddings/memory_embedding_metadata.json'
        original = metadata.read_bytes()
        doc = json.loads(original)
        doc['inputs_sha256'] = '0' * 64
        metadata.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'cache identity/inputs'):
            self.mixture_build()
        metadata.write_bytes(original)
        done = self.bundle / 'embeddings/memory_done.npy'
        mask = np.load(done)
        mask[0] = False
        np.save(done, mask)
        with self.assertRaisesRegex(ValueError, 'Incomplete memory'):
            self.mixture_build()
        mask[0] = True
        np.save(done, mask)
        report = self.bundle / 'report.json'
        value = json.loads(report.read_text())
        value['source_sha256'] = {}
        report.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'source hashes'):
            self.mixture_build()

    def test_model_contract_candidate_selector_and_evaluator(self):
        self.build()
        self.setup_mixture()
        self.mixture_build()
        features = self.output / 'v2/train/features.npz'
        model = self.root / 'models/subtlex/model.txt'
        builder.recipe.train_from_npz(features, model)
        with contextlib.redirect_stdout(io.StringIO()):
            selected = candidates.run(features, root=self.root, model=model, diversity_weight=.1)
        self.assertEqual(selected['counts']['candidates'], 2 * candidates.BUDGET)
        self.assertEqual(selected['diversity_cache']['root'], str((self.bundle / 'embeddings').resolve()))
        qs, contexts, _, sources = load_dataset('v2', ['train'], root=self.root, include_answers=False)
        inputs = evaluator.SelectorHistories(qs, contexts, sources, features, model, top_k=2)
        self.assertEqual(len(inputs.get(qs[0])[0]), 2)
        other_model = self.root / 'models/other_alpha.txt'
        builder.recipe.train_from_npz(self.bundle / 'features.npz', other_model)
        contract.save_model_recipe(other_model, {**contract.bundle_recipe(features), 'alpha': .75})
        with self.assertRaisesRegex(ValueError, 'Incompatible model/bundle'):
            evaluator.SelectorHistories(qs, contexts, sources, features, other_model, top_k=2)
        with self.assertRaisesRegex(ValueError, 'Incompatible model/bundle'):
            candidates.run(features, root=self.root, model=other_model)
        changed = contract.bundle_recipe(features)
        changed['alpha'] = .75
        with self.assertRaisesRegex(ValueError, 'Incompatible model/bundle'):
            candidates.SavedModels(model, None, builder.FEATURE_NAMES, feature_recipe=changed)
        changed = contract.bundle_recipe(features)
        changed['processed_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Incompatible model/bundle'):
            candidates.SavedModels(model, None, builder.FEATURE_NAMES, feature_recipe=changed)
        model.write_text(model.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Stale model'):
            candidates.SavedModels(model, None, builder.FEATURE_NAMES,
                                   feature_recipe=contract.bundle_recipe(features))
        contract.sidecar_path(model).unlink()
        with self.assertRaisesRegex(ValueError, 'requires a matching model sidecar'):
            candidates.SavedModels(model, None, builder.FEATURE_NAMES,
                                   feature_recipe=contract.bundle_recipe(features))

    def test_invalid_recipe_options_and_baseline_overwrite_are_rejected(self):
        self.setup_mixture()
        for changes, error in (({'alpha': None}, 'alpha must'),
                               ({'alpha': float('nan')}, 'alpha must'),
                               ({'output_root': self.root / 'data/raw'}, 'Outputs must stay'),
                               ({'rebuild_cache': True}, 'read-only embedding caches')):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                self.mixture_build(**changes)

    def test_default_output_reuses_its_own_embedding_cache_without_writes(self):
        self.build()
        cache = self.bundle / 'embeddings'
        before = {p: (builder.sha256(p), p.stat().st_mtime_ns) for p in cache.iterdir()}
        with patch.object(builder, 'Encoder', side_effect=AssertionError('No encoder needed')):
            with contextlib.redirect_stdout(io.StringIO()):
                builder.run(root=self.root, alpha=.75, subtlex_prior=self.prior,
                            subtlex_manifest=self.manifest, embedding_cache_root=self.root / 'data/ranker55')
        self.assertEqual(self.bundle, self.root / 'data/ranker55/v2/train')
        self.assertEqual(before, {p: (builder.sha256(p), p.stat().st_mtime_ns) for p in cache.iterdir()})
        self.assertEqual(contract.bundle_recipe(self.bundle / 'features.npz')['alpha'], .75)

    def test_old_or_missing_recipe_metadata_is_rejected_by_all_consumers(self):
        self.build()
        model = self.root / 'models/supported.txt'
        builder.recipe.train_from_npz(self.bundle / 'features.npz', model)
        with np.load(self.bundle / 'features.npz', allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        metadata = json.loads(str(arrays['metadata_json'].item()))
        qs, contexts, _, sources = load_dataset('v2', ['train'], root=self.root, include_answers=False)
        bad = self.root / 'unsupported.npz'
        for old in ({'name': 'legacy_v1'}, None):
            changed = dict(metadata)
            if old is None:
                changed.pop('feature_recipe')
            else:
                changed['feature_recipe'] = old
            np.savez(bad, **{**arrays, 'metadata_json': np.array(json.dumps(changed))})
            with self.subTest(recipe=old):
                with self.assertRaisesRegex(ValueError, 'require the subtlex_mix'):
                    builder.recipe.train_from_npz(bad, self.root / 'models/unsupported.txt')
                with self.assertRaisesRegex(ValueError, 'require the subtlex_mix'):
                    candidates.run(bad, root=self.root, model=model)
                with self.assertRaisesRegex(ValueError, 'require the subtlex_mix'):
                    evaluator.SelectorHistories(qs, contexts, sources, bad, model, top_k=2)

    def test_future_messages_answers_options_and_references_cannot_change_weights(self):
        self.setup_mixture()
        counts, total, lock = builder.subtlex.load_prior(self.prior, self.manifest)
        engine = builder.subtlex.Mixture(counts, total, builder.subtlex.configuration(.5, lock))
        # V1 fixture's second question has a negative cutoff excluding the final reply.
        qs, memories, _ = builder.load_message_dataset('v1', ['comparison'], root=self.root)
        q = qs[-1]
        messages = [memories[mid] for mid in q['memory_ids']]
        old = engine.history(q, messages).weights(['tea', 'future'], .5)
        path = self.root / 'data/raw/personamem_v1/shared_contexts_32k.jsonl'
        document = json.loads(path.read_text())
        document['context'][-1]['content'] = 'Future secret tea tea tea'
        path.write_text(json.dumps(document) + '\n')
        qs2, memories2, _ = builder.load_message_dataset('v1', ['comparison'], root=self.root)
        q2 = {**qs2[-1], 'answer': 'different', 'options': ['different'], 'reference': 'different'}
        fresh = builder.subtlex.Mixture(counts, total, builder.subtlex.configuration(.5, lock))
        new = fresh.history(q2, [memories2[mid] for mid in q2['memory_ids']]).weights(['tea', 'future'], .5)
        self.assertEqual(old, new)


if __name__ == '__main__':
    unittest.main()
