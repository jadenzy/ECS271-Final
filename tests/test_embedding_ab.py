import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from embedding_ab_experiment import partition, subset_bundle, retrieval_metrics
from finetune_embedding_pilot import selected_indices
from embedding_ab_math import CachedInputs
import ranker55 as recipe
from tune_ranker55 import Bundle


class EmbeddingABTests(unittest.TestCase):
    def test_partition_is_persona_disjoint_exhaustive_and_samples_exact_a_count(self):
        qids = [f'q{i}' for i in range(80)]
        personas = np.repeat([f'p{i}' for i in range(8)], 10)
        p = partition(qids, personas, count=35)
        self.assertEqual(p, partition(qids, personas, count=35))
        self.assertFalse(set(p['a_personas']) & set(p['b_personas']))
        self.assertFalse(set(p['a_question_ids']) & set(p['b_question_ids']))
        self.assertEqual(set(p['a_question_ids'] + p['b_question_ids']), set(qids))
        self.assertEqual(len(p['fine_tune_question_ids']), 35)
        self.assertTrue(set(p['fine_tune_question_ids']) <= set(p['a_question_ids']))
        for ids, people in [('a_question_ids', 'a_personas'), ('b_question_ids', 'b_personas')]:
            self.assertEqual({personas[qids.index(q)] for q in p[ids]}, set(p[people]))

    def test_question_filter_applies_before_sampling_and_rejects_foreign_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ids.json'
            path.write_text(json.dumps(['q1', 'q3', 'q5']))
            index = selected_indices([f'q{i}' for i in range(8)], question_ids=path, pick=2)
            self.assertTrue(set(index) <= {1, 3, 5})
            path.write_text(json.dumps(['q99']))
            with self.assertRaisesRegex(ValueError, 'outside'):
                selected_indices(['q0'], question_ids=path)

    def test_subsetting_keeps_complete_groups_labels_and_original_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'all.npz', root / 'b/features.npz'
            arrays = dict(X=np.arange(6*55,dtype=np.float32).reshape(6,55),
                          y=np.array([0,1,1,0,0,1],dtype=np.int8), offsets=np.array([0,2,5,6]),
                          question_ids=np.array(['q0','q1','q2']), persona_ids=np.array(['p0','p1','p2']),
                          message_ids=np.array(['m0','m1','m2','m3','m4','m5']),
                          message_indices=np.arange(6), content_token_lengths=np.arange(6)+10,
                          reference_labels=np.array([0,2,2,0,0,2]), judged_labels=np.array([-1,2,2,-1,-1,2]),
                          feature_names=np.array([str(i) for i in range(55)]),metadata_json=np.asarray(json.dumps({})))
            np.savez(source,**arrays)
            subset_bundle(source,target,['q2','q0'])
            with Bundle(target) as b:
                np.testing.assert_array_equal(b.read('offsets'),[0,2,3])
                np.testing.assert_array_equal(b.read('X'),arrays['X'][[0,1,5]])
                np.testing.assert_array_equal(b.read('message_ids'),['m0','m1','m5'])
                np.testing.assert_array_equal(b.read('y'),[0,1,1])
                np.testing.assert_array_equal(b.read('question_ids'),['q0','q2'])

    def test_retrieval_metrics_include_every_reference_not_only_any_hit(self):
        n=20
        targets=dict(questions=[dict(example_id='q',persona_id='p',question_type='neutral_preferences')],
                     offsets=np.array([0,n]), positive=np.array([i in [0,19] for i in range(n)]),
                     complete=np.array([True]),missing=np.array([0]),ideal_counts=np.array([2]))
        metrics,rows=retrieval_metrics(-np.arange(n,dtype=float),targets)
        self.assertEqual(metrics['all_questions']['hit_at_10'],1.)
        self.assertEqual(metrics['reference_recall_at_15'],.5)
        self.assertEqual(metrics['every_visible_reference_top15'],0.)
        self.assertNotIn('incremental_hit_over_cosine',metrics['all_questions'])

    def test_math_cache_binds_subset_rows_to_original_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = np.zeros((2,768),dtype=np.float32)
            memory[0,0],memory[1,1]=1.,1.
            np.save(root/'memory_embeddings.npy',memory)
            np.save(root/'question_embeddings.npy',memory[:1])
            np.save(root/'memory_done.npy',np.ones(2,dtype=bool))
            np.save(root/'question_done.npy',np.ones(1,dtype=bool))
            (root/'messages.jsonl').write_text('\n'.join(json.dumps(dict(message_id=f'm{i}',embedding_index=i)) for i in range(2)))
            q=dict(example_id='q',embedding_index=0,question_type='neutral_preferences',memory_ids=['m0','m1'])
            (root/'questions.jsonl').write_text(json.dumps(q))
            path=root/'features.npz'
            np.savez(path,X=np.zeros((2,55),dtype=np.float32),y=np.array([1,0]),
                     offsets=np.array([0,2]),persona_ids=np.array(['b']),question_ids=np.array(['q']),
                     feature_names=np.asarray(recipe.FEATURE_NAMES),message_ids=np.array(['m0','m1']),
                     metadata_json=np.asarray(json.dumps(dict(embedding_cache_path=str(root)))))
            data=CachedInputs(path)
            np.testing.assert_array_equal(data.mi,[0,1])
            np.testing.assert_array_equal(data.qi,[0])
            data.close()
            q['memory_ids'].reverse()
            (root/'questions.jsonl').write_text(json.dumps(q))
            with self.assertRaisesRegex(ValueError,'differ'):
                CachedInputs(path)


if __name__ == '__main__':
    unittest.main()
