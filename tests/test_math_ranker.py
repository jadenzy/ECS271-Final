import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from math_ranker_experiment import SETTINGS, background, mmr_choice, normalized_scores, persona_folds, train_adapter
from evaluate_math_ranker import pack_mmr


class MathRankerTests(unittest.TestCase):
    def test_every_persona_stays_in_one_reproducible_fold(self):
        people = np.repeat([f'v2:{i}' for i in range(20)], 3)
        folds = persona_folds(people)
        np.testing.assert_array_equal(folds, persona_folds(people))
        for p in np.unique(people):
            self.assertEqual(len(np.unique(folds[people == p])), 1)
        self.assertEqual(set(folds), set(range(5)))

    def test_mmr_selects_complementary_evidence_and_respects_availability(self):
        relevance = normalized_scores([4., 3.99, 3.8, 0.])
        redundancy = np.array([1., 1., 0., 0.])
        available = np.array([False, True, True, True])
        self.assertEqual(mmr_choice(relevance, redundancy, available, 0.), 1)
        self.assertEqual(mmr_choice(relevance, redundancy, available, .1), 2)
        np.testing.assert_array_equal(normalized_scores([2., 2.]), [0., 0.])

    def test_adapter_learns_question_message_relation_from_frozen_vectors(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Training pilot requires CUDA')
        memory = torch.zeros(2, 768, device='cuda')
        memory[0, 1], memory[1, 0] = 1., 1.
        query = torch.zeros(1, 768, device='cuda')
        query[0, 0] = 1.
        data = SimpleNamespace(offsets=np.array([0, 2]), mi=np.array([0, 1]), qi=np.array([0]),
                               X=np.array([[0.], [1.]]), types=['neutral_preferences'],
                               bundle=SimpleNamespace(read=lambda _: np.array([1, 0])))
        U, V, loss = train_adapter(data, memory, query, np.array([0]),
                                   {**SETTINGS, 'epochs': 30, 'learning_rate': .03})
        with torch.no_grad():
            scores = (memory @ query[0] + (memory @ V) @ (query @ U)[0]).cpu().numpy()
        self.assertGreater(scores[0], scores[1])
        self.assertTrue(np.isfinite(loss))
        bkg = background(memory, query, 1)
        np.testing.assert_allclose(bkg, [0., 1.])

    def test_packing_rejects_oversize_message_and_keeps_whole_complement(self):
        import json
        question = dict(dataset='v2', question='Why?', options=[dict(label='a', text='A')])
        config = dict(baseline='selector', system_prompt='Instructions', user_template='{question}\n{options}',
                      context_tokens=100)
        class Blocks:
            def messages(self, chosen, bare):
                return bare + [dict(role='user', content=json.dumps(chosen))]
        class Server:
            actual_context = 100
            verified = False
            def render_prompt(self, messages):
                cost = sum([6, 20, 4, 4][i] for i in json.loads(messages[-1]['content'])) if len(messages) == 3 else 0
                return 'prompt', 10 + cost
            def verify_selector_prompt(self, messages, rendered, bare, bare_rendered):
                self.verified = True
                assert rendered[1] - bare_rendered[1] <= 10
        server = Server()
        similarity = np.eye(4)
        similarity[0, 1] = similarity[1, 0] = 1.
        chosen, info = pack_mmr(server, question, config, Blocks(), np.array([4., 3.99, 3.8, 0.]),
                                similarity, .1, 10)
        self.assertEqual(chosen, [0, 2])
        self.assertEqual(info['packed_memory_tokens'], 10)
        self.assertEqual(info['budget_skipped_messages'], 2)
        self.assertTrue(server.verified)


if __name__ == '__main__':
    unittest.main()
