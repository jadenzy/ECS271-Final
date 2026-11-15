from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reader_one_pass import ARMS, checked_inputs, choose_messages, score_one


class ReaderOnePassTests(unittest.TestCase):
    def setUp(self):
        self.selection = [dict(message_id=f'm{i}', score=float(12-i), message_index=i) for i in range(12)]
        self.judged = {f'm{i}': dict(useful=float(i), not_useful=0.) for i in range(12)}
        self.members = [dict(message_id=f'm{i}', message_index=i,
                             source_message=dict(role='user', content=f'memory-{i}')) for i in range(12)]
        self.q = dict(example_id='q', persona_id='p', history_id='h', source_end_index=12,
                      dataset='v2', question_type='recall', question='Which fact?',
                      options=[dict(label=l, text=f'option-{l}') for l in 'abcd'])

    def test_equal_count_policies_follow_their_own_scores(self):
        self.assertEqual(choose_messages(self.selection, self.judged, ARMS[0]), [f'm{i}' for i in range(12)])
        self.assertEqual(choose_messages(self.selection, self.judged, ARMS[1]), [f'm{i}' for i in range(8)])
        self.assertEqual(choose_messages(self.selection, self.judged, ARMS[2]), [f'm{i}' for i in range(11, 3, -1)])
        self.assertEqual(len(choose_messages(self.selection[:3], self.judged, ARMS[1])), 3)

    def test_single_score_call_keeps_original_options_and_chronological_memory(self):
        class Server:
            calls = 0
            def render_prompt(self, messages):
                return 'prompt', 100
            def verify_selector_prompt(self, *args):
                pass
            def score(self, messages, **kwargs):
                self.calls += 1
                self.messages = messages
                return dict(predicted_label='b', label_conditional_logprobs=dict(a=-2., b=-.1, c=-3., d=-4.),
                            prompt_tokens=100, inference_seconds=.1, memory_tokens=50, prompt_sha256='hash')
        server = Server()
        config = dict(baseline='selector', system_prompt='Instructions', user_template='{question}\n{options}', answer_labels=list('abcd'))
        row = score_one(server, self.q, config, self.members, self.selection, self.judged, ARMS[2], 'b')
        self.assertEqual(server.calls, 1)
        self.assertEqual(len(row['shifts']), 1)
        self.assertEqual(row['shifts'][0]['shift'], 0)
        self.assertTrue(row['correct'])
        self.assertEqual(row['selected_memory_count'], 8)
        self.assertEqual(server.messages[-1]['content'], 'Which fact?\na) option-a\nb) option-b\nc) option-c\nd) option-d')
        text = server.messages[1]['content']
        self.assertNotIn('memory-3', text)
        self.assertLess(text.index('memory-4'), text.index('memory-11'))

    def test_saved_inputs_reject_duplicates_foreign_messages_and_changed_scores(self):
        ranker = SimpleNamespace(pools={('h', 12): self.members}, groups={'q': 0},
                                 data=SimpleNamespace(offsets=[0, 12]), scores=np.arange(12, 0, -1, dtype=float))
        s = dict(example_id='q', selected_messages=self.selection, packed_memory_tokens=100)
        j = dict(example_id='q', messages=self.judged)
        checked_inputs([self.q], ranker, [s], [j])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            checked_inputs([self.q], ranker, [s, s], [j])
        with self.assertRaisesRegex(ValueError, 'IDs differ'):
            checked_inputs([self.q], ranker, [s], [dict(example_id='q', messages={})])
        changed = [dict(self.selection[0], score=99.), *self.selection[1:]]
        with self.assertRaisesRegex(ValueError, 'frozen ranker'):
            checked_inputs([self.q], ranker, [dict(s, selected_messages=changed)], [j])


if __name__ == '__main__':
    unittest.main()
