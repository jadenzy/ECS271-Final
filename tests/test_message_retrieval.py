"""Synthetic fixtures only: no real preprocessing, pretrained models or GPU."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_ranker_features as features
import message_dataset as messages
from raw_datasets import load_dataset, message_memories, read_rows, v2_history_path


def write_fixture(root):
    def save(relative, value, jsonl=False):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + ('\n' if jsonl else ''), encoding='utf-8')

    def csv_file(relative, rows):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    history = [{'role': 'system', 'content': 'SECRET PERSONA'},
               {'role': 'user', 'content': 'I prefer tea.'},
               {'role': 'assistant', 'content': 'Tea remembered.'},
               {'role': 'user', 'content': 'Forget my old preference.'},
               {'role': 'assistant', 'content': None}]
    # The link intentionally differs from the persona filename convention.
    save('data/raw/personamem_v2/data/chat_history_32k/linked_history.json', {'chat_history': history})
    v2_row = {'persona_id': '7', 'chat_history_32k_link': 'data/chat_history_32k/linked_history.json',
              'user_query': json.dumps({'role': 'user', 'content': 'tea preference?'}),
              'correct_answer': 'REFERENCE ANSWER', 'incorrect_answers': '["DISTRACTOR"]',
              'pref_type': 'preference', 'related_conversation_snippet': 'ANNOTATED SECRET'}
    for split in ('train', 'val', 'benchmark'):
        csv_file(f'data/raw/personamem_v2/benchmark/text/{split}.csv', [v2_row])
    save('data/splits/v2/train_ids.json', ['v2:train:00000'])

    v1_row = {'persona_id': '4', 'question_id': 'q0', 'shared_context_id': 'context',
              'all_options': '["REFERENCE ANSWER", "DISTRACTOR"]', 'correct_answer': 'REFERENCE ANSWER',
              'end_index_in_shared_context': '2', 'user_question_or_message': 'tea preference?',
              'question_type': 'preference'}
    # Cut off BEFORE the assistant, then include it using a negative cutoff.
    csv_file('data/raw/personamem_v1/questions_32k.csv', [v1_row, {
        **v1_row, 'question_id': 'q1', 'end_index_in_shared_context': '-2'}])
    save('data/raw/personamem_v1/shared_contexts_32k.jsonl', {'context': history}, jsonl=True)
    save('data/splits/v1/comparison_ids.json', ['v1:comparison:q0', 'v1:comparison:q1'])

    locomo_row = {'question_id': 'c1_q0', 'question': 'tea preference?', 'question_type': 'preference',
                  'choices': ['REFERENCE ANSWER', 'DISTRACTOR'], 'correct_choice_index': 0,
                  'haystack_session_ids': ['s1', 's2'], 'haystack_session_datetimes': ['Monday', 'Tuesday'],
                  'haystack_sessions': [[{'role': 'user', 'content': 'I prefer tea.'}],
                      [{'role': 'assistant', 'content': 'New session reply'},
                       {'role': 'user', 'content': 'Forget that'}, {'role': 'assistant', 'content': 'Forgotten'}]]}
    save('data/raw/locomo/locomo_mc10.json', locomo_row, jsonl=True)
    save('data/splits/locomo/comparison_ids.json', ['locomo:c1_q0'])


class SingleMessageTests(unittest.TestCase):
    def test_roles_orphans_nulls_and_boundaries(self):
        source = [{'role': 'system', 'content': 'private'},
                  {'role': 'user', 'content': 'User: First\nline'},
                  {'role': 'assistant', 'content': 'Assistant: Complete reply'},
                  {'role': 'user', 'content': 'Preference update'},
                  {'role': 'user', 'content': 'Forget this'},
                  {'role': 'assistant', 'content': None},
                  {'role': 'assistant', 'content': 'Orphan'},
                  {'role': 'user', 'content': 'Before system'},
                  {'role': 'system', 'content': 'boundary'},
                  {'role': 'assistant', 'content': 'After system'},
                  {'role': 'user', 'content': 'Session end', 'session_id': 's1'},
                  {'role': 'assistant', 'content': 'New session', 'session_id': 's2'}]
        original = json.dumps(source)
        rows = message_memories('history', source)
        self.assertEqual(rows[0]['text'], 'User: First\nline')
        self.assertEqual(rows[1]['text'], 'Assistant: Complete reply')
        self.assertEqual([row['message_index'] for row in rows], [1, 2, 3, 4, 6, 7, 9, 10, 11])
        self.assertEqual([row['source_message'] for row in rows], [source[i] for i in [1, 2, 3, 4, 6, 7, 9, 10, 11]])
        self.assertEqual(rows[3]['text'], 'Forget this')
        self.assertEqual(json.dumps(source), original)

    def test_history_link_resolution_rejects_wrong_source_and_escape(self):
        base = Path(tempfile.gettempdir()) / 'source'
        path = v2_history_path(base, 'https://example.test/resolve/main/data/chat_history_32k/linked.json?download=true')
        self.assertEqual(path, (base / 'data/chat_history_32k/linked.json').resolve())
        for link in ('raw_data/persona.json', 'data/chat_history_128k/x.json', 'data/chat_history_32k/../../bad.json'):
            with self.subTest(link=link), self.assertRaises(ValueError):
                v2_history_path(base, link)


class DirectPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_fixture(self.root)

    def load(self, dataset, split_root=None):
        splits = ['train'] if dataset == 'v2' else ['comparison']
        return messages.load_message_dataset(dataset, splits, root=self.root, split_root=split_root)

    def test_all_raw_loaders_work_without_preparation_and_without_answer_leakage(self):
        for dataset in ('v2', 'v1', 'locomo'):
            with self.subTest(dataset=dataset):
                questions, memories, sources = self.load(dataset)
                self.assertTrue(memories)
                self.assertTrue(all(m['text'] == m['source_message']['content'] for m in memories.values()))
                self.assertTrue(all(':message:' in mid for mid in memories))
                self.assertTrue(any('data/splits/' in name for name in sources))
                retrieval = json.dumps([questions, memories])
                for secret in ('REFERENCE ANSWER', 'DISTRACTOR', 'SECRET PERSONA', 'ANNOTATED SECRET'):
                    self.assertNotIn(secret, retrieval)
                self.assertTrue(all('options' not in q for q in questions))
        self.assertFalse((self.root / 'data/messages').exists())

    def test_visibility_stable_ids_and_session_provenance(self):
        questions, memories, _ = self.load('v1')
        early, later = questions
        self.assertEqual(early['memory_ids'][0], later['memory_ids'][0])
        self.assertEqual([memories[key]['message_index'] for key in early['memory_ids']], [1])
        self.assertEqual([memories[key]['message_index'] for key in later['memory_ids']], [1, 2])
        questions, memories, _ = self.load('locomo')
        self.assertEqual([memories[key]['message_index'] for key in questions[0]['memory_ids']], [0, 1, 2, 3])
        self.assertEqual(memories[questions[0]['memory_ids'][0]]['source_message']['timestamp'], 'Monday')

    def test_retrieval_does_not_require_answer_fields(self):
        files = list((self.root / 'data/raw').rglob('*.csv'))
        for path in files:
            with path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                for field in ('correct_answer', 'incorrect_answers', 'all_options'):
                    row.pop(field, None)
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        path = self.root / 'data/raw/locomo/locomo_mc10.json'
        row = json.loads(path.read_text())
        row.pop('choices')
        row.pop('correct_choice_index')
        path.write_text(json.dumps(row) + '\n')
        for dataset in ('v2', 'v1', 'locomo'):
            questions, memories, _ = self.load(dataset)
            self.assertTrue(memories)
            self.assertTrue(all('options' not in q for q in questions))

    def test_custom_split_ids_select_the_original_messages(self):
        split_root = self.root / 'chosen_splits'
        (split_root / 'v1').mkdir(parents=True)
        (split_root / 'v1/comparison_ids.json').write_text('["v1:comparison:q1"]')
        questions, memories, sources = self.load('v1', split_root)
        self.assertEqual([q['example_id'] for q in questions], ['v1:comparison:q1'])
        self.assertEqual([m['message_index'] for m in memories.values()], [1, 2])
        self.assertIn('chosen_splits/v1/comparison_ids.json', sources)

    def test_stale_intermediate_files_are_ignored(self):
        stale = self.root / 'data/messages/v2'
        stale.mkdir(parents=True)
        (stale / 'manifest.json').write_text('invalid stale manifest')
        questions, memories, _ = self.load('v2')
        self.assertEqual(len(questions), 1)
        self.assertEqual([m['text'] for m in memories.values()], ['I prefer tea.', 'Tea remembered.', 'Forget my old preference.'])
        self.assertEqual((stale / 'manifest.json').read_text(), 'invalid stale manifest')

    def test_missing_split_ids_fail_before_loading_model_weights(self):
        with patch.object(features, 'verified_model', side_effect=AssertionError('must not load')) as model, \
             patch.object(features, 'Encoder', side_effect=AssertionError('must not load')) as encoder:
            with self.assertRaises(FileNotFoundError):
                features.run(root=self.root, splits=['val'])
        model.assert_not_called()
        encoder.assert_not_called()

    def test_invalid_split_is_rejected_before_reading_or_loading_models(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            features.parse_args(['--datasets', 'v1', '--splits', 'train'])
        with self.assertRaisesRegex(ValueError, 'Invalid splits'):
            messages.load_message_dataset('v1', ['train'], root=self.root)


if __name__ == '__main__':
    unittest.main()
