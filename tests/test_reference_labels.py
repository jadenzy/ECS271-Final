"""Exact annotation labels: per-question isolation, single-message units and missing evidence."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import label_reference_messages as labels
from unittest.mock import patch
from raw_datasets import message_memories
from raw_datasets import read_rows, sha256
from test_message_retrieval import write_fixture


class MessageLabelTests(unittest.TestCase):
    def setUp(self):
        history = [{'role': 'user', 'content': 'Tea'}, {'role': 'assistant', 'content': 'Remembered'},
                   {'role': 'user', 'content': 'Tea'}, {'role': 'assistant', 'content': 'Tea'},
                   {'role': 'user', 'content': 'Tea later'}]
        rows = message_memories('history', history)
        self.memories = {row['message_id']: row for row in rows}
        self.question = {'example_id': 'v2:val:00000', 'split': 'train', 'history_id': 'history',
                         'memory_ids': list(self.memories)}

    def test_exact_role_matches_all_occurrences_and_labels_messages_independently(self):
        row = labels.label_question(self.question, self.memories, [('user', 'Tea')])
        self.assertEqual([m['label'] for m in row['memories']], [2, 0, 2, 0, 0])
        self.assertEqual(row['schema_version'], 3)
        self.assertEqual([m['message_index'] for m in row['memories']], [0, 1, 2, 3, 4])
        self.assertEqual(row['reference_source_message_indices'], [0, 2])

    def test_reference_is_question_specific_and_assistant_only_reference_is_supported(self):
        first = labels.label_question(self.question, self.memories, [('assistant', 'Tea')])
        second = labels.label_question(self.question, self.memories, [('assistant', 'Remembered')])
        self.assertEqual([m['label'] for m in first['memories']], [0, 0, 0, 2, 0])
        self.assertEqual([m['label'] for m in second['memories']], [0, 2, 0, 0, 0])
        self.assertEqual(first['memories'][3]['role'], 'assistant')

    def test_missing_substring_or_case_changed_reference_raises_instead_of_false_negatives(self):
        for content in ('missing', 'Tea lat', 'tea'):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, 'absent from visible memories'):
                labels.label_question(self.question, self.memories, [('user', content)])

    def test_invalid_or_empty_reference_lists_are_rejected(self):
        for value in ([], {}, [{'role': 'system', 'content': 'Tea'}],
                      [{'role': 'user', 'content': None}], [{'role': 'user', 'content': ''}]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                labels.reference_keys(value, 'question')


class LabelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_fixture(self.root)

    def csv_reference(self, split, snippet, *, empty_history_turn=False):
        path = self.root / f'data/raw/personamem_v2/benchmark/text/{split}.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['related_conversation_snippet'] = json.dumps(snippet)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if empty_history_turn:
            history_path = self.root / 'data/raw/personamem_v2/data/chat_history_32k/linked_history.json'
            history = json.loads(history_path.read_text())
            history['chat_history'].append({'role': 'assistant', 'content': ''})
            history_path.write_text(json.dumps(history), encoding='utf-8')

    def run_labels(self, splits=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return labels.run(splits, root=self.root)

    def test_transferred_validation_source_id_is_labeled_using_current_training_membership(self):
        self.csv_reference('train', [{'role': 'assistant', 'content': 'Tea remembered.'}])
        self.csv_reference('val', [{'role': 'user', 'content': 'I prefer tea.'}])
        (self.root / 'data/splits/v2/train_ids.json').write_text(json.dumps(['v2:val:00000']))
        before = {path: sha256(path) for path in (self.root / 'data/raw').rglob('*') if path.is_file()}
        report = self.run_labels()
        rows = read_rows(self.root / 'data/reference_labels/v2/train.jsonl')
        self.assertEqual(rows[0]['example_id'], 'v2:val:00000')
        self.assertEqual(rows[0]['split'], 'train')
        self.assertEqual([m['label'] for m in rows[0]['memories']], [2, 0, 0])
        self.assertEqual(report['schema_version'], 3)
        self.assertEqual(rows[0]['memories'][0]['message_index'], 1)
        self.assertEqual(report['splits']['train']['positive_source_messages'], 1)
        self.assertEqual(report['splits']['train']['negative_source_messages'], 2)
        self.assertNotIn('REFERENCE ANSWER', json.dumps(rows))
        self.assertNotIn('DISTRACTOR', json.dumps(rows))
        for path, digest in before.items():
            self.assertEqual(sha256(path), digest)
        self.assertFalse((self.root / 'data/messages').exists())

    def test_empty_reference_turn_must_exist_in_raw_history_and_has_no_memory_label(self):
        snippet = [{'role': 'user', 'content': 'I prefer tea.'}, {'role': 'assistant', 'content': ''}]
        self.csv_reference('train', snippet, empty_history_turn=True)
        report = self.run_labels()
        self.assertEqual(report['splits']['train']['empty_reference_messages'], 1)
        self.assertEqual(report['splits']['train']['positive_source_messages'], 1)

    def test_missing_empty_reference_turn_is_rejected(self):
        self.csv_reference('train', [{'role': 'user', 'content': 'I prefer tea.'},
                                     {'role': 'assistant', 'content': ''}])
        with self.assertRaisesRegex(ValueError, 'empty reference turn is absent'):
            self.run_labels()

    def test_missing_reference_does_not_replace_previous_labels_or_publish_partial_splits(self):
        self.csv_reference('train', [{'role': 'user', 'content': 'I prefer tea.'}])
        self.csv_reference('val', [{'role': 'user', 'content': 'Missing annotation'}])
        (self.root / 'data/splits/v2/val_ids.json').write_text(json.dumps(['v2:val:00000']))
        self.run_labels(['train'])
        output = self.root / 'data/reference_labels/v2'
        before = sha256(output / 'train.jsonl')
        with self.assertRaisesRegex(ValueError, 'absent from visible memories'):
            self.run_labels(['train', 'val'])
        self.assertEqual(sha256(output / 'train.jsonl'), before)
        self.assertFalse((output / 'val.jsonl').exists())
        self.assertEqual(list(output.glob('*.tmp')), [])

    def test_raw_source_or_split_changes_during_loading_are_rejected(self):
        self.csv_reference('train', [{'role': 'user', 'content': 'I prefer tea.'}])
        source = self.root / 'data/raw/personamem_v2/benchmark/text/train.csv'
        original = source.read_bytes()
        load = labels.load_message_dataset

        def change_source(*args, **kwargs):
            loaded = load(*args, **kwargs)
            source.write_bytes(original + b'\n')
            return loaded

        with patch.object(labels, 'load_message_dataset', side_effect=change_source):
            with self.assertRaisesRegex(ValueError, 'Changed raw source'):
                self.run_labels()
        source.write_bytes(original)

        def change_split(*args, **kwargs):
            loaded = load(*args, **kwargs)
            (self.root / 'data/splits/v2/train_ids.json').write_text('[]')
            return loaded

        with patch.object(labels, 'load_message_dataset', side_effect=change_split):
            with self.assertRaisesRegex(ValueError, 'active split IDs'):
                self.run_labels()

    def test_unsafe_output_path_is_rejected_before_any_files_are_written(self):
        with self.assertRaisesRegex(ValueError, 'under data/reference_labels'):
            labels.run(root=self.root, output_root=self.root / 'data/messages')


if __name__ == '__main__':
    unittest.main()
