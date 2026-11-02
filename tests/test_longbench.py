"""LongBench v2 documents load as passage messages with a-d answers for the shared evaluator."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_qwen as evaluation
from message_dataset import load_message_dataset
import qwen_llamacpp_server as serving
from raw_datasets import PASSAGE_CHARACTERS, PASSAGE_WORDS, load_dataset, passages


def row(identifier, context, answer='C'):
    return {'_id': identifier, 'domain': 'Single-Document QA', 'sub_domain': 'Academic', 'difficulty': 'easy',
            'length': 'short', 'question': f'What does {identifier} report?', 'answer': answer, 'context': context,
            **{f'choice_{letter}': f'{identifier} option {letter}' for letter in 'ABCD'}}


class LongBenchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.context = '\n'.join(['Title line', '', 'A short paragraph.'] + [' '.join(['word'] * 40)] * 6)
        path = self.root / 'data/raw/longbench_v2/data.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps([row('kept', self.context), row('other', 'Not selected.')]), encoding='utf-8')
        self.split = self.root / 'data/splits/longbench/comparison_ids.json'
        self.split.parent.mkdir(parents=True)
        self.split.write_text(json.dumps(['longbench:kept']), encoding='utf-8')

    def test_passages_keep_whole_lines_within_both_limits(self):
        long_line = ' '.join(f'w{i}' for i in range(400))
        text = '\n'.join(['First line.', '', 'Second line.', long_line, 'x' * 3000, 'Last line.'])
        pieces = passages(text)
        self.assertEqual(pieces[0], 'First line.\n\nSecond line.')
        for piece in pieces:
            self.assertLessEqual(len(piece.split()), PASSAGE_WORDS)
            self.assertLessEqual(len(piece), PASSAGE_CHARACTERS)
        # Nothing is lost or reordered; only whitespace inside over-long lines changes.
        self.assertEqual(''.join(' '.join(pieces).split()), ''.join(text.split()))
        self.assertEqual(passages('\n\n  \n'), [])

    def test_loader_reads_options_answers_and_one_passage_history_per_question(self):
        questions, contexts, targets, sources = load_dataset('longbench', root=self.root)
        self.assertEqual([q['example_id'] for q in questions], ['longbench:kept'])
        q = questions[0]
        self.assertEqual((q['persona_id'], q['history_id'], q['split'], q['version']),
                         ('longbench:kept', 'longbench:kept', 'comparison', '32k'))
        self.assertEqual((q['question_type'], q['sub_domain'], q['difficulty']), ('Single-Document QA', 'Academic', 'easy'))
        self.assertEqual([o['label'] for o in q['options']], list('abcd'))
        self.assertEqual(q['options'][2]['text'], 'kept option C')
        self.assertEqual(targets, {'longbench:kept': 'c'})
        history = contexts['longbench:kept']
        self.assertEqual([m['content'] for m in history], passages(self.context))
        self.assertEqual({m['role'] for m in history}, {'user'})
        self.assertEqual(set(sources), {'data/raw/longbench_v2/data.json', 'data/splits/longbench/comparison_ids.json'})
        hidden, _, no_targets, _ = load_dataset('longbench', root=self.root, include_answers=False)
        self.assertNotIn('options', hidden[0])
        self.assertEqual(no_targets, {})
        loaded, memories, _ = load_message_dataset('longbench', root=self.root)
        self.assertEqual([memories[m]['text'] for m in loaded[0]['memory_ids']], passages(self.context))

    def test_unknown_answers_and_unmatched_ids_are_rejected(self):
        path = self.root / 'data/raw/longbench_v2/data.json'
        path.write_text(json.dumps([row('kept', self.context, answer='E')]), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'A, B, C or D'):
            load_dataset('longbench', root=self.root)
        self.split.write_text(json.dumps(['longbench:missing']), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'split IDs'):
            load_dataset('longbench', root=self.root)

    def test_evaluator_and_server_accept_longbench(self):
        args = evaluation.parse_args(['--mode', 'selector', '--dataset', 'longbench'])
        self.assertEqual(args.split, 'comparison')
        self.assertEqual(args.selector_features, evaluation.ROOT / 'data/ranker55/longbench/comparison/features.npz')
        config = evaluation.evaluation_config('longbench', 'full_history')
        self.assertIn('context passages', config['system_prompt'])
        self.assertEqual(config['answer_labels'], list('abcd'))
        self.assertEqual(serving.server_settings('full_history', dataset='longbench')['ContextTokens'], 34560)
        self.assertEqual(serving.server_settings('selector', dataset='longbench')['ContextTokens'], 3048)


if __name__ == '__main__':
    unittest.main()
