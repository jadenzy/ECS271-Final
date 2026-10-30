"""No-memory input isolation, shared scoring, outputs and interrupted-run resume."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_qwen as evaluation
import qwen_llamacpp_server as serving


def question(index=0):
    return {'example_id': f'v2:test:{index}', 'dataset': 'v2', 'split': 'test',
            'version': '32k', 'persona_id': 'v2:7', 'history_id': 'history',
            'question_type': 'preference', 'question': 'What do I prefer?',
            'options': [{'label': label, 'text': label.upper()} for label in 'abcd']}


class NoMemoryTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((evaluation.ROOT / 'configs/evaluate_qwen.json').read_text())
        self.config.update(baseline='no_memory', context_tokens=4096)

    def test_prompt_has_only_shared_instructions_question_and_choices(self):
        q = {**question(), 'persona_profile': 'SECRET PROFILE', 'correct_label': 'SECRET GOLD'}
        messages = evaluation.build_messages(q, self.config, [])
        self.assertEqual(len(messages), 2)
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])
        self.assertNotIn('SECRET', json.dumps(messages))
        for option in q['options']:
            self.assertIn(f'{option["label"]}) {option["text"]}', messages[-1]['content'])
        with self.assertRaisesRegex(ValueError, 'No-memory'):
            evaluation.build_messages(q, self.config, [{'role': 'user', 'content': 'SECRET HISTORY'}])

    def test_raw_label_scoring_needs_no_history_or_packing(self):
        server = serving.LlamaCppServer(Path('.'), self.config)
        self.addCleanup(server.session.close)
        server.label_ids = dict(zip('abcd', range(4)))
        response = {'completion_probabilities': [{'top_logprobs': [
            {'id': i, 'logprob': value} for i, value in enumerate((-3., -1., -4., -2.))]}]}
        with patch.object(server, 'render_prompt', return_value=('question-only prompt', 70)) as render, \
             patch.object(server, 'post', return_value=response) as post, \
             patch.object(server, 'question_memory', return_value={'mean_gpu_memory_mib': 1200}), \
             patch.object(evaluation, 'load_dataset', side_effect=AssertionError('No raw-history loading')), \
             contextlib.redirect_stdout(io.StringIO()):
            result = evaluation.score_question(server, question(), [], {'history_messages': 0}, server.config)
        render.assert_called_once()
        self.assertEqual(len(render.call_args.args[0]), 2)
        post.assert_called_once_with('/completion', {'prompt': 'question-only prompt', 'n_predict': 1,
                                                    'n_probs': 32, 'post_sampling_probs': False})
        self.assertEqual(result['predicted_label'], 'b')
        self.assertEqual(result['memory_tokens'], 0)
        self.assertEqual(result['history_input_tokens'], 0)
        self.assertEqual(result['mean_gpu_memory_mib'], 1200)

    def test_cli_is_v2_only_without_retrieval_or_server_settings(self):
        args = evaluation.parse_args([], fixed_mode='no_memory')
        self.assertEqual((args.mode, args.dataset, args.split), ('no_memory', 'v2', 'test'))
        for argv in (['--dataset', 'v1'], ['--memory-tokens', '0'], ['--lr-model', 'model'],
                     ['--scores-dir', 'features'], ['--context-tokens', '4096']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.parse_args(argv, fixed_mode='no_memory')




if __name__ == '__main__':
    unittest.main()
