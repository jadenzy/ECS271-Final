"""Packing equivalence, native token boundaries, and final-prompt reuse."""
import contextlib
import io
import random
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_qwen as evaluation
import qwen_llamacpp_server as serving


class PackingTests(unittest.TestCase):
    def test_cached_blocks_match_original_formatter_for_all_datasets_and_session_gaps(self):
        for dataset in ('v1', 'v2', 'locomo'):
            question = {'dataset': dataset, 'example_id': 'example', 'question': 'Why?',
                        'options': [{'label': 'a', 'text': 'A'}]}
            members = [{'message_index': index * 3,
                        'source_message': {'role': 'user' if index % 2 else 'assistant',
                                           'content': ' café 中文\n[embedded]\n\t ' + str(index),
                                           'session_id': ('s1', 's2', 's2', 's3')[index],
                                           'timestamp': 'Tuesday'}} for index in range(4)]
            config = {'baseline': 'selector', 'system_prompt': 'Instructions',
                      'user_template': '{question}\n{options}'}
            blocks = evaluation.SelectorMessageBlocks(dataset, members)
            bare = evaluation.build_messages(question, config, [])
            for chosen in ([], [0], [3, 0], [2, 1], [0, 2, 3], [3, 2, 1, 0]):
                with self.subTest(dataset=dataset, chosen=chosen):
                    self.assertEqual(blocks.messages(chosen, bare), evaluation.build_messages(
                        question, config, evaluation.selected_history(members, chosen)))

    def server(self):
        server = serving.LlamaCppServer(Path('.'), {'baseline': 'selector', 'memory_tokens': 100,
            'context_tokens': 1000, 'initial_top_logprobs': 32, 'maximum_top_logprobs': 128})
        self.addCleanup(server.session.close)
        server.local_tokenizer = MagicMock()
        server.label_ids = {label: i for i, label in enumerate('abcd')}
        return server

    def test_scoring_reuses_prepared_full_and_bare_prompt_without_rendering_or_tokenizing(self):
        server = self.server()
        messages = [{'role': 'system', 'content': 'Instructions'}, {'role': 'user', 'content': 'History'},
                    {'role': 'user', 'content': 'Question'}]
        server.remember_rendered_prompt(messages, ('EXACT PROMPT', 90))
        server.remember_rendered_prompt([messages[0], messages[-1]], ('BARE', 20))
        response = {'completion_probabilities': [{'top_logprobs':
                    [{'id': i, 'logprob': -i - 1.0} for i in range(4)]}]}
        with patch.object(server, 'post', return_value=response) as post, \
                patch.object(server, 'question_memory', return_value={}), contextlib.redirect_stdout(io.StringIO()):
            result = server.score(messages, request_id='example', history_messages=1)
        server.local_tokenizer.render_prompt.assert_not_called()
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], '/completion')
        self.assertEqual(post.call_args.args[1]['prompt'], 'EXACT PROMPT')
        self.assertEqual(result['memory_tokens'], 70)

    def test_final_verification_rejects_text_or_count_mismatch_before_completion(self):
        for native in (('different', 90), ('exact', 91)):
            server = self.server()
            messages = [{'role': 'user', 'content': 'Question'}]
            with patch.object(server, 'render_prompt_http', return_value=native), \
                    patch.object(server, 'post') as post, self.assertRaisesRegex(ValueError, 'differs'):
                server.verify_selector_prompt(messages, ('exact', 90), messages, ('bare', 20))
            post.assert_not_called()

    def test_render_cache_is_bounded_and_keys_include_message_contents(self):
        server = self.server()
        for index in range(10):
            server.remember_rendered_prompt([{'role': 'user', 'content': str(index)}], (str(index), index))
        self.assertEqual(len(server._rendered_prompts), 4)
        self.assertEqual(server.render_prompt([{'role': 'user', 'content': '9'}]), ('9', 9))
        server.local_tokenizer.render_prompt.assert_not_called()
        server.local_tokenizer.render_prompt.return_value = ('new', 20)
        self.assertEqual(server.render_prompt([{'role': 'user', 'content': 'changed'}]), ('new', 20))


@unittest.skipUnless((serving.ROOT / 'models/llama.cpp/llama.dll').is_file(), 'Pinned native runtime not installed')
class NativeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local = serving.LocalQwenTokenizer(serving.server_settings('selector'))
        cls.addClassCleanup(cls.local.close)

    def test_cached_counts_equal_complete_native_counts_for_unicode_and_special_tokens(self):
        rng = random.Random(271)
        alphabet = ['a', 'é', '\u0301', '中', '🐈', '[', ']', '<|im_end|>', '\n', '\r',
                    '\t', ' ', '\u00a0', '\u2003', "'", '!', '0', '\x00']
        for index in range(300):
            text = 'prefix\n' + ''.join(rng.choices(alphabet, k=200))
            text += '\n[Message 1 | role: user]\n' + ''.join(rng.choices(alphabet, k=100))
            text += '\n\n[Message 2 | role: assistant]\nend'
            with self.subTest(index=index):
                self.assertEqual(self.local.token_count(text), len(self.local.tokenize(text)))

    def test_unsupported_pre_tokenizer_uses_full_native_count(self):
        with patch.object(self.local, 'cache_block_counts', False):
            text = 'Heading:\n\n[Message 1]\nHello!\n\n[Message 2]\nWorld.'
            self.assertEqual(self.local.token_count(text), len(self.local.tokenize(text)))


if __name__ == '__main__':
    unittest.main()
