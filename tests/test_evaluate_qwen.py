"""Evaluation checks that never load Qwen weights or launch the model server."""
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_qwen as evaluation
import qwen_llamacpp_server as serving


def question():
    return {'example_id': 'v2:test:0', 'dataset': 'v2', 'split': 'test', 'version': '32k', 'persona_id': 'v2:7', 'history_id': 'history', 'question': 'What do I prefer?', 'options': [{'label': label, 'text': label.upper()} for label in 'abcd']}


class PredictionRecoveryTests(unittest.TestCase):
    def test_incomplete_final_json_or_utf8_is_preserved_and_retried(self):
        complete = '{"example_id":"saved","text":"caf\u00e9"}\n'.encode('utf-8')
        for tail in (b'{"example_id":"unfinished', b'{"example_id":"unfinished","text":"\xe2\x82'):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'predictions.jsonl'
                path.write_bytes(complete + tail)
                with contextlib.redirect_stdout(io.StringIO()):
                    rows = evaluation.read_predictions(path)
                self.assertEqual(rows, [{'example_id': 'saved', 'text': 'caf\u00e9'}])
                self.assertEqual(path.read_bytes(), complete)
                backups = list(path.parent.glob('predictions.jsonl.incomplete-*.bak'))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_bytes(), tail)
                self.assertEqual(evaluation.read_predictions(path), rows)
                self.assertEqual(list(path.parent.glob('*.bak')), backups)

    def test_complete_last_record_without_newline_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.jsonl'
            original = b'{"example_id":"saved"}'
            path.write_bytes(original)
            self.assertEqual(evaluation.read_predictions(path), [{'example_id': 'saved'}])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_corrupt_completed_line_is_rejected_without_changing_the_file(self):
        for content in (b'{broken}\n', b'{broken}\n{"example_id":"later"}\n', b'[]'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'predictions.jsonl'
                path.write_bytes(content)
                with self.assertRaises(ValueError):
                    evaluation.read_predictions(path)
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(list(path.parent.iterdir()), [path])

    def test_no_complete_predictions_can_recover_and_retry_the_first_question(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.jsonl'
            self.assertEqual(evaluation.read_predictions(path), [])
            path.write_bytes(b'{"example_id":')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluation.read_predictions(path), [])
            self.assertEqual(path.read_bytes(), b'')


class FullHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = evaluation.runtime_metadata()
        cls.runtime_config = json.loads((evaluation.ROOT / 'configs/qwen_server.json').read_text())

    def setUp(self):
        self.config = json.loads((evaluation.ROOT / 'configs/evaluate_qwen.json').read_text())
        self.config['baseline'] = 'full_history'
        self.config['dataset_version'] = '32k'
        self.config['dataset'] = 'v2'
        self.config['dataset_name'] = 'PersonaMem-v2'
        self.config['server_settings'] = copy.deepcopy(self.settings)
        self.history = [{'role': 'system', 'content': 'SOURCE PERSONA'}, {'role': 'user', 'content': 'I used to prefer coffee.'}, {'role': 'assistant', 'content': 'Understood.'}, {'role': 'user', 'content': 'FUTURE MESSAGE MUST NOT APPEAR'}]

    def test_native_command_exports_gguf_backend_without_loading_it(self):
        for dataset, mode, context in (('v2', 'full_history', self.runtime_config['v2_context_tokens']),
                                       ('v1', 'full_history', self.runtime_config['v1_context_tokens']),
                                       ('locomo', 'full_history', self.runtime_config['locomo_context_tokens']), ('locomo', 'no_memory', 4096),
                                       ('v2', 'no_memory', 4096)):
            with self.subTest(dataset=dataset, mode=mode):
                metadata = evaluation.runtime_metadata(dataset, mode)
                command = metadata['Command']
                self.assertEqual(metadata['ContextTokens'], context)
                self.assertEqual(command[command.index('--ctx-size') + 1], str(context))
                self.assertEqual(metadata['Model'], 'Qwen3.5-4B-Q4_K_M')
                self.assertEqual(metadata['Dtype'], 'Q4_K_M')
                self.assertFalse(metadata['CpuOffload'])
                self.assertFalse(metadata['Thinking'])
                self.assertTrue(command[0].endswith('llama-server.exe'))
                self.assertEqual(metadata['Backend'], 'llama_cpp')
                self.assertTrue(all(isinstance(arg, str) for arg in command))
                self.assertEqual(command[command.index('--n-gpu-layers') + 1], '-1')
                self.assertEqual(command[command.index('--flash-attn') + 1], 'on')
                self.assertIn('--cache-prompt', command)
                self.assertIn('--no-context-shift', command)

    def test_shared_json_context_is_used_by_direct_server_and_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            config = {**self.runtime_config, 'v2_context_tokens': 2048}
            (root / 'configs/qwen_server.json').write_text(json.dumps(config))
            with patch.object(evaluation, 'ROOT', root):
                settings = evaluation.runtime_metadata()
            command = settings['Command']
            self.assertEqual(command[command.index('--ctx-size') + 1], '2048')
            self.assertFalse(Path(settings['ModelPath']).exists())

    def test_cli_cannot_override_server_context_or_sampling(self):
        for options in (['--mode', 'lr', '--context-tokens', '4096'], ['--temperature', '0.5'], ['--gpu-layers', '0']):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.parse_args(options)


    def test_full_history_keeps_all_dialogue_and_has_one_initial_system(self):
        visible = self.history[:-1]
        messages = evaluation.build_messages(question(), self.config, visible)
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'user'])
        self.assertTrue(all(message['content'] in messages[1]['content'] for message in visible))
        self.assertEqual(visible, self.history[:-1])
        self.assertIn(question()['question'], messages[-1]['content'])
        for option in question()['options']:
            self.assertIn(f'{option["label"]}) {option["text"]}', messages[-1]['content'])

    def test_missing_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'requires the visible source history'):
            evaluation.build_messages(question(), self.config, None)

    def test_repeated_system_messages_are_preserved_in_place(self):
        visible = self.history[:-1] + [{'role': 'system', 'content': 'SECOND SESSION PERSONA'}, {'role': 'user', 'content': 'Now I prefer tea.'}]
        messages = evaluation.build_messages(question(), self.config, visible)
        transcript = messages[1]['content']
        self.assertIn('[Message 3 | role: system]\nSECOND SESSION PERSONA', transcript)
        self.assertLess(transcript.index('Understood.'), transcript.index('SECOND SESSION PERSONA'))
        self.assertLess(transcript.index('SECOND SESSION PERSONA'), transcript.index('Now I prefer tea.'))


    def test_split_switches_are_removed(self):
        for flag in ('--split', '-s'):
            for split in ('train', 'val', 'test', 'development'):
                with self.subTest(flag=flag, split=split), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    evaluation.parse_args([flag, split])
                self.assertEqual(error.exception.code, 2)

    def test_evaluation_uses_test_or_full_comparison_scope(self):
        for dataset in ('v2', 'v1', 'locomo'):
            with self.subTest(dataset=dataset):
                args = evaluation.parse_args(['--dataset', dataset])
                self.assertEqual(args.split, 'test' if dataset == 'v2' else 'comparison')
                self.assertFalse(hasattr(args, 'subset'))
                for flag in ('--subset', '-S'):
                    for subset in ('compact', 'full'):
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                            evaluation.parse_args(['--dataset', dataset, flag, subset])

    def test_resume_obeys_fixed_test_scope(self):
        for dataset, split, subset, supported in (
            ('v2', 'test', 'compact', True), ('v2', 'train', 'compact', False),
            ('v2', 'val', 'full', False), ('v1', 'comparison', 'full', True),
            ('locomo', 'comparison', 'full', True), ('v1', 'comparison', 'compact', False),
            ('locomo', 'comparison', 'compact', False),
        ):
            protocol = {'config': {'dataset': dataset, 'subset': subset, 'baseline': 'full_history'}, 'split': split}
            with self.subTest(dataset=dataset, split=split, subset=subset):
                self.assertEqual(evaluation.supported_test_selection(protocol), supported)
                if supported:
                    argv = evaluation.resume_arguments(Path('results/run/protocol.json'), protocol)
                    self.assertNotIn('--split', argv)
                    self.assertNotIn('--subset', argv)
                    args = evaluation.parse_args(argv)
                    self.assertEqual((args.dataset, args.split), (dataset, split))
                else:
                    with self.assertRaisesRegex(ValueError, 'Only V2 test'):
                        evaluation.resume_arguments(Path('results/run/protocol.json'), protocol)

    def test_unknown_mode_argument_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            evaluation.parse_args(['--mode', 'unknown'])
        self.assertEqual(error.exception.code, 2)


    def test_evaluator_starts_and_stops_exported_native_process(self):
        process = MagicMock()
        process.poll.return_value = None
        connection = MagicMock()
        connection.status_code = 200
        context = self.settings['ContextTokens'] or 32768
        reported_context = 2048
        connection.json.return_value = {'default_generation_settings': {'n_ctx': reported_context}}
        socket = MagicMock()
        port = self.settings['Port'] or 43210
        socket.__enter__.return_value.getsockname.return_value = ('127.0.0.1', port)
        command = self.settings['Command']
        with tempfile.TemporaryDirectory() as directory:
            config = {**self.config, 'context_tokens': context}
            server = serving.LlamaCppServer(Path(directory), config)
            server.session = MagicMock()
            server.session.get.return_value = connection
            with patch.object(serving, 'server_settings', return_value=self.settings) as launcher, patch.object(serving.socket, 'socket', return_value=socket), patch.object(serving.threading, 'Thread'), patch.object(serving.subprocess, 'Popen', return_value=process) as start, patch.object(server, 'post', return_value={'tokens': [64]}):
                with server:
                    self.assertEqual(server.actual_context, reported_context)
                    self.assertEqual(server.config['context_tokens'], context)
                self.assertEqual(start.call_args.args[0], command)
                self.assertEqual(start.call_args.kwargs['env']['GGML_CUDA_DISABLE_GRAPHS'], '1')
                launcher.assert_not_called()
            self.assertIsNone(server.hardware['gpu_offload_confirmed'])
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {'qwen-server.log'})
        process.terminate.assert_called_once()
        process.wait.assert_called_once()

    def test_question_memory_excludes_startup_and_other_questions(self):
        server = serving.LlamaCppServer(Path('.'), self.config)
        self.addCleanup(server.session.close)
        server.memory_samples = [
            {'sampled_at': 1, 'gpu_used_mib': 9000, 'ram_working_set_mib': 9000},
            {'sampled_at': 10, 'gpu_used_mib': 3000, 'ram_working_set_mib': 1000},
            {'sampled_at': 12, 'gpu_used_mib': 5000, 'ram_working_set_mib': 3000},
            {'sampled_at': 14, 'gpu_used_mib': 9000, 'ram_working_set_mib': 9000},
        ]
        memory = server.question_memory(10, 13)
        self.assertEqual(memory['mean_gpu_memory_mib'], 4000)
        self.assertEqual(memory['mean_ram_working_set_mib'], 2000)
        self.assertEqual(memory['peak_gpu_memory_mib'], 5000)
        self.assertEqual(memory['gpu_memory_samples'], 2)
        self.assertEqual(memory['memory_sampling'], 'during_request')

    def test_restart_replaces_owned_process_and_preserves_log_and_samples(self):
        first_process, second_process = MagicMock(), MagicMock()
        first_process.poll.return_value = second_process.poll.return_value = None
        first_session, second_session = MagicMock(), MagicMock()
        connection = MagicMock()
        connection.status_code = 200
        context = self.settings['ContextTokens']
        connection.json.return_value = {'default_generation_settings': {'n_ctx': context}}
        first_session.get.return_value = second_session.get.return_value = connection
        socket = MagicMock()
        socket.__enter__.return_value.getsockname.return_value = ('127.0.0.1', self.settings['Port'])
        first_sampler, second_sampler = MagicMock(), MagicMock()
        with tempfile.TemporaryDirectory() as directory, patch.object(serving.requests, 'Session', side_effect=[first_session, second_session]), patch.object(serving, 'server_settings', return_value=self.settings), patch.object(serving.socket, 'socket', return_value=socket), patch.object(serving.threading, 'Thread', side_effect=[first_sampler, second_sampler]), patch.object(serving.subprocess, 'Popen', side_effect=[first_process, second_process]) as start:
            server = serving.LlamaCppServer(Path(directory), {**self.config, 'context_tokens': context})
            with patch.object(server, 'post', return_value={'tokens': [64]}), server:
                initial_event = server.stop_sampling
                sample = {'sampled_at': 1, 'gpu_used_mib': 1234, 'gpu_total_mib': 6144}
                server.memory_samples.append(sample)
                server.log.write('INVALID SCORE EVIDENCE\n')
                server.log.flush()
                server.restart()
                self.assertIs(server.process, second_process)
                self.assertIs(server.session, second_session)
                self.assertFalse(server.session.trust_env)
                self.assertTrue(initial_event.is_set())
                self.assertIsNot(server.stop_sampling, initial_event)
                self.assertFalse(server.stop_sampling.is_set())
                self.assertEqual(server.memory_samples, [sample])
                self.assertEqual(start.call_args_list[0].args, start.call_args_list[1].args)
                for key in ('cwd', 'env', 'stderr', 'creationflags'):
                    self.assertEqual(start.call_args_list[0].kwargs[key], start.call_args_list[1].kwargs[key])
                self.assertEqual(server.server_restart_count, 1)
            log = (Path(directory) / 'qwen-server.log').read_text(encoding='utf-8')
            self.assertIn('INVALID SCORE EVIDENCE', log)
            self.assertIn('Recovery server restart 1', log)
            self.assertEqual(log.count('Command:'), 2)
            self.assertEqual(server.hardware['server_restart_count'], 1)
            self.assertEqual(server.hardware['gpu_memory_samples'], 1)
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {'qwen-server.log'})
        for process in (first_process, second_process):
            process.terminate.assert_called_once()
            process.wait.assert_called_once()
        first_session.close.assert_called_once()
        second_session.close.assert_called_once()
        first_sampler.join.assert_called_once()
        second_sampler.join.assert_called_once()

    def test_connection_reset_reports_native_cuda_crash_and_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'qwen-server.log').write_text('prompt processing\nCUDA error: an illegal instruction was encountered\n')
            server = serving.LlamaCppServer(out, self.config)
            self.addCleanup(server.session.close)
            server.base = 'http://127.0.0.1:8080'
            server.process = MagicMock()
            server.process.poll.return_value = 1
            with patch.object(server.session, 'post', side_effect=serving.requests.ConnectionError('WinError 10054')):
                with self.assertRaisesRegex(RuntimeError, 'exited with code 1 during /tokenize.*CUDA error: an illegal instruction'):
                    server.post('/tokenize', {'content': 'test'})

    def test_short_question_gets_snapshot_and_missing_readings_stay_null(self):
        server = serving.LlamaCppServer(Path('.'), self.config)
        self.addCleanup(server.session.close)
        server.memory_samples = [{'sampled_at': 1, 'gpu_used_mib': 9000}]
        with patch.object(server, 'sample_memory', return_value={'gpu_used_mib': 0, 'ram_working_set_mib': 1000}) as sample:
            memory = server.question_memory(10, 10.1)
        sample.assert_called_once()
        self.assertEqual(memory['mean_gpu_memory_mib'], 0)
        self.assertEqual(memory['mean_ram_working_set_mib'], 1000)
        self.assertEqual(memory['memory_sampling'], 'post_request_snapshot')
        missing = serving.memory_statistics([{}])
        self.assertIsNone(missing['mean_gpu_memory_mib'])
        self.assertIsNone(missing['mean_ram_working_set_mib'])
        self.assertEqual(missing['gpu_memory_samples'], 0)

    def test_scoring_records_memory_and_runtime_without_model_startup(self):
        server = serving.LlamaCppServer(Path('.'), {**self.config, 'context_tokens': 32768})
        self.addCleanup(server.session.close)
        server.label_ids = {label: index for index, label in enumerate('abcd')}
        memory = serving.memory_statistics([{'gpu_used_mib': 4000, 'ram_working_set_mib': 2000}])
        responses = [{'prompt': '<|im_start|>assistant\n'}, {'tokens': [1, 2]}, {'prompt': 'question only'}, {'tokens': [1]}, {'completion_probabilities': [{'top_logprobs': [{'id': index, 'logprob': -index - 1.0} for index in range(4)]}]}]
        with patch.object(server, 'post', side_effect=responses) as post, patch.object(server, 'question_memory', return_value=memory) as measure, patch.object(serving.time, 'perf_counter', side_effect=[10, 11, 12, 15]), contextlib.redirect_stdout(io.StringIO()):
            prediction = evaluation.score_question(server, question(), self.history[:-1], {'history_messages': 3}, server.config)
        completion = post.call_args_list[-1]
        self.assertEqual(completion.args[0], '/completion')
        self.assertNotIn('grammar', completion.args[1])
        self.assertNotIn('ignore_eos', completion.args[1])
        self.assertFalse(completion.args[1]['post_sampling_probs'])
        for name in ('temperature', 'top_k', 'top_p', 'min_p', 'presence_penalty', 'repeat_penalty', 'cache_prompt'):
            self.assertNotIn(name, completion.args[1])
        self.assertEqual(post.call_args_list[0].args[1], {'messages': evaluation.build_messages(question(), self.config, self.history[:-1])})
        measure.assert_called_once_with(10, 15)
        self.assertEqual(prediction['elapsed_seconds'], 5)
        self.assertEqual(prediction['inference_seconds'], 1)
        self.assertEqual(prediction['history_input_tokens'], 1)
        self.assertEqual(prediction['history_messages'], 3)
        self.assertEqual(prediction['mean_gpu_memory_mib'], 4000)
        self.assertEqual(prediction['mean_ram_working_set_mib'], 2000)

    def test_selector_budget_is_enforced_before_any_completion_request(self):
        server = serving.LlamaCppServer(Path('.'), {**self.config, 'baseline': 'selector',
                                                   'memory_tokens': 10, 'context_tokens': 32768})
        self.addCleanup(server.session.close)
        messages = [{'role': 'system', 'content': 'Instructions'}, {'role': 'user', 'content': 'History'},
                    {'role': 'user', 'content': 'Question/options'}]
        with patch.object(server, 'render_prompt', side_effect=[('full', 120), ('bare', 100)]), \
                patch.object(server, 'post') as post, self.assertRaisesRegex(ValueError, 'memory budget is 10'):
            server.score(messages, request_id='budget-test', history_messages=1)
        post.assert_not_called()

    def test_selector_respects_requested_context_when_server_reports_a_larger_capacity(self):
        server = serving.LlamaCppServer(Path('.'), {**self.config, 'baseline': 'selector',
                                                   'memory_tokens': 2048, 'context_tokens': 3048})
        self.addCleanup(server.session.close)
        server.actual_context = 4096
        messages = [{'role': 'system', 'content': 'Instructions'}, {'role': 'user', 'content': 'History'},
                    {'role': 'user', 'content': 'Question/options'}]
        with patch.object(server, 'render_prompt', return_value=('full', 3048)), \
                patch.object(server, 'post') as post, self.assertRaisesRegex(ValueError, 'context is 3048'):
            server.score(messages, request_id='context-test', history_messages=1)
        post.assert_not_called()

    def test_sampled_slash_is_ignored_and_only_raw_answer_labels_are_scored(self):
        server = serving.LlamaCppServer(Path('.'), {**self.config, 'context_tokens': 32768})
        self.addCleanup(server.session.close)
        server.label_ids = {label: index for index, label in enumerate('abcd')}
        raw = {0: -3.0, 1: -1.0, 2: -4.0, 3: -2.0, 14: -0.01}
        responses = [{'prompt': '<|im_start|>assistant\n'}, {'tokens': [1, 2]}, {'prompt': 'question only'}, {'tokens': [1]}, {'content': '/', 'completion_probabilities': [{'top_logprobs': [{'id': index, 'logprob': value} for index, value in raw.items()]}]}]
        with patch.object(server, 'post', side_effect=responses) as post, patch.object(server, 'question_memory', return_value={}), contextlib.redirect_stdout(io.StringIO()):
            prediction = evaluation.score_question(server, question(), self.history[:-1], {}, server.config)
        request = post.call_args_list[-1].args[1]
        self.assertNotIn('grammar', request)
        self.assertNotIn('ignore_eos', request)
        self.assertFalse(request['post_sampling_probs'])
        self.assertEqual(prediction['predicted_label'], 'b')
        self.assertEqual(prediction['label_raw_logprobs'], {'a': -3.0, 'b': -1.0, 'c': -4.0, 'd': -2.0})
        self.assertEqual(set(prediction['label_conditional_logprobs']), set('abcd'))
        self.assertAlmostEqual(sum(math.exp(p) for p in prediction['label_conditional_logprobs'].values()), 1.0)

    def test_unconstrained_scoring_expands_top_probs_until_all_labels_are_present(self):
        server = serving.LlamaCppServer(Path('.'), {**self.config, 'context_tokens': 32768})
        self.addCleanup(server.session.close)
        server.label_ids = {label: index for index, label in enumerate('abcd')}
        def result(ids):
            return {'content': '/', 'completion_probabilities': [{'top_logprobs': [{'id': index, 'logprob': -index - 1.0} for index in ids]}]}
        responses = [{'prompt': '<|im_start|>assistant\n'}, {'tokens': [1, 2]}, {'prompt': 'question only'}, {'tokens': [1]}, result([0, 1, 2, 14]), result([0, 1, 2, 3, 14])]
        with patch.object(server, 'post', side_effect=responses) as post, patch.object(server, 'question_memory', return_value={}), contextlib.redirect_stdout(io.StringIO()):
            prediction = evaluation.score_question(server, question(), self.history[:-1], {}, server.config)
        requests = [call.args[1] for call in post.call_args_list if call.args[0] == '/completion']
        self.assertEqual([request['n_probs'] for request in requests], [32, 128])
        self.assertTrue(all('grammar' not in request and not request['post_sampling_probs'] for request in requests))
        self.assertEqual(prediction['logprob_attempts'], 2)
        self.assertEqual(prediction['predicted_label'], 'a')

    def test_invalid_label_logprobs_retry_without_changing_server_settings(self):
        for invalid in (None, float('nan'), float('inf'), float('-inf'), 'invalid', False):
            with self.subTest(invalid=invalid):
                server = serving.LlamaCppServer(Path('.'), {**self.config, 'context_tokens': 32768})
                self.addCleanup(server.session.close)
                server.config = copy.deepcopy(server.config)
                server.label_ids = {label: index for index, label in enumerate('abcd')}
                def result(values):
                    return {'completion_probabilities': [{'top_logprobs': [{'id': index, 'logprob': value} for index, value in enumerate(values)]}]}
                responses = [{'prompt': '<|im_start|>assistant\n'}, {'tokens': [1, 2]}, {'prompt': 'question only'}, {'tokens': [1]}, result([invalid, -1.0, -4.0, -2.0]), result([-3.0, -1.0, -4.0, -2.0])]
                with patch.object(server, 'post', side_effect=responses) as post, patch.object(server, 'question_memory', return_value={}), patch.object(serving.time, 'perf_counter', side_effect=[10, 11, 12, 13, 15, 16]), contextlib.redirect_stdout(io.StringIO()) as output:
                    prediction = evaluation.score_question(server, question(), self.history[:-1], {}, server.config)
                requests = [call.args[1] for call in post.call_args_list if call.args[0] == '/completion']
                self.assertEqual(requests[0], requests[1])
                self.assertTrue(all('cache_prompt' not in request for request in requests))
                self.assertIn('retrying the full prompt with server defaults (1/2)', output.getvalue())
                self.assertEqual(prediction['label_raw_logprobs'], {'a': -3.0, 'b': -1.0, 'c': -4.0, 'd': -2.0})
                self.assertEqual(prediction['predicted_label'], 'b')
                self.assertEqual(prediction['logprob_attempts'], 2)
                self.assertEqual(prediction['inference_seconds'], 3)
                self.assertEqual(prediction['elapsed_seconds'], 6)
                json.dumps(prediction, allow_nan=False)

    def test_persistent_null_logprobs_stop_after_one_restart_and_bounded_retries(self):
        server = serving.LlamaCppServer(Path('.'), {**self.config, 'context_tokens': 32768})
        self.addCleanup(server.session.close)
        server.config = copy.deepcopy(server.config)
        server.label_ids = {label: index for index, label in enumerate('abcd')}
        invalid = {'completion_probabilities': [{'top_logprobs': [{'id': index, 'logprob': None} for index in range(4)]}]}
        responses = [{'prompt': '<|im_start|>assistant\n'}, {'tokens': [1, 2]}, {'prompt': 'question only'}, {'tokens': [1]}] + [invalid] * 6
        with patch.object(server, 'post', side_effect=responses) as post, patch.object(server, 'restart') as restart, patch.object(server, 'question_memory') as memory, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, r'v2:test:0 after a fresh server restart and 2 retries; no prediction saved'):
                evaluation.score_question(server, question(), self.history[:-1], {}, server.config)
        requests = [call.args[1] for call in post.call_args_list if call.args[0] == '/completion']
        self.assertTrue(all('cache_prompt' not in r for r in requests))
        self.assertEqual([r['n_probs'] for r in requests], [32] * 6)
        restart.assert_called_once_with()
        memory.assert_not_called()

    def test_persistent_invalid_scores_recover_after_restart_with_full_runtime(self):
        server = serving.LlamaCppServer(Path('.'), {**self.config, 'context_tokens': 32768})
        self.addCleanup(server.session.close)
        server.label_ids = {label: index for index, label in enumerate('abcd')}
        def result(values):
            return {'completion_probabilities': [{'top_logprobs': [{'id': index, 'logprob': value} for index, value in enumerate(values)]}]}
        invalid = result([None] * 4)
        valid = result([-3.0, -1.0, -4.0, -2.0])
        responses = [{'prompt': '<|im_start|>assistant\n'}, {'tokens': [1, 2]}, {'prompt': 'question only'}, {'tokens': [1]}, invalid, invalid, invalid, valid]
        with patch.object(server, 'post', side_effect=responses) as post, patch.object(server, 'restart') as restart, patch.object(server, 'question_memory', return_value={}) as memory, patch.object(serving.time, 'perf_counter', side_effect=[10, 11, 12, 13, 14, 15, 16, 30, 32, 33]), contextlib.redirect_stdout(io.StringIO()) as output:
            prediction = evaluation.score_question(server, question(), self.history[:-1], {}, server.config)
        restart.assert_called_once_with()
        requests = [call.args[1] for call in post.call_args_list if call.args[0] == '/completion']
        self.assertEqual({r['prompt'] for r in requests}, {'<|im_start|>assistant\n'})
        self.assertTrue(all(not r['post_sampling_probs'] for r in requests))
        self.assertNotIn('cache_prompt', requests[-1])
        self.assertIn('Restarting the owned Qwen model server once', output.getvalue())
        self.assertEqual(prediction['predicted_label'], 'b')
        self.assertEqual(prediction['label_raw_logprobs'], dict(zip('abcd', [-3.0, -1.0, -4.0, -2.0])))
        self.assertEqual(prediction['logprob_attempts'], 4)
        self.assertEqual(prediction['server_restarts'], 1)
        self.assertEqual(prediction['inference_seconds'], 5)
        self.assertEqual(prediction['elapsed_seconds'], 23)
        memory.assert_called_once_with(10, 33)
        json.dumps(prediction, allow_nan=False)

    def test_invalid_present_label_is_retried_before_expanding_missing_labels(self):
        server = serving.LlamaCppServer(Path('.'), {**self.config, 'context_tokens': 32768})
        self.addCleanup(server.session.close)
        server.config = copy.deepcopy(server.config)
        server.label_ids = {label: index for index, label in enumerate('abcd')}
        def result(ids, invalid=False):
            return {'completion_probabilities': [{'top_logprobs': [{'id': index, 'logprob': None if invalid else -index - 1.0} for index in ids]}]}
        responses = [{'prompt': '<|im_start|>assistant\n'}, {'tokens': [1, 2]}, {'prompt': 'question only'}, {'tokens': [1]}, result([0], invalid=True), result([0, 1, 2]), result([0, 1, 2, 3])]
        with patch.object(server, 'post', side_effect=responses) as post, patch.object(server, 'question_memory', return_value={}), contextlib.redirect_stdout(io.StringIO()):
            prediction = evaluation.score_question(server, question(), self.history[:-1], {}, server.config)
        requests = [call.args[1] for call in post.call_args_list if call.args[0] == '/completion']
        self.assertEqual([r['n_probs'] for r in requests], [32, 32, 128])
        self.assertTrue(all('cache_prompt' not in r for r in requests))
        self.assertEqual(prediction['logprob_attempts'], 3)
        self.assertEqual(prediction['predicted_label'], 'a')


    def test_cache_grouping_keeps_limited_persona_coverage_and_all_history_groups(self):
        rows = [{**question(), 'example_id': f'{person}:{hist}:{i}',
                 'persona_id': person, 'history_id': hist}
                for i in range(3) for person in ('v2:1', 'v2:2') for hist in ('a', 'b')]
        ordered = evaluation.order_for_cache(rows)
        self.assertEqual({q['example_id'] for q in ordered}, {q['example_id'] for q in rows})
        groups = [(q['persona_id'], q['history_id']) for q in ordered]
        self.assertEqual(groups, sorted(groups))
        limited = evaluation.order_for_cache(rows, 4)
        self.assertEqual({q['persona_id'] for q in limited}, {'v2:1', 'v2:2'})
        self.assertEqual({q['example_id'] for q in limited}, {q['example_id'] for q in rows[:4]})


    def test_questions_share_history_prefix_without_accumulating_previous_queries(self):
        first = evaluation.build_messages(question(), self.config, self.history[:-1])
        second = evaluation.build_messages({**question(), 'question': 'A different question'}, self.config, self.history[:-1])
        self.assertEqual(first[:-1], second[:-1])
        self.assertNotEqual(first[-1], second[-1])
        self.assertNotIn(question()['question'], second[-1]['content'])

    def test_128k_and_old_benchmark_arguments_are_rejected(self):
        for args in (['--dataset-version', '128k'], ['--benchmark', 'personamem_v2_32k']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.parse_args(args)


    def test_insufficient_context_is_rejected_without_truncation(self):
        server = serving.LlamaCppServer(Path('.'), {**self.config, 'context_tokens': 32768})
        server.actual_context = 100
        self.addCleanup(server.session.close)
        with patch.object(server, 'post', side_effect=[{'prompt': 'complete prompt'}, {'tokens': list(range(100))}]) as post:
            with self.assertRaisesRegex(ValueError, 'will not be truncated'):
                evaluation.score_question(server, question(), self.history[:-1], {}, server.config)
        self.assertEqual([call.args[0] for call in post.call_args_list], ['/apply-template', '/tokenize'])


if __name__ == '__main__':
    unittest.main()
