"""Native launch configuration and direct llama.cpp token-ID scoring."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import download_models
import evaluate_qwen
import qwen_llamacpp_server as serving


class NativeTests(unittest.TestCase):
    def test_selector_context_is_memory_budget_plus_one_thousand_for_all_datasets(self):
        for dataset in ('v2', 'v1', 'locomo'):
            for budget in (1024, 2048, 4096):
                with self.subTest(dataset=dataset, budget=budget):
                    settings = evaluate_qwen.runtime_metadata(dataset, 'selector', memory_tokens=budget)
                    self.assertEqual(settings['ContextTokens'], budget + 1000)
                    command = settings['Command']
                    self.assertEqual(command[command.index('--ctx-size') + 1], str(budget + 1000))
                    direct = serving.connection_settings(serving.parse_args(
                        ['--dataset', dataset, '--mode', 'selector', '--memory-tokens', str(budget)]))
                    self.assertEqual(direct['ContextTokens'], settings['ContextTokens'])
            self.assertEqual(serving.server_settings('selector', dataset=dataset)['ContextTokens'], 3048)

    def test_native_selector_context_rejects_invalid_budgets(self):
        for argv in (['--memory-tokens', '2048'], ['--mode', 'selector', '--memory-tokens', '0'],
                     ['--mode', 'selector', '--memory-tokens', '262144']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                serving.parse_args(argv)

    def test_evaluator_contains_no_server_class_or_network_process_implementation(self):
        tree = ast.parse(Path(evaluate_qwen.__file__).read_text(encoding='utf-8'))
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        self.assertFalse(any('Server' in name for name in classes), classes)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or '').split('.')[0])
        self.assertFalse(imports & {'requests', 'socket', 'subprocess', 'threading'}, imports)
        self.assertEqual(serving.LlamaCppServer.__module__, serving.__name__)

    def test_only_pinned_four_billion_quantized_model_is_supported(self):
        path = serving.ROOT / 'configs/qwen_server.json'
        original = path.read_bytes()
        settings = serving.server_settings(dataset='v1')
        self.assertEqual(settings['Model'], 'Qwen3.5-4B-Q4_K_M')
        self.assertEqual(settings['Dtype'], 'Q4_K_M')
        self.assertTrue(settings['ModelPath'].endswith('Qwen3.5-4B-Q4_K_M.gguf'))
        assets = json.loads((serving.ROOT / 'configs/assets.lock.json').read_text(encoding='utf-8'))
        self.assertEqual(settings['ModelRevision'], assets['answer']['revision'])
        self.assertEqual(Path(settings['ModelPath']), serving.ROOT / assets['answer']['files'][0]['path'])
        self.assertNotIn('answer_2b', assets)
        self.assertEqual(path.read_bytes(), original)

    def test_resume_keeps_the_sole_model_and_question_selection(self):
        protocol = {'config': {'dataset': 'v1', 'baseline': 'full_history', 'subset': 'full'}, 'split': 'comparison', 'limit': 3}
        arguments = evaluate_qwen.resume_arguments(Path('results/run/protocol.json'), protocol)
        args = evaluate_qwen.parse_args(arguments)
        self.assertNotIn('--model-size', arguments)
        self.assertEqual((args.dataset, args.limit), ('v1', 3))

    def test_old_model_predictions_cannot_be_mixed_with_four_billion_q4(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.jsonl'
            path.write_bytes(b'preserve saved predictions')
            previous = {'model': 'Qwen3.5-2B-BF16-GGUF', 'model_revision': 'old',
                        'model_dtype': 'bfloat16', 'backend': 'llama_cpp'}
            current = {**previous, 'model': 'Qwen3.5-4B-Q4_K_M', 'model_revision': 'new',
                       'model_dtype': 'Q4_K_M'}
            with self.assertRaisesRegex(ValueError, 'different model'):
                evaluate_qwen.validate_resume_model({'config': previous}, current, path)
            self.assertEqual(path.read_bytes(), b'preserve saved predictions')

    def test_removed_model_switches_are_rejected(self):
        for parser in (serving.parse_args, evaluate_qwen.parse_args):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser(['--model-size', '2b'])

    def test_print_settings_never_starts_a_process(self):
        out = io.StringIO()
        with patch.object(serving.subprocess, 'run', side_effect=AssertionError('Unexpected launch')), contextlib.redirect_stdout(out):
            serving.main(['--dataset', 'v1', '--print-settings'])
        settings = json.loads(out.getvalue())
        self.assertEqual(settings['Backend'], 'llama_cpp')
        self.assertEqual(settings['ContextTokens'], 33280)
        self.assertEqual(settings['Generation']['presence_penalty'], 2.0)
        self.assertFalse(settings['ContextShift'])
        self.assertFalse(settings['Fit'])
        self.assertEqual(settings['ChatTemplateSource'], 'gguf_metadata')
        self.assertNotIn('--chat-template-file', settings['Command'])

    def test_native_endpoint_preserves_raw_ids_and_enables_prompt_cache(self):
        raw = {'completion_probabilities': [{'top_logprobs': [
            {'id': 64, 'logprob': -1.25}, {'id': 65, 'logprob': -2.5}]}], 'truncated': False}
        response = MagicMock(status_code=200)
        response.json.return_value = raw
        server = serving.LlamaCppServer(Path('.'), {
            'backend': 'llama_cpp', 'server_settings': {'RequestTimeoutSeconds': 120}})
        self.addCleanup(server.session.close)
        server.base = 'http://127.0.0.1:8080'
        payload = {'prompt': 'exact prompt', 'n_predict': 1, 'n_probs': 64, 'post_sampling_probs': False}
        with patch.object(server.session, 'post', return_value=response) as post:
            self.assertEqual(server.post('/completion', payload), raw)
        post.assert_called_once_with('http://127.0.0.1:8080/completion',
                                     json={**payload, 'cache_prompt': True}, timeout=120)
        self.assertNotIn('cache_prompt', payload)
        self.assertFalse(server.session.trust_env)
        self.assertNotIn('Authorization', server.session.headers)

    def test_runtime_zip_rejects_traversal_and_verifies_pinned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'runtime.zip'
            data = b'pinned runtime fixture'
            source = {'files': [{'path': 'models/llama.cpp/server.exe', 'bytes': len(data),
                                  'sha256': hashlib.sha256(data).hexdigest()}]}
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../outside', b'bad')
            with self.assertRaises(ValueError):
                download_models.extract_runtime_archives(root, source, [archive])
            self.assertFalse((root / 'models/outside').exists())
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('server.exe', data)
            download_models.extract_runtime_archives(root, source, [archive])
            self.assertEqual((root / 'models/llama.cpp/server.exe').read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
