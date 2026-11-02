"""Reference-label isolation and saved-ranker evaluation without Qwen or a GPU."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_qwen as evaluation
import select_label_candidates as candidates
from feature_contract import save_model_recipe
from subtlex_fixture import recipe as test_recipe
from raw_datasets import load_dataset, memories_for_question
from test_message_retrieval import write_fixture


class MemoryModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import lightgbm as lgb
        X = np.random.default_rng(19).normal(size=(120, 55)).astype(np.float32)
        model = lgb.LGBMRanker(n_estimators=3, num_leaves=4, min_child_samples=3,
                              n_jobs=1, verbosity=-1)
        model.fit(X, (X[:, 0] > 0).astype(int), group=[40, 40, 40],
                  feature_name=candidates.FEATURE_NAMES)
        cls.model_text = model.booster_.model_to_string()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_fixture(self.root)
        for name in ('evaluate_qwen.json', 'qwen_server.json', 'assets.lock.json'):
            (self.root / 'configs').mkdir(exist_ok=True)
            shutil.copyfile(evaluation.ROOT / 'configs' / name, self.root / 'configs' / name)
        for name in ('evaluate_qwen.py', 'qwen_llamacpp_server.py'):
            (self.root / 'scripts').mkdir(exist_ok=True)
            shutil.copyfile(evaluation.ROOT / 'scripts' / name, self.root / 'scripts' / name)
        for dataset in ('v2', 'v1', 'locomo'):
            self.save(f'data/splits/{dataset}/persona_split.json', {'subset': 'compact' if dataset == 'v2' else 'comparison'})
        self.save('data/splits/v2/test_ids.json', ['v2:train:00000'])
        path = self.root / 'data/raw/personamem_v2/data/chat_history_32k/linked_history.json'
        value = json.loads(path.read_text())
        value['chat_history'][-1]['content'] = 'Final reply.'
        path.write_text(json.dumps(value), encoding='utf-8')
        path = self.root / 'data/raw/personamem_v1/shared_contexts_32k.jsonl'
        value = json.loads(path.read_text())
        value['context'][-1]['content'] = 'Future reply.'
        path.write_text(json.dumps(value) + '\n', encoding='utf-8')
        for split in ('train', 'val', 'benchmark'):
            self.edit_csv(f'data/raw/personamem_v2/benchmark/text/{split}.csv',
                          incorrect_answers=json.dumps(['B', 'C', 'D']),
                          related_conversation_snippet=json.dumps([{'role': 'user', 'content': 'I prefer tea.'}]))
        self.edit_csv('data/raw/personamem_v1/questions_32k.csv',
                      all_options=json.dumps(['REFERENCE ANSWER', 'B', 'C', 'D']))
        path = self.root / 'data/raw/locomo/locomo_mc10.json'
        value = json.loads(path.read_text())
        value['choices'] = ['REFERENCE ANSWER', *[str(i) for i in range(9)]]
        path.write_text(json.dumps(value) + '\n', encoding='utf-8')
        self.model = self.root / 'models/subtlex_mix/alpha_0.5/ranker55.txt'
        self.model.parent.mkdir(parents=True)
        self.model.write_text(self.model_text, encoding='utf-8')
        save_model_recipe(self.model, test_recipe())

    def save(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def edit_csv(self, relative, **values):
        path = self.root / relative
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.update(values)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def inputs(self, dataset):
        split = 'test' if dataset == 'v2' else 'comparison'
        questions, contexts, _, sources = load_dataset(dataset, [split], root=self.root)
        return questions, contexts, sources

    def features(self, dataset):
        questions, contexts, sources = self.inputs(dataset)
        split = questions[0]['split']
        ids, offsets = [], [0]
        for q in questions:
            ids.extend(m['message_id'] for m in memories_for_question(q, contexts))
            offsets.append(len(ids))
        arrays = {'X': np.random.default_rng(9).normal(size=(len(ids), 55)).astype(np.float32),
                  'feature_names': np.array(candidates.FEATURE_NAMES),
                  'offsets': np.array(offsets), 'question_ids': np.array([q['example_id'] for q in questions]),
                  'message_ids': np.array(ids),
                  'metadata_json': np.array(json.dumps({'dataset': dataset, 'split': split, 'feature_recipe':test_recipe(),
                                                      'memory_unit': 'single_message', 'source_sha256': sources})),
                  # These would fail if the evaluator ever tried to load them.
                  'y': np.array([{'poison': 'do not read labels'}], dtype=object),
                  'judged_labels': np.array([{'poison': 'do not read Qwen labels'}], dtype=object)}
        path = self.root / f'data/ranker55/{dataset}/{split}/features.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        return path, arrays

    def test_cli_supports_fixed_dataset_modes_and_selector_defaults(self):
        for dataset in ('v2', 'v1', 'locomo'):
            for mode in evaluation.EVALUATION_MODES:
                argv = ['--dataset', dataset, '--mode', mode]
                supported = dataset == 'v2' or mode in ('full_history', 'selector')
                with self.subTest(dataset=dataset, mode=mode), patch.object(evaluation, 'ROOT', self.root):
                    if not supported:
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                            evaluation.parse_args(argv)
                        continue
                    args = evaluation.parse_args(argv)
                    self.assertFalse(hasattr(args, 'subset'))
                    self.assertEqual(args.split, 'test' if dataset == 'v2' else 'comparison')
                    if mode == 'selector':
                        self.assertIsNone(args.top_k)
                        self.assertEqual(args.memory_tokens, 2048)
                        self.assertEqual(args.selector_model, self.model)
                        self.assertEqual(args.selector_features, self.root / f'data/ranker55/{dataset}/{args.split}/features.npz')
        explicit = evaluation.parse_args(['--mode', 'selector', '--top-k', '10'])
        self.assertEqual(explicit.top_k, 10)
        self.assertIsNone(explicit.memory_tokens)
        self.assertEqual(evaluation.parse_args(['--mode', 'selector', '--token-budget', '1024']).memory_tokens, 1024)
        for argv in (['--mode', 'selector', '--top-k', '0'], ['--top-k', '10'], ['--selector-model', 'x'],
                     ['--mode', 'selector', '--memory-tokens', '0'], ['--memory-tokens', '2048'],
                     ['--mode', 'selector', '--top-k', '10', '--memory-tokens', '2048']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.parse_args(argv)

    def test_references_are_original_label_two_only_not_qwen_label_one(self):
        self.save('data/qwen_labels/test.jsonl', {'label': 1, 'text': 'Tea remembered.'})
        questions, contexts, sources = self.inputs('v2')
        inputs = evaluation.ReferenceHistories(questions, contexts, sources, root=self.root)
        history, source = inputs.get(questions[0])
        self.assertEqual(history, [{'role': 'user', 'content': 'I prefer tea.'}])
        self.assertEqual(source['reference_label'], 2)
        self.assertEqual(source['reference_messages_missing_from_visible_history'], 0)
        self.assertEqual(inputs.provenance['included_labels'], [2])
        self.assertFalse(inputs.provenance['qwen_generated_labels_used'])
        prompt = evaluation.build_messages(questions[0], evaluation.evaluation_config('v2', 'reference_only'), history)
        self.assertIn('I prefer tea.', prompt[1]['content'])
        for excluded in ('Tea remembered.', 'Forget my old preference.', 'SECRET PERSONA', 'all visible source messages'):
            self.assertNotIn(excluded, prompt[1]['content'])

    def test_missing_annotated_reference_is_preserved_and_reported_as_oracle(self):
        self.edit_csv('data/raw/personamem_v2/benchmark/text/train.csv',
                      related_conversation_snippet=json.dumps([{'role': 'user', 'content': 'Original absent reference'}]))
        questions, contexts, sources = self.inputs('v2')
        inputs = evaluation.ReferenceHistories(questions, contexts, sources, root=self.root)
        history, source = inputs.get(questions[0])
        self.assertEqual(history[0]['content'], 'Original absent reference')
        self.assertEqual(source['reference_messages_missing_from_visible_history'], 1)
        self.assertEqual(source['reference_label'], 2)

    def test_selector_stable_ties_chronology_and_locomo_metadata_without_labels(self):
        features, _ = self.features('locomo')
        questions, contexts, sources = self.inputs('locomo')
        inputs = evaluation.SelectorHistories(questions, contexts, sources, features, self.model, top_k=3)
        self.assertEqual(inputs.models.models[0].params['device_type'], 'cpu')
        booster = inputs.models.models[0]
        with patch.object(booster, 'predict', wraps=booster.predict) as cpu_predict:
            inputs.models.predict(questions[0], np.ones((4, 55)))
        self.assertEqual(cpu_predict.call_args.kwargs, {'num_threads': 4, 'device_type': 'cpu'})
        with patch.object(inputs.models, 'predict', return_value=(np.array([1., 5., 5., 6.]), 0)) as predict:
            history, source = inputs.get(questions[0])
        self.assertEqual([m['source_message_index'] for m in history], [1, 2, 3])
        self.assertEqual([m['rank'] for m in source['selected_messages']], [2, 3, 1])
        self.assertTrue(all(m['session_id'] == 's2' and m['timestamp'] == 'Tuesday' for m in history))
        for key in ('options', 'correct_answer', 'reference_labels', 'judged_labels'):
            self.assertNotIn(key, predict.call_args.args[0])
        prompt = evaluation.build_messages(questions[0], evaluation.evaluation_config('locomo', 'selector'), history)
        self.assertIn('timestamp: Tuesday', prompt[1]['content'])
        self.assertIn('[Message 1 | role: assistant]', prompt[1]['content'])
        self.assertNotIn('I prefer tea.', prompt[1]['content'])
        inputs.top_k = 2
        with patch.object(inputs.models, 'predict', return_value=(np.ones(4), 0)):
            history, _ = inputs.get(questions[0])
        self.assertEqual([m['source_message_index'] for m in history], [0, 1])

    def test_selector_respects_v1_positive_and_negative_cutoffs(self):
        features, _ = self.features('v1')
        questions, contexts, sources = self.inputs('v1')
        inputs = evaluation.SelectorHistories(questions, contexts, sources, features, self.model, top_k=10)
        for q, indices in zip(questions, ([1], [1, 2])):
            history, source = inputs.get(q)
            self.assertEqual([m['source_message_index'] for m in history], indices)
            self.assertEqual(source['available_memory_count'], len(indices))
            self.assertNotIn('Future reply.', json.dumps(history))

    def test_selector_rejects_wrong_split_stale_sources_missing_metadata_and_wrong_pool(self):
        path, arrays = self.features('v2')
        questions, contexts, sources = self.inputs('v2')
        for variant in ('split', 'sources', 'metadata', 'pool', 'missing_question'):
            bad = dict(arrays)
            metadata = json.loads(str(arrays['metadata_json'].item()))
            if variant == 'split':
                metadata['split'] = 'train'
            elif variant == 'sources':
                metadata['source_sha256'] = {}
            elif variant == 'pool':
                bad['message_ids'] = arrays['message_ids'][::-1]
            elif variant == 'missing_question':
                bad['question_ids'] = np.array(['v2:test:99999'])
            bad['metadata_json'] = np.array(json.dumps(metadata))
            if variant == 'metadata':
                bad.pop('metadata_json')
            np.savez(path, **bad)
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                evaluation.SelectorHistories(questions, contexts, sources, path, self.model)

    def fake_server(self):
        server = MagicMock()
        server.__enter__.return_value = server
        server.hardware = {}
        server.actual_context = 32768
        # A deterministic stand-in for the native tokenizer, including message
        # headers and chat framing. Production always calls the real server.
        def render(messages):
            cost = len(messages[1]['content']) + 10 if len(messages) == 3 else 0
            return 'prompt', 100 + cost
        server.render_prompt.side_effect = render
        def score(messages, **kwargs):
            labels = list('abcdefghij') if 'or j.' in messages[0]['content'] else list('abcd')
            probabilities = {label: float(-i-1) for i, label in enumerate(labels)}
            return {'predicted_label': 'a', 'label_conditional_logprobs': probabilities,
                    'label_raw_logprobs': probabilities, 'prompt_tokens': render(messages)[1],
                    'memory_tokens': render(messages)[1] - 100,
                    'inference_seconds': .01, 'elapsed_seconds': .02 + kwargs['selection_seconds']}
        server.score.side_effect = score
        return server

    def test_token_budget_skips_oversized_messages_and_includes_formatting(self):
        question = self.inputs('v2')[0][0]
        config = {**evaluation.evaluation_config('v2', 'selector'), 'context_tokens': 32768}
        members = [{'message_index': i + 1, 'source_message': {'role': 'user', 'content': text}}
                   for i, text in enumerate(('LONG' * 1000, 'Small second.', 'Small third.'))]
        server = self.fake_server()
        expected_history = evaluation.selected_history(members, [1, 2])
        # Exact fit includes the heading, original indices, role labels, chat
        # framing, separators, and content, not just tokenized message text.
        budget = server.render_prompt(evaluation.build_messages(question, config, expected_history))[1] - 100
        chosen, metrics = evaluation.pack_selector_messages(server, question, config, members, [0, 2, 1], budget)
        self.assertEqual(chosen, [2, 1])
        self.assertEqual(evaluation.selected_history(members, chosen), expected_history)
        self.assertEqual(metrics['packed_memory_tokens'], budget)
        self.assertEqual(metrics['budget_skipped_messages'], 1)
        self.assertGreater(budget, sum(len(m['content']) for m in expected_history))
        server.score.assert_not_called()

    def test_budget_reserves_answer_token_and_allows_empty_selection(self):
        questions, contexts, _ = self.inputs('locomo')
        question = questions[0]
        config = {**evaluation.evaluation_config('locomo', 'selector'), 'context_tokens': 100000}
        members = memories_for_question(question, contexts)
        server = self.fake_server()
        first = evaluation.build_messages(question, config, evaluation.selected_history(members, [0]))
        first_tokens = server.render_prompt(first)[1]
        server.actual_context = first_tokens + 1  # exactly one answer token left
        chosen, metrics = evaluation.pack_selector_messages(server, question, config, members, list(range(4)), 9000)
        self.assertEqual(chosen, [0])
        self.assertEqual(metrics['effective_memory_token_budget'], first_tokens - 100)
        self.assertEqual(metrics['packed_memory_tokens'], first_tokens - 100)
        self.assertIn('timestamp: Monday', first[1]['content'])
        server.actual_context = 32768  # do not exceed the requested cap if the server reports more
        config['context_tokens'] = first_tokens + 1
        chosen, metrics = evaluation.pack_selector_messages(server, question, config, members, list(range(4)), 9000)
        self.assertEqual(chosen, [0])
        self.assertEqual(metrics['effective_memory_token_budget'], first_tokens - 100)
        chosen, metrics = evaluation.pack_selector_messages(server, question, config, members, list(range(4)), 1)
        self.assertEqual(chosen, [])
        self.assertEqual(metrics['packed_memory_tokens'], 0)
        self.assertEqual(len(evaluation.build_messages(question, config, [])), 2)
        server.actual_context = 100  # question fits but its one-token answer does not
        with self.assertRaisesRegex(ValueError, 'Question/options prompt'):
            evaluation.pack_selector_messages(server, question, config, members, list(range(4)), 9000)

    def test_token_budget_does_not_impose_a_ten_message_cap(self):
        question = self.inputs('v2')[0][0]
        config = {**evaluation.evaluation_config('v2', 'selector'), 'context_tokens': 32768}
        members = [{'message_index': i, 'source_message': {'role': 'user', 'content': 'Short.'}} for i in range(15)]
        chosen, metrics = evaluation.pack_selector_messages(self.fake_server(), question, config, members, list(range(15)), 2048)
        self.assertEqual(len(chosen), 15)
        self.assertLessEqual(metrics['packed_memory_tokens'], 2048)

    def test_score_cutoff_stops_at_low_scores_only_after_the_token_floor(self):
        question = self.inputs('v2')[0][0]
        config = {**evaluation.evaluation_config('v2', 'selector'), 'context_tokens': 32768}
        members = [{'message_index': i, 'source_message': {'role': 'user', 'content': 'Short.'}} for i in range(6)]
        scores, ranked, server = np.array([3., 2.5, 1.2, 1., .4, .2]), list(range(6)), self.fake_server()
        chosen, metrics = evaluation.pack_selector_messages(server, question, config, members, ranked, 2048,
                                                            scores=scores, score_cutoff=1.5)
        self.assertEqual((chosen, metrics['score_cutoff_stop']), ([0, 1], True))  # 1.2 is more than 1.5 below 3
        # Below the floor, lower-scoring messages are still packed; the cutoff applies once it is reached.
        floor = server.render_prompt(evaluation.build_messages(
            question, config, evaluation.selected_history(members, [0, 1, 2])))[1] - 100
        chosen, metrics = evaluation.pack_selector_messages(server, question, config, members, ranked, 2048,
                                                            scores=scores, score_cutoff=1.5, floor_tokens=floor)
        self.assertEqual((chosen, metrics['packed_memory_tokens']), ([0, 1, 2], floor))
        chosen, metrics = evaluation.pack_selector_messages(server, question, config, members, ranked, 2048)
        self.assertEqual(len(chosen), 6)
        self.assertNotIn('score_cutoff_stop', metrics)
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (['--mode', 'selector', '--floor-tokens', '10'], ['--mode', 'selector', '--score-cutoff', '1', '--top-k', '5'],
                         ['--mode', 'selector', '--score-cutoff', '-1'], ['--score-cutoff', '1'],
                         ['--mode', 'selector', '--score-cutoff', '1', '--floor-tokens', '5000']):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    evaluation.parse_args(argv)
        args = evaluation.parse_args(['--mode', 'selector', '--score-cutoff', '1.5', '--floor-tokens', '1000'])
        self.assertEqual((args.memory_tokens, args.score_cutoff, args.floor_tokens), (2048, 1.5, 1000))

    def test_all_supported_modes_write_correct_reports_and_selector_resume_is_guarded(self):
        for dataset in ('v2', 'v1', 'locomo'):
            self.features(dataset)
            modes = evaluation.EVALUATION_MODES if dataset == 'v2' else ('full_history', 'selector')
            for mode in modes:
                output = self.root / f'results/{dataset}_{mode}'
                argv = ['--dataset', dataset, '--mode', mode, '--output', str(output)]
                server = self.fake_server()
                with self.subTest(dataset=dataset, mode=mode), patch.object(evaluation, 'ROOT', self.root), \
                        patch.object(evaluation.qwen_server, 'LlamaCppServer', return_value=server), \
                        contextlib.redirect_stdout(io.StringIO()):
                    evaluation.main(argv)
                    summary = json.loads((output / 'summary.json').read_text())
                    protocol = json.loads((output / 'protocol.json').read_text())
                    report = (output / 'REPORT.md').read_text(encoding='utf-8')
                    self.assertEqual(summary['baseline'], f'frozen_qwen_{mode}')
                    self.assertEqual(summary['selected_questions'], 2 if dataset == 'v1' else 1)
                    self.assertEqual(protocol['config']['initial_top_logprobs'], 32)
                    if mode == 'reference_only':
                        self.assertEqual(protocol[mode]['included_labels'], [2])
                        self.assertIn('label 1 messages are excluded', report)
                    if mode == 'selector':
                        self.assertNotIn('No selector training, retrieval', report)
                        self.assertIn('LightGBM-selected', report)
                        self.assertEqual(protocol['selector']['memory_tokens'], 2048)
                        self.assertEqual(protocol['config']['context_tokens'], 3048)
                        self.assertEqual(protocol['selector']['configured_context_tokens'], 3048)
                        command = protocol['config']['server_settings']['Command']
                        self.assertEqual(command[command.index('--ctx-size') + 1], '3048')
                        self.assertIn('--memory-tokens', evaluation.resume_arguments(output / 'protocol.json', protocol))
                        saved = (output / 'predictions.jsonl').read_bytes()
                        before = server.score.call_count
                        evaluation.main(evaluation.resume_arguments(output / 'protocol.json', protocol))
                        self.assertEqual(server.score.call_count, before)
                        with self.assertRaisesRegex(ValueError, 'different memory inputs'):
                            evaluation.main(argv + ['--top-k', '1'])
                        with self.assertRaisesRegex(ValueError, 'different memory inputs'):
                            evaluation.main(argv + ['--memory-tokens', '1024'])
                        protocol_path = output / 'protocol.json'
                        original = protocol_path.read_bytes()
                        previous = json.loads(original)
                        previous['selector']['configured_context_tokens'] = 41216
                        protocol_path.write_text(json.dumps(previous), encoding='utf-8')
                        with self.assertRaisesRegex(ValueError, 'different memory inputs'):
                            evaluation.main(argv)
                        protocol_path.write_bytes(original)
                        self.assertEqual((output / 'predictions.jsonl').read_bytes(), saved)

    def test_selector_dry_run_validates_and_ranks_without_starting_qwen(self):
        for dataset in ('v2', 'v1', 'locomo'):
            self.features(dataset)
            stream = io.StringIO()
            with patch.object(evaluation, 'ROOT', self.root), contextlib.redirect_stdout(stream), \
                    patch.object(evaluation.qwen_server, 'LlamaCppServer', side_effect=AssertionError('Qwen must not start')):
                evaluation.main(['--dataset', dataset, '--mode', 'selector', '--dry-run'])
            text = stream.getvalue()
            details = json.loads(text[text.index('{'):])
            self.assertFalse(details['model_loaded'])
            self.assertTrue(details['selector_model_loaded'])
            self.assertEqual(details['context_tokens'], 3048)
            self.assertGreater(details['first_question_inputs']['available_memory_count'], 0)
            self.assertEqual(details['first_question_inputs']['memory_token_budget'], 2048)
            self.assertFalse(details['first_question_inputs']['token_packing_performed'])


if __name__ == '__main__':
    unittest.main()
