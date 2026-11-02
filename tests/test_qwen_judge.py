"""Candidate-only relevance judging through the existing native logprob scorer."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import label_reference_messages as labels
import qwen_llamacpp_server as serving
from raw_datasets import read_rows, sha256
from test_message_retrieval import write_fixture

REAL_SLEEP = time.sleep  # the judge's restart pause is patched; fake server delays are not


class JudgeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        write_fixture(self.root)
        path = self.root / 'data/raw/personamem_v2/benchmark/text/train.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['related_conversation_snippet'] = json.dumps([{'role':'user','content':'I prefer tea.'}])
        # A second question over the same history checks ordering across questions.
        rows.append({**rows[0], 'user_query': json.dumps({'role': 'user', 'content': 'drink choice?'})})
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (self.root / 'data/splits/v2/train_ids.json').write_text(
            json.dumps(['v2:train:00000', 'v2:train:00001']), encoding='utf-8')
        for name in ('qwen_server.json', 'evaluate_qwen.json', 'assets.lock.json'):
            (self.root / 'configs').mkdir(exist_ok=True)
            shutil.copyfile(labels.ROOT / 'configs' / name, self.root / 'configs' / name)
        questions, self.memories, _ = labels.load_message_dataset('v2', ['train'], root=self.root)
        self.question = questions[0]
        members = [self.memories[mid] for mid in self.question['memory_ids']]
        def row(m, label):
            return {k:m[k] for k in ('message_id','message_index','role','text')} | {'label':label}
        def record(question):
            return {k:question[k] for k in ('example_id','dataset','split','persona_id','history_id','question')} | dict(
                schema_version=1, references=[row(members[0], 2)], candidates=[row(m, None) for m in members[1:]],
                unjudged_message_ids=[], unselected_label=None)
        self.record, self.second_record = record(questions[0]), record(questions[1])
        self.records = [self.record]
        self.input = self.root / 'data/label_candidates/v2/train.jsonl'
        self.input.parent.mkdir(parents=True)
        self.save_input()
        self.out = self.root / 'data/qwen_labels/v2'
        # Concurrent judging calls the server in any order, so answers are keyed by message.
        self.useful = {m['message_id']: m['text'] == 'Tea remembered.' for m in members}

    def save_input(self):
        self.input.write_text(''.join(json.dumps(r) + '\n' for r in self.records), encoding='utf-8')

    def judged_server(self, delay=None):
        server = MagicMock()
        def score(messages, *, request_id, history_messages):
            question_id, message_id = request_id.split('|')
            if delay:
                REAL_SLEEP(delay(question_id))
            return self.prediction('a' if self.useful[message_id] else 'b')
        server.score.side_effect = score
        return server

    @staticmethod
    def prediction(label='a'):
        scores = {'a':-.2, 'b':-2.}
        if label == 'b':
            scores = {'a':-2., 'b':-.2}
        import math
        norm = math.log(sum(math.exp(v) for v in scores.values()))
        return dict(predicted_label=label, label_raw_logprobs=scores,
                    label_conditional_logprobs={k:v-norm for k,v in scores.items()},
                    prompt_sha256='synthetic', prompt_tokens=100, logprob_attempts=1)

    def run_judge(self, server, **options):
        manager = MagicMock()
        manager.__enter__.return_value = server
        with patch.object(serving, 'LlamaCppServer', return_value=manager) as launch, \
                patch.object(labels.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            result = labels.run_judge(root=self.root, **options)
        return result, launch

    def test_selected_pairs_get_zero_or_one_references_stay_two_and_resume_is_idle(self):
        before = sha256(self.input)
        server = self.judged_server()
        report, launch = self.run_judge(server)
        self.assertEqual(report['status'], 'complete')
        output = read_rows(self.out / 'train.jsonl')[0]
        self.assertEqual([m['label'] for m in output['memories']], [2, 1, 0])
        self.assertEqual(output['references'][0]['label'], 2)
        self.assertEqual([m['label'] for m in output['candidates']], [1, 0])
        self.assertEqual(server.score.call_count, 2)
        self.assertEqual(sha256(self.input), before)
        config = launch.call_args.args[1]
        self.assertEqual(config['answer_labels'], ['b', 'a'])
        self.assertEqual(config['baseline'], 'judge')
        for call in server.score.call_args_list:
            prompt = json.dumps(call.args[0])
            self.assertNotIn('REFERENCE ANSWER', prompt)
            self.assertNotIn('DISTRACTOR', prompt)
            self.assertNotIn('SECRET PERSONA', prompt)
            self.assertNotIn('I prefer tea.', prompt)  # reference evidence is not shown to the judge
        second, launch = self.run_judge(server)
        launch.assert_not_called()
        self.assertEqual(second['status'], 'complete')

    def test_unselected_messages_remain_unjudged(self):
        omitted = self.record['candidates'].pop()
        self.record['unjudged_message_ids'] = [omitted['message_id']]
        self.save_input()
        self.run_judge(self.judged_server())
        output = read_rows(self.out / 'train.jsonl')[0]
        self.assertEqual([m['label'] for m in output['memories']], [2, 1, -1])
        self.assertIsNone(output['unselected_label'])

    def test_stale_text_reference_override_and_wrong_split_fail_before_server(self):
        for category, field, value in (('candidates','text','changed'), ('references','label',1)):
            old = self.record[category][0][field]
            self.record[category][0][field] = value
            self.save_input()
            with patch.object(serving, 'LlamaCppServer') as launch, self.assertRaises(ValueError):
                labels.run_judge(root=self.root)
            launch.assert_not_called()
            self.record[category][0][field] = old
        self.record['split'] = 'test'
        self.save_input()
        with patch.object(serving, 'LlamaCppServer') as launch, self.assertRaises(ValueError):
            labels.run_judge(root=self.root)
        launch.assert_not_called()

    def test_failures_do_not_become_zero_and_resume_checks_input_hash(self):
        server = MagicMock()
        server.score.side_effect = RuntimeError('Qwen model server closed the connection')
        manager = MagicMock()
        manager.__enter__.return_value = server
        with patch.object(serving, 'LlamaCppServer', return_value=manager) as launch, \
                patch.object(labels.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, 'closed the connection'):
            labels.run_judge(root=self.root, replicas=1, parallel=8)
        # Persistent server failures are restarted a bounded number of times.
        self.assertEqual(launch.call_count, labels.MAX_SERVER_RESTARTS + 1)
        self.assertEqual([call.args[1]['parallel'] for call in launch.call_args_list], [8] * launch.call_count)
        self.assertEqual(read_rows(self.out / 'train.jsonl'), [])
        report = json.loads((self.out / 'report.json').read_text())
        self.assertEqual((report['status'], report['completed_questions']), ('interrupted', 0))
        failures = (self.out / 'judge-failures.log').read_text(encoding='utf-8')
        self.assertEqual(failures.count('closed the connection'), labels.MAX_SERVER_RESTARTS + 1)
        self.input.write_bytes(self.input.read_bytes() + b'\n')
        with patch.object(serving, 'LlamaCppServer') as launch, self.assertRaisesRegex(ValueError, 'different'):
            labels.run_judge(root=self.root)
        launch.assert_not_called()

    def test_deterministic_scoring_errors_are_not_retried(self):
        server = MagicMock()
        server.score.side_effect = ValueError('Qwen judge prediction disagrees with the raw label scores')
        manager = MagicMock()
        manager.__enter__.return_value = server
        with patch.object(serving, 'LlamaCppServer', return_value=manager) as launch, \
                patch.object(labels.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, 'disagrees'):
            labels.run_judge(root=self.root)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(json.loads((self.out / 'report.json').read_text())['status'], 'interrupted')

    def test_transient_server_failure_restarts_and_completes(self):
        healthy = self.judged_server()
        failing = MagicMock()
        failing.score.side_effect = RuntimeError('Qwen model server did not respond during /completion')
        managers = [MagicMock(), MagicMock()]
        for manager, server in zip(managers, (failing, healthy)):
            manager.__enter__.return_value = server
        with patch.object(serving, 'LlamaCppServer', side_effect=managers) as launch, \
                patch.object(labels.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            report = labels.run_judge(root=self.root, replicas=1)
        self.assertEqual((report['status'], launch.call_count), ('complete', 2))
        self.assertEqual([m['label'] for m in read_rows(self.out / 'train.jsonl')[0]['memories']], [2, 1, 0])
        self.assertIn('train.jsonl', report['files_sha256'])

    def out_of_memory_server(self):
        failing = MagicMock()
        def out_of_memory(*args, **kwargs):
            (self.out / 'servers/0/qwen-server.log').write_text('CUDA error: out of memory\n', encoding='utf-8')
            raise RuntimeError('Qwen model server exited with code 1 during /completion')
        failing.score.side_effect = out_of_memory
        return failing

    def launch_sequence(self, servers):
        managers = [MagicMock() for _ in servers]
        for manager, server in zip(managers, servers):
            manager.__enter__.return_value = server
        return patch.object(serving, 'LlamaCppServer', side_effect=managers)

    @staticmethod
    def option(config, flag):
        command = config['server_settings']['Command']
        return command[command.index(flag) + 1]

    def test_gpu_memory_failure_halves_parallel_requests(self):
        with self.launch_sequence([self.out_of_memory_server(), self.judged_server()]) as launch, \
                patch.object(labels.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            report = labels.run_judge(root=self.root, parallel=8, replicas=1)
        self.assertEqual([call.args[1]['parallel'] for call in launch.call_args_list], [8, 4])
        self.assertEqual((report['status'], report['parallel']), ('complete', 4))
        protocol = json.loads((self.out / 'protocol.json').read_text())
        self.assertEqual(self.option(protocol['config'], '--parallel'), '4')

    def test_gpu_memory_failure_drops_a_replica_before_slots(self):
        servers = [self.out_of_memory_server()] * 3 + [self.judged_server()] * 2
        with self.launch_sequence(servers) as launch, \
                patch.object(labels.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            report = labels.run_judge(root=self.root)
        self.assertEqual(launch.call_count, 5)
        self.assertEqual((report['status'], report['replicas'], report['parallel']), ('complete', 2, 4))
        self.assertEqual(json.loads((self.out / 'protocol.json').read_text())['replica_ports'], [8080, 8081])

    def test_default_replicas_use_separate_ports_logs_and_share_requests(self):
        servers = [self.judged_server() for _ in range(3)]
        self.records = [self.record, self.second_record]
        self.save_input()
        with self.launch_sequence(servers) as launch, contextlib.redirect_stdout(io.StringIO()):
            report = labels.run_judge(root=self.root)
        configs = [call.args[1] for call in launch.call_args_list]
        self.assertEqual([c['server_settings']['Port'] for c in configs], [8080, 8081, 8082])
        self.assertEqual([self.option(c, '--port') for c in configs], ['8080', '8081', '8082'])
        self.assertEqual({self.option(c, '--parallel') for c in configs}, {'4'})
        self.assertEqual({self.option(c, '--cache-ram') for c in configs}, {'0'})
        self.assertIn('--no-cache-prompt', configs[0]['server_settings']['Command'])
        self.assertEqual([call.args[0] for call in launch.call_args_list],
                         [self.out / 'servers' / str(i) for i in range(3)])
        # Four candidate requests alternate between the three servers.
        self.assertEqual([s.score.call_count for s in servers], [2, 1, 1])
        self.assertEqual((report['status'], report['replicas'], report['completed_questions']), ('complete', 3, 2))

    def test_questions_are_saved_in_candidate_order_when_later_ones_finish_first(self):
        self.records = [self.record, self.second_record]
        self.save_input()
        first = self.record['example_id']
        server = self.judged_server(delay=lambda qid: .3 if qid == first else 0)
        report, _ = self.run_judge(server)
        saved = read_rows(self.out / 'train.jsonl')
        self.assertEqual([r['example_id'] for r in saved], [first, self.second_record['example_id']])
        self.assertEqual(report['completed_questions'], 2)
        self.assertEqual(report['grade_counts'], {'2': 2, '1': 2, '0': 2})

    def test_changed_judge_prompt_blocks_resume(self):
        self.run_judge(self.judged_server())
        protocol_path = self.out / 'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        protocol['config']['judge_prompt_sha256'] = 'older prompt'
        protocol_path.write_text(json.dumps(protocol))
        with patch.object(serving, 'LlamaCppServer') as launch, self.assertRaisesRegex(ValueError, 'judge prompt'):
            labels.run_judge(root=self.root)
        launch.assert_not_called()

    def test_judged_candidates_cannot_be_rescored(self):
        self.record['candidates'][0]['label'] = 0
        self.save_input()
        with patch.object(serving, 'LlamaCppServer') as launch, self.assertRaisesRegex(ValueError, 'unjudged'):
            labels.run_judge(root=self.root)
        launch.assert_not_called()

    def test_native_score_requests_pre_sampling_token_logprobs(self):
        config = dict(baseline='judge', context_tokens=4096, initial_top_logprobs=2, maximum_top_logprobs=8)
        server = serving.LlamaCppServer(self.root, config)
        self.addCleanup(server.session.close)
        server.label_ids = {'b':22, 'a':11}
        response = {'completion_probabilities':[{'top_logprobs':[
            {'id':22,'logprob':-.2}, {'id':11,'logprob':-2.}]}]}
        memory = self.memories[self.record['candidates'][0]['message_id']]
        with patch.object(server, 'render_prompt', return_value=('prompt',100)), \
                patch.object(server, 'post', return_value=response) as post, \
                patch.object(server, 'question_memory', return_value={}), contextlib.redirect_stdout(io.StringIO()):
            judged = labels.judge_candidate(server, self.question, memory)
        self.assertEqual(judged['label'], 0)
        payload = post.call_args.args[1]
        self.assertFalse(payload['post_sampling_probs'])
        self.assertEqual(payload['n_predict'], 1)
        self.assertNotIn('logit_bias', payload)
        self.assertIn('label_raw_logprobs', judged['judge'])


if __name__ == '__main__':
    unittest.main()
