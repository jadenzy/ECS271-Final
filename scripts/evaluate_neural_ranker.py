"""Frozen Qwen answer test for the neural rankers at the 3.0-cutoff baseline's message counts.

python scripts/evaluate_neural_ranker.py --variant B --datasets v2 v1 locomo
python scripts/evaluate_neural_ranker.py --variant C --datasets v2 v1 locomo
python scripts/evaluate_neural_ranker.py --variant C_tuned --datasets v2 v1 locomo   # from neural_ranker_tune.py

For each question the saved 3.0-cutoff, 8,192-token run (results/embedding_ab/transfer_cutoff/{dataset}_cut3_mem8192)
fixes how many messages Qwen reads. The neural ranker (scripts/neural_ranker.py --stage score) ranks the whole
visible pool, and packing accepts its highest-ranked whole messages that fit the same 8,192-token cap until it has
that many; they are shown chronologically. Same prompt, server settings and answer-order policy as the baseline
(V2/V1 four cyclic orders, LoCoMo one pass). A question whose chosen set equals the baseline's reuses the baseline's
reads, so those questions carry no GPU drift. Targets are read only after selection and scoring.

Outputs: results/neural_ranker/{dataset}_{variant}_matched/ (predictions.jsonl, resumable; protocol.json;
report.json; runtime.json; server log). Compare with compare_selector_runs.py (--rotation for V2/V1).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import build_ranker_features as builder
from embedding_ab_math import CachedInputs
from evaluate_embedding_ab_cutoff import OUTPUT as CUTOFF_OUTPUT
from evaluate_embedding_ab_transfer import CACHE, DATASETS, FrozenHistories, accept_protocol, answer_scores, frozen_models
from evaluate_qwen import build_messages, evaluation_config, order_for_cache, pack_selector_messages, runtime_metadata, selected_history
from option_rotation_pilot import rotated, summarize
from raw_datasets import ROOT, load_dataset, sha256
from tune_ranker55 import write_json
import qwen_llamacpp_server as qwen_server

OUTPUT = ROOT / 'results/neural_ranker'
CAP = 8192


class MatchedHistories(FrozenHistories):
    def __init__(self, questions, contexts, data, scores, counts):
        super().__init__(questions, contexts, data, scores)
        self.counts = counts

    def get(self, question, server, config):
        key = question['history_id'], question['source_end_index']
        members = self.pools[key]
        i = self.groups[question['example_id']]
        a, b = map(int, self.data.offsets[i:i + 2])
        scores = np.asarray(self.scores[a:b], dtype=float)
        ranked = np.argsort(-scores, kind='stable').tolist()
        target = self.counts[question['example_id']]
        chosen, packing = pack_selector_messages(server, question, config, members, ranked, CAP, blocks=self.blocks[key],
                                                 max_messages=target)
        history = selected_history(members, chosen)
        return history, dict(**packing, selected_memory_count=len(history), target_message_count=target,
                             selected_messages=[dict(message_id=members[j]['message_id'], message_index=members[j]['message_index'],
                                                     score=float(scores[j])) for j in chosen])


def report(out, dataset, rows, labels, variant):
    split, rotations = DATASETS[dataset]
    result = summarize(rows, labels) if rotations > 1 else dict(
        questions=len(rows), accuracy={'one_pass': float(np.mean([r['correct'] for r in rows]))})
    result.update(status='complete', dataset=dataset, split=split, rotations=rotations, variant=variant, memory_tokens=CAP,
                  mean_selected_messages=float(np.mean([r['selected_memory_count'] for r in rows])),
                  mean_target_messages=float(np.mean([r['target_message_count'] for r in rows])),
                  share_below_target=float(np.mean([r['selected_memory_count'] < r['target_message_count'] for r in rows])),
                  mean_memory_tokens=float(np.mean([r['packed_memory_tokens'] for r in rows])),
                  share_reused_baseline_reads=float(np.mean([r['reused_baseline_read'] for r in rows])))
    write_json(out / 'report.json', result)
    print(f'COMPLETE {dataset} {variant}: {result["questions"]} questions; {result["accuracy"]}', flush=True)


def score(dataset, variant):
    started = time.perf_counter()
    _, record, frozen = frozen_models()
    split, rotations = DATASETS[dataset]
    out = OUTPUT / f'{dataset}_{variant}_matched'
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / 'predictions.jsonl'
    rows = [json.loads(line) for line in predictions.open(encoding='utf-8') if line.strip()] if predictions.exists() else []
    base_dir = CUTOFF_OUTPUT / f'{dataset}_cut3_mem8192'
    base = {r['example_id']: r for r in (json.loads(line) for line in (base_dir / 'predictions.jsonl').open(encoding='utf-8') if line.strip())}
    base_protocol = json.loads((base_dir / 'protocol.json').read_text())
    config = evaluation_config(dataset, 'selector')
    labels = config['answer_labels']
    questions, contexts, targets, sources = load_dataset(dataset, [split], root=ROOT)
    questions = order_for_cache(questions)
    path = CACHE / dataset / split
    scoring = json.loads((path / f'neural_{variant}_scoring.json').read_text())
    training = json.loads((OUTPUT / f'training_{variant}.json').read_text())
    if (scoring['model_sha256'] != training['model_sha256'] or scoring['features_sha256'] != sha256(path / 'features.npz')
            or scoring['scores_sha256'] != sha256(path / f'neural_{variant}_scores.npy')):
        raise ValueError('Neural scores differ from the frozen model or features')
    data = CachedInputs(path / 'features.npz')
    for source, expected in sources.items():
        if data.meta['source_sha256'].get(source) != expected:
            raise ValueError('Current raw source or split differs from the feature cache')
    if set(base) != set(data.qids) or base_protocol['question_ids'] != data.qids or base_protocol['config']['memory_tokens'] != CAP:
        raise ValueError('Baseline run does not cover this cohort at the 8,192 cap')
    counts = {qid: len(r['selected_messages']) for qid, r in base.items()}
    scores = np.load(path / f'neural_{variant}_scores.npy', mmap_mode='r')
    histories = MatchedHistories(questions, contexts, data, scores, counts)
    settings = runtime_metadata(dataset, 'selector', memory_tokens=CAP)
    config.update(memory_tokens=CAP, input_policy='raw_complete_messages_v1', server_settings=settings,
                  model=settings['Model'], context_tokens=settings['ContextTokens'])
    if config != base_protocol['config']:
        raise ValueError('Qwen configuration differs from the baseline run')
    protocol = dict(dataset=dataset, split=split, rotations=rotations, config=config, variant=variant,
                    neural_model_sha256=training['model_sha256'], neural_scoring_sha256=sha256(path / f'neural_{variant}_scoring.json'),
                    baseline_predictions=str((base_dir / 'predictions.jsonl').relative_to(ROOT)),
                    baseline_predictions_sha256=sha256(base_dir / 'predictions.jsonl'), baseline_protocol_sha256=sha256(base_dir / 'protocol.json'),
                    raw_sources=sources, question_ids=data.qids, questions_sha256=builder.fingerprint(questions),
                    targets_sha256=builder.fingerprint(targets), script_sha256=sha256(Path(__file__)),
                    scorer_sha256=sha256(Path(qwen_server.__file__)), frozen_encoder=frozen['encoder'],
                    selection_policy=('neural ranker order over the whole visible pool; accept whole messages that fit an 8,192-token '
                                      'cap until the 3.0-cutoff baseline\'s message count for this question; chronological display'),
                    uses_evaluation_labels_for_selection=False)
    done = accept_protocol(out / 'protocol.json', protocol, rows, data.qids)
    todo = [q for q in questions if q['example_id'] not in done]
    print(f'SCORING {dataset} with neural ranker {variant}: {len(questions)} questions, {len(done)} complete', flush=True)
    if todo:
        with qwen_server.LlamaCppServer(out, config) as server, predictions.open('a', encoding='utf-8', buffering=1) as stream:
            for q in todo:
                history, selection = histories.get(q, server, config)
                old = base[q['example_id']]
                same = {m['message_id'] for m in old['selected_messages']} == {m['message_id'] for m in selection['selected_messages']}
                shifts = old['shifts'] if same else []
                for shift in range(0 if same else rotations):
                    p = server.score(build_messages(rotated(q, shift), config, history),
                                     request_id=f'{q["example_id"]}|shift{shift}', history_messages=len(history))
                    conditional = p['label_conditional_logprobs']
                    shifts.append(dict(shift=shift, predicted_label=p['predicted_label'], label_conditional_logprobs=conditional,
                                       answer_logprobs=answer_scores(conditional, labels, shift), prompt_tokens=p['prompt_tokens'],
                                       inference_seconds=p['inference_seconds'], timings=p.get('timings')))
                gold = labels.index(targets[q['example_id']])
                row = dict(example_id=q['example_id'], persona_id=q['persona_id'], question_type=q['question_type'],
                           gold_index=gold, predicted_label=shifts[0]['predicted_label'],
                           correct=shifts[0]['predicted_label'] == labels[gold], shifts=shifts, reused_baseline_read=same, **selection)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                rows.append(row)
                if len(rows) % 25 == 0:
                    print(f'{dataset} {variant}: {len(rows)}/{len(questions)} complete; {(time.perf_counter() - started) / 60:.1f} min', flush=True)
    if {r['example_id'] for r in rows} != set(data.qids):
        raise ValueError('Incomplete evaluation')
    report(out, dataset, rows, labels, variant)
    write_json(out / 'runtime.json', dict(seconds_this_process=time.perf_counter() - started, resumed_questions=len(done)))
    data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['B', 'C', 'B_tuned', 'C_tuned'], required=True,
                        help='B/C from neural_ranker.py; *_tuned from neural_ranker_tune.py')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=['v2', 'v1', 'locomo'])
    args = parser.parse_args()
    for dataset in args.datasets:
        score(dataset, args.variant)


if __name__ == '__main__':
    main()
