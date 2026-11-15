"""Frozen fine-tuned 57-feature ranker with a relative score cutoff on the packed selection.

python scripts/evaluate_embedding_ab_cutoff.py --datasets v1 --cutoff 3.0
python scripts/evaluate_embedding_ab_cutoff.py --datasets v1 --cutoff 3.0 --summarize-only

Same frozen encoder, adapter, ranker scores, Qwen prompt, 2,048-token packing and answer-order
policy as evaluate_embedding_ab_transfer.py. After packing, only the messages scoring within
--cutoff of the question's top score are shown (the chosen messages keep chronological order).
The rule needs no Qwen judgment and reads no labels. The distance was fixed from a coverage
analysis on V2 validation (reader_validation.py's filter_cut3); nothing is tuned here.

Outputs under results/embedding_ab/transfer_cutoff/{dataset}_cut{distance}/: predictions.jsonl
(resumable; one row per question, its predicted_label/correct fields are answer order 0),
protocol.json, report.json, runtime.json and the server log. Compare with the unfiltered run in
results/embedding_ab/transfer/{dataset}/ using compare_selector_runs.py --rotation.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import build_ranker_features as builder
from embedding_ab_experiment import verify_encoder
from embedding_ab_math import CachedInputs
from evaluate_embedding_ab_transfer import (CACHE, DATASETS, MEMORY_TOKENS, MODEL_ROOT, FrozenHistories, accept_protocol,
                                            answer_scores, frozen_models)
from evaluate_qwen import build_messages, evaluation_config, order_for_cache, runtime_metadata, selected_history
from option_rotation_pilot import bootstrap, rotated, summarize
from raw_datasets import ROOT, load_dataset, sha256
from tune_ranker55 import write_json
import qwen_llamacpp_server as qwen_server

OUTPUT = ROOT / 'results/embedding_ab/transfer_cutoff'


class CutoffHistories(FrozenHistories):
    def __init__(self, questions, contexts, data, scores, cutoff):
        super().__init__(questions, contexts, data, scores)
        self.cutoff = cutoff

    def get(self, question, server, config):
        history, selection = super().get(question, server, config)
        selected = selection['selected_messages']
        top = max(s['score'] for s in selected)
        kept_ids = {s['message_id'] for s in selected if s['score'] >= top - self.cutoff}
        members = self.pools[question['history_id'], question['source_end_index']]
        history = selected_history(members, [j for j, m in enumerate(members) if m['message_id'] in kept_ids])
        kept = [s for s in selected if s['message_id'] in kept_ids]
        return history, dict(selection, selected_memory_count=len(history), selected_messages=kept,
                             messages_before_cutoff=len(selected), packed_memory_tokens_before_cutoff=selection['packed_memory_tokens'])


def report(out, dataset, rows, labels, cutoff):
    split, rotations = DATASETS[dataset]
    result = summarize(rows, labels) if rotations > 1 else dict(
        questions=len(rows), accuracy={'one_pass': float(np.mean([r['correct'] for r in rows]))})
    result.update(status='complete', dataset=dataset, split=split, rotations=rotations, memory_tokens=MEMORY_TOKENS, score_cutoff=cutoff,
                  mean_selected_messages=float(np.mean([r['selected_memory_count'] for r in rows])),
                  mean_messages_before_cutoff=float(np.mean([r['messages_before_cutoff'] for r in rows])))
    write_json(out / 'report.json', result)
    print(f'COMPLETE {dataset}: {result["questions"]} questions; {result["accuracy"]}', flush=True)


def score(dataset, cutoff, summarize_only=False):
    started = time.perf_counter()
    _, record, frozen = frozen_models()
    split, rotations = DATASETS[dataset]
    out = OUTPUT / f'{dataset}_cut{cutoff:g}'
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / 'predictions.jsonl'
    rows = [json.loads(line) for line in predictions.open(encoding='utf-8') if line.strip()] if predictions.exists() else []
    config = evaluation_config(dataset, 'selector')
    labels = config['answer_labels']
    if summarize_only:
        return report(out, dataset, rows, labels, cutoff)
    questions, contexts, targets, sources = load_dataset(dataset, [split], root=ROOT)
    questions = order_for_cache(questions)
    path = CACHE / dataset / split / 'features.npz'
    scoring = json.loads((path.parent / 'scoring.json').read_text())
    if (scoring['ranker_sha256'] != frozen['ranker_sha256'] or scoring['adapter_sha256'] != frozen['adapter_sha256']
            or scoring['features_sha256'] != sha256(path) or scoring['scores_sha256'] != sha256(path.parent / 'ranker_scores.npy')):
        raise ValueError('Feature/score cache differs from the frozen models')
    data = CachedInputs(path)
    if data.meta['embedding_recipe'] != record['embedding_recipe']:
        raise ValueError('Feature encoder differs from the trained ranker')
    for source, expected in sources.items():
        if data.meta['source_sha256'].get(source) != expected:
            raise ValueError('Current raw source or split differs from the feature cache')
    scores = np.load(path.parent / 'ranker_scores.npy', mmap_mode='r')
    histories = CutoffHistories(questions, contexts, data, scores, cutoff)
    if any([o['label'] for o in q['options']] != labels for q in questions):
        raise ValueError('Raw option labels differ from the configured scorer')
    settings = runtime_metadata(dataset, 'selector', memory_tokens=MEMORY_TOKENS)
    config.update(memory_tokens=MEMORY_TOKENS, input_policy='raw_complete_messages_v1', server_settings=settings,
                  model=settings['Model'], context_tokens=settings['ContextTokens'])
    protocol = dict(dataset=dataset, split=split, rotations=rotations, config=config, frozen_model=frozen, score_cutoff=cutoff,
                    scoring_sha256=sha256(path.parent / 'scoring.json'), raw_sources=sources, question_ids=data.qids,
                    questions_sha256=builder.fingerprint(questions), targets_sha256=builder.fingerprint(targets),
                    script_sha256=sha256(Path(__file__)), scorer_sha256=sha256(Path(qwen_server.__file__)),
                    selection_policy=f'greedy rank order under native token budget, then keep messages within {cutoff:g} of the top score; '
                                     'whole messages, chronological display',
                    uses_evaluation_labels_for_selection=False)
    done = accept_protocol(out / 'protocol.json', protocol, rows, data.qids)
    todo = [q for q in questions if q['example_id'] not in done]
    print(f'SCORING {dataset} with score cutoff {cutoff:g}: {len(questions)} questions, {len(done)} complete, {rotations} answer order(s)', flush=True)
    if todo:
        with qwen_server.LlamaCppServer(out, config) as server, predictions.open('a', encoding='utf-8', buffering=1) as stream:
            for q in todo:
                history, selection = histories.get(q, server, config)
                shifts = []
                for shift in range(rotations):
                    p = server.score(build_messages(rotated(q, shift), config, history),
                                     request_id=f'{q["example_id"]}|shift{shift}', history_messages=len(history))
                    conditional = p['label_conditional_logprobs']
                    shifts.append(dict(shift=shift, predicted_label=p['predicted_label'], label_conditional_logprobs=conditional,
                                       answer_logprobs=answer_scores(conditional, labels, shift), prompt_tokens=p['prompt_tokens'],
                                       inference_seconds=p['inference_seconds'], timings=p.get('timings')))
                # Targets are read only after memory selection and all answer scoring.
                gold = labels.index(targets[q['example_id']])
                row = dict(example_id=q['example_id'], persona_id=q['persona_id'], question_type=q['question_type'],
                           gold_index=gold, predicted_label=shifts[0]['predicted_label'],
                           correct=shifts[0]['predicted_label'] == labels[gold], shifts=shifts, **selection)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                rows.append(row)
                if len(rows) % 25 == 0:
                    print(f'{dataset}: {len(rows)}/{len(questions)} complete; {(time.perf_counter() - started) / 60:.1f} min', flush=True)
    if {r['example_id'] for r in rows} != set(data.qids):
        raise ValueError('Incomplete evaluation')
    report(out, dataset, rows, labels, cutoff)
    write_json(out / 'runtime.json', dict(seconds_this_process=time.perf_counter() - started, resumed_questions=len(done)))
    data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=['v1'])
    parser.add_argument('--cutoff', type=float, default=3.0, help='keep messages within this distance of the top score')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    for dataset in args.datasets:
        score(dataset, args.cutoff, args.summarize_only)


if __name__ == '__main__':
    main()
