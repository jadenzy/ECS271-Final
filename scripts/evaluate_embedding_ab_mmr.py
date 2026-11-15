"""Frozen fine-tuned 57-feature ranker with redundancy-aware (MMR) packing.

python scripts/evaluate_embedding_ab_mmr.py --datasets v1
python scripts/evaluate_embedding_ab_mmr.py --datasets v1 --summarize-only

Same frozen encoder, adapter, ranker scores, Qwen prompt, 2,048-token budget and answer-order
policy as evaluate_embedding_ab_transfer.py; only the packing differs. Packing follows the
combined mathematical pilot (evaluate_math_ranker.pack_mmr): relevance is the ranker score
min-max normalized over the visible pool, and each next message maximizes relevance minus
penalty x its largest nonnegative cosine similarity (fine-tuned message vectors) to an
already accepted message. Whole messages only, exact native token costs, chronological display.
The penalty is fixed at the pilot's 0.1; nothing is tuned here.

Outputs under results/embedding_ab/transfer_mmr/{dataset}/: predictions.jsonl (resumable),
protocol.json, report.json, runtime.json and the server log. Compare with the no-MMR run in
results/embedding_ab/transfer/{dataset}/ using compare_selector_runs.py --rotation.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import build_ranker_features as builder
from embedding_ab_experiment import OUTPUT as TRAINING_RECORDS, verify_encoder
from embedding_ab_math import CachedInputs
from evaluate_embedding_ab_transfer import CACHE, DATASETS, MEMORY_TOKENS, MODEL_ROOT, accept_protocol, answer_scores
from evaluate_math_ranker import pack_mmr
from evaluate_qwen import (SelectorMessageBlocks, build_messages, evaluation_config, order_for_cache,
                           runtime_metadata, selected_history)
from option_rotation_pilot import bootstrap, rotated, summarize
from raw_datasets import ROOT, load_dataset, memories_for_question, sha256
from tune_ranker55 import write_json
import qwen_llamacpp_server as qwen_server

OUTPUT = ROOT / 'results/embedding_ab/transfer_mmr'
PENALTY = 0.1


def frozen_identity():
    """Read-only check that the cached scores come from the saved encoder, adapter and ranker."""
    info = verify_encoder()
    record = json.loads((TRAINING_RECORDS / 'finetuned_ranker57.json').read_text())
    for name, key in [('ranker57.txt', 'model_sha256'), ('adapter.npz', 'adapter_sha256')]:
        if sha256(MODEL_ROOT / name) != record[key]:
            raise ValueError('Frozen ranker/adapter changed')
    return dict(encoder=info, ranker_sha256=record['model_sha256'], adapter_sha256=record['adapter_sha256'],
                memory_tokens=MEMORY_TOKENS, redundancy_selection_used=True, mmr_penalty=PENALTY,
                mmr_similarity='cosine of the fine-tuned message embeddings', training_or_tuning=False), record


class MMRHistories:
    def __init__(self, questions, contexts, data, scores):
        self.data, self.scores = data, scores
        self.groups = {qid: i for i, qid in enumerate(data.qids)}
        if {q['example_id'] for q in questions} != set(data.qids):
            raise ValueError('Evaluation cohort differs from score cache')
        if scores.shape != (int(data.offsets[-1]),) or not np.isfinite(scores).all():
            raise ValueError('Invalid cached ranker scores')
        message_ids = data.bundle.read('message_ids')
        self.pools, self.blocks, self.similarities = {}, {}, {}
        for q in questions:
            key = q['history_id'], q['source_end_index']
            i = self.groups[q['example_id']]
            a, b = map(int, data.offsets[i:i + 2])
            members = memories_for_question(q, contexts)
            if [m['message_id'] for m in members] != message_ids[a:b].tolist():
                raise ValueError('Score rows differ from the raw visible message pool')
            if key not in self.pools:
                self.pools[key] = members
                self.blocks[key] = SelectorMessageBlocks(q['dataset'], members)
                vectors = np.asarray(data.memory[data.mi[a:b]], dtype=np.float32)
                self.similarities[key] = np.clip(vectors @ vectors.T, -1., 1.)

    def get(self, question, server, config):
        key = question['history_id'], question['source_end_index']
        i = self.groups[question['example_id']]
        a, b = map(int, self.data.offsets[i:i + 2])
        scores, members = np.asarray(self.scores[a:b], dtype=float), self.pools[key]
        chosen, packing = pack_mmr(server, question, config, self.blocks[key], scores,
                                   self.similarities[key], PENALTY, MEMORY_TOKENS)
        history = selected_history(members, chosen)
        return history, dict(**packing, selected_memory_count=len(history),
                             selected_messages=[dict(message_id=members[j]['message_id'], message_index=members[j]['message_index'],
                                                     score=float(scores[j]), selection_order=k) for k, j in enumerate(chosen)])


def report(dataset, rows, labels):
    split, rotations = DATASETS[dataset]
    if rotations > 1:
        result = summarize(rows, labels)
    else:
        correct = np.array([r['correct'] for r in rows], dtype=float)
        people = np.array([r['persona_id'] for r in rows])
        result = dict(questions=len(rows), personas=int(len(np.unique(people))), accuracy={'one_pass': float(correct.mean())},
                      ci95_persona_bootstrap=bootstrap(correct, people))
    result.update(status='complete', dataset=dataset, split=split, rotations=rotations, memory_tokens=MEMORY_TOKENS,
                  redundancy_selection_used=True, mmr_penalty=PENALTY,
                  mean_selected_messages=float(np.mean([r['selected_memory_count'] for r in rows])),
                  mean_memory_tokens=float(np.mean([r['packed_memory_tokens'] for r in rows])))
    write_json(OUTPUT / dataset / 'report.json', result)
    print(f'COMPLETE {dataset}: {result["questions"]} questions; {result["accuracy"]}', flush=True)


def score(dataset, summarize_only=False):
    started = time.perf_counter()
    frozen, record = frozen_identity()
    split, rotations = DATASETS[dataset]
    out = OUTPUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / 'predictions.jsonl'
    rows = [json.loads(line) for line in predictions.open(encoding='utf-8') if line.strip()] if predictions.exists() else []
    config = evaluation_config(dataset, 'selector')
    labels = config['answer_labels']
    if summarize_only:
        return report(dataset, rows, labels)
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
    histories = MMRHistories(questions, contexts, data, scores)
    if any([o['label'] for o in q['options']] != labels for q in questions):
        raise ValueError('Raw option labels differ from the configured scorer')
    settings = runtime_metadata(dataset, 'selector', memory_tokens=MEMORY_TOKENS)
    config.update(memory_tokens=MEMORY_TOKENS, input_policy='raw_complete_messages_v1', server_settings=settings,
                  model=settings['Model'], context_tokens=settings['ContextTokens'])
    protocol = dict(dataset=dataset, split=split, rotations=rotations, config=config, frozen_model=frozen,
                    scoring_sha256=sha256(path.parent / 'scoring.json'), raw_sources=sources, question_ids=data.qids,
                    questions_sha256=builder.fingerprint(questions), targets_sha256=builder.fingerprint(targets),
                    script_sha256=sha256(Path(__file__)), scorer_sha256=sha256(Path(qwen_server.__file__)),
                    selection_policy=f'MMR (penalty {PENALTY}) under native token budget; whole messages, chronological display',
                    uses_evaluation_labels_for_selection=False)
    done = accept_protocol(out / 'protocol.json', protocol, rows, data.qids)
    todo = [q for q in questions if q['example_id'] not in done]
    print(f'SCORING {dataset} with MMR {PENALTY}: {len(questions)} questions, {len(done)} complete, {rotations} answer order(s)', flush=True)
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
    report(dataset, rows, labels)
    write_json(out / 'runtime.json', dict(seconds_this_process=time.perf_counter() - started, resumed_questions=len(done)))
    data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=['v1'])
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    for dataset in args.datasets:
        score(dataset, args.summarize_only)


if __name__ == '__main__':
    main()
