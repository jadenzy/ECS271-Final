"""Evaluate the frozen combined mathematical pilot on active V2 test only.

python scripts/evaluate_math_ranker.py
python scripts/evaluate_math_ranker.py --summarize-only
Uses the existing native Qwen scorer and four-order aggregation. Resumable.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from math_ranker_experiment import (CACHE, MODEL_ROOT, NAMES, OUTPUT, Inputs, extra_features,
                                    mmr_choice, normalized_scores, validate_cosine, vectors_on_gpu)
from raw_datasets import ROOT, load_dataset, memories_for_question, sha256
from evaluate_qwen import (SelectorMessageBlocks, build_messages, evaluation_config,
                           order_for_cache, runtime_metadata, selected_history)
from option_rotation_pilot import bootstrap, rotated, summarize
from tune_ranker55 import write_json
import qwen_llamacpp_server as qwen_server


def pack_mmr(server, question, config, blocks, scores, similarities, penalty, budget):
    """MMR updates only after acceptance; exact template costs preserve whole messages."""
    bare = build_messages(question, config, [])
    _, bare_tokens = server.render_prompt(bare)
    context = min(getattr(server, 'actual_context', None) or config['context_tokens'], config['context_tokens'])
    effective = min(budget, context - bare_tokens - 1)
    if effective < 0:
        raise ValueError('Question/options exceed context')
    relevance = normalized_scores(scores)
    available = np.ones(len(scores), dtype=bool)
    redundancy = np.zeros(len(scores))
    chosen, skipped, used = [], 0, 0
    final_messages, final_rendered = bare, server.render_prompt(bare)
    while available.any():
        index = mmr_choice(relevance, redundancy, available, penalty)
        available[index] = False
        messages = blocks.messages(chosen + [index], bare)
        rendered = server.render_prompt(messages)
        cost = rendered[1] - bare_tokens
        if cost < 0:
            raise ValueError('Negative native token cost')
        if cost <= effective:
            chosen.append(index)
            used = cost
            redundancy = np.maximum(redundancy, np.maximum(0., similarities[:, index]))
            final_messages, final_rendered = messages, rendered
        else:
            skipped += 1
    server.verify_selector_prompt(final_messages, final_rendered, bare, server.render_prompt(bare))
    return chosen, dict(packed_memory_tokens=used, memory_token_budget=budget,
                        effective_memory_token_budget=effective, budget_skipped_messages=skipped)


class Histories:
    def __init__(self, questions, contexts, data, scores, cfg):
        self.cfg, self.scores, self.data = cfg, scores, data
        self.groups = {qid: i for i, qid in enumerate(data.qids)}
        self.pools, self.blocks, self.similarities = {}, {}, {}
        for q in questions:
            key = q['history_id'], q['source_end_index']
            i = self.groups[q['example_id']]
            a, b = map(int, data.offsets[i:i+2])
            members = memories_for_question(q, contexts)
            if [m['message_id'] for m in members] != data.bundle.read('message_ids')[a:b].tolist():
                raise ValueError('Raw history and cached score rows differ')
            if key not in self.pools:
                self.pools[key] = members
                self.blocks[key] = SelectorMessageBlocks('v2', members)
                vectors = np.asarray(data.memory)[data.mi[a:b]]
                self.similarities[key] = np.clip(vectors @ vectors.T, -1., 1.)

    def get(self, question, server, config):
        key = question['history_id'], question['source_end_index']
        group = self.groups[question['example_id']]
        a, b = map(int, self.data.offsets[group:group+2])
        scores, members = self.scores[a:b], self.pools[key]
        chosen, packing = pack_mmr(server, question, config, self.blocks[key], scores,
                                  self.similarities[key], self.cfg['mmr_penalty'], self.cfg['memory_tokens'])
        history = selected_history(members, chosen)
        return history, {**packing, 'selected_memory_count': len(history),
                         'selected_messages': [{'message_id': members[i]['message_id'],
                                                'score': float(scores[i]), 'selection_order': j}
                                               for j, i in enumerate(chosen)]}


def prepare(data, training):
    import torch
    import lightgbm as lgb
    cfg = training['settings']
    if data.meta['feature_recipe'] != training['feature_recipe']:
        raise ValueError('Train/test lexical feature recipes differ')
    if set(data.personas) & set(training['train_personas']):
        raise ValueError('Test personas overlap training')
    for name, key in [('ranker57.txt', 'model_sha256'), ('adapter.npz', 'adapter_sha256')]:
        if sha256(MODEL_ROOT / name) != training[key]:
            raise ValueError('Trained artifact hash mismatch')
    memory, query = vectors_on_gpu(data)
    alignment = validate_cosine(data, memory, query)
    with np.load(MODEL_ROOT / 'adapter.npz') as adapter:
        U, V, bank = [torch.tensor(adapter[k], device='cuda') for k in ['U', 'V', 'bank']]
    extra = np.full((len(data.X), 2), np.nan, dtype=np.float32)
    extra_features(data, memory, query, U, V, bank, np.arange(len(data.qids)), cfg, extra)
    del memory, query, U, V, bank
    gc.collect()
    torch.cuda.empty_cache()
    if not np.isfinite(extra).all():
        raise ValueError('Nonfinite test features')
    np.save(CACHE / 'test_extra.npy', extra)
    booster = lgb.Booster(model_file=str(MODEL_ROOT / 'ranker57.txt'))
    if booster.feature_name() != NAMES:
        raise ValueError('Wrong experimental model feature order')
    scores = booster.predict(np.column_stack([data.X, extra]), num_threads=4)
    np.save(CACHE / 'test_scores.npy', scores)
    return scores, alignment


def report(rows):
    baseline_path = ROOT / 'results/option_rotation/v2/test/references_only/predictions.jsonl'
    baseline = {r['example_id']: r for r in map(json.loads, baseline_path.open(encoding='utf-8'))}
    ids = [r['example_id'] for r in rows]
    active = json.loads((ROOT / 'data/splits/v2/test_ids.json').read_text())
    if set(ids) != set(active) or len(ids) != len(active):
        raise ValueError('Final report requires every active test question exactly once')
    groups = np.array([r['persona_id'] for r in rows])
    paired = {}
    for rotation in [False, True]:
        def correct(r):
            pred = int(np.sum([s['answer_logprobs'] for s in r['shifts']], axis=0).argmax()) if rotation else int(np.argmax(r['shifts'][0]['answer_logprobs']))
            return int(pred == r['gold_index'])
        old = np.array([correct(baseline[q]) for q in ids])
        new = np.array([correct(r) for r in rows])
        diff = new - old
        paired['four_orders' if rotation else 'one_pass'] = dict(
            baseline_accuracy=float(old.mean()), combined_accuracy=float(new.mean()), difference=float(diff.mean()),
            ci95_persona_bootstrap=bootstrap(diff, groups), gained=int((diff > 0).sum()), lost=int((diff < 0).sum()))
    result = dict(status='complete', dataset='v2', split='test', **summarize(rows, list('abcd')),
                  paired_against_reference_ranker=paired,
                  mean_selected_messages=float(np.mean([r['selected_memory_count'] for r in rows])),
                  mean_memory_tokens=float(np.mean([r['packed_memory_tokens'] for r in rows])),
                  caveat='One fixed combined experiment; component contributions are not isolated. No test tuning.')
    write_json(OUTPUT / 'report.json', result)
    print(json.dumps(paired, indent=2), flush=True)
    # Reference annotations enter only this post-inference diagnostic.
    from tune_ranker55 import aggregate, score_questions, validation_targets
    import lightgbm as lgb
    with BundleForReport() as data:
        targets = validation_targets(data, ROOT, 'test')
        baseline_scores = lgb.Booster(model_file=str(ROOT / 'models/subtlex_mix/alpha_0.5/ranker55.txt')).predict(data.read('X'), num_threads=4)
        new_scores = np.load(CACHE / 'test_scores.npy')
        diagnostic = {name: aggregate(score_questions(s, data.read('X')[:, 0], targets))
                      for name, s in [('baseline', baseline_scores), ('combined_ranker_before_mmr', new_scores)]}
        mids, offsets = data.read('message_ids'), targets['offsets']
        coverage = []
        by_qid = {r['example_id']: r for r in rows}
        for i, q in enumerate(targets['questions']):
            a, b = map(int, offsets[i:i+2])
            selected = {s['message_id'] for s in by_qid[q['example_id']]['selected_messages']}
            positive = mids[a:b][targets['positive'][a:b]].tolist()
            found = sum(mid in selected for mid in positive)
            coverage.append(dict(example_id=q['example_id'], reference_count=len(positive), selected_reference_count=found,
                                 recall=found/max(1, len(positive)), every_visible_reference=found == len(positive)))
        diagnostic['combined_selection'] = dict(reference_recall=float(np.mean([r['recall'] for r in coverage])),
                                                every_visible_reference=float(np.mean([r['every_visible_reference'] for r in coverage])))
        write_json(OUTPUT / 'retrieval.json', diagnostic)


def BundleForReport():
    from tune_ranker55 import Bundle
    return Bundle(ROOT / 'data/ranker55/v2/test/features.npz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    predictions = OUTPUT / 'predictions.jsonl'
    rows = [json.loads(line) for line in predictions.open(encoding='utf-8')] if predictions.exists() else []
    if args.summarize_only:
        report(rows)
        return
    started = time.perf_counter()
    training = json.loads((OUTPUT / 'training.json').read_text())
    cfg = training['settings']
    questions, contexts, targets, sources = load_dataset('v2', ['test'], root=ROOT)
    questions = order_for_cache(questions)
    data = Inputs('test')
    scores, alignment = prepare(data, training)
    histories = Histories(questions, contexts, data, scores, cfg)
    config = evaluation_config('v2', 'selector')
    settings = runtime_metadata('v2', 'selector', memory_tokens=cfg['memory_tokens'])
    config.update(memory_tokens=cfg['memory_tokens'], input_policy='raw_complete_messages_v1', server_settings=settings,
                  model=settings['Model'], context_tokens=settings['ContextTokens'])
    protocol = dict(settings=cfg, model_sha256=training['model_sha256'], adapter_sha256=training['adapter_sha256'],
                    test_features_sha256=sha256(data.path), question_ids=data.qids, sources=sources,
                    embedding_cache_sha256=data.cache_hashes, cosine_alignment_max_error=alignment,
                    config=config, script_sha256=sha256(Path(__file__)), selection_uses_labels_or_answers=False,
                    training_record_sha256=sha256(OUTPUT / 'training.json'))
    protocol_path = OUTPUT / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Resume protocol differs; refusing to mix test predictions')
    write_json(protocol_path, protocol)
    done = {r['example_id'] for r in rows}
    if len(done) != len(rows) or not done <= set(data.qids):
        raise ValueError('Duplicate or foreign resumed questions')
    todo = [q for q in questions if q['example_id'] not in done]
    if todo:
        with qwen_server.LlamaCppServer(OUTPUT, config) as server, predictions.open('a', encoding='utf-8', buffering=1) as stream:
            for q in todo:
                history, selection = histories.get(q, server, config)
                shifts = []
                for shift in range(4):
                    p = server.score(build_messages(rotated(q, shift), config, history),
                                     request_id=f'{q["example_id"]}|shift{shift}', history_messages=len(history))
                    conditional = p['label_conditional_logprobs']
                    answers = [conditional['abcd'[(a - shift) % 4]] for a in range(4)]
                    if not np.isfinite(answers).all():
                        raise ValueError('Nonfinite answer scores')
                    shifts.append(dict(shift=shift, predicted_label=p['predicted_label'], label_conditional_logprobs=conditional,
                                       answer_logprobs=answers, prompt_tokens=p['prompt_tokens'], inference_seconds=p['inference_seconds'],
                                       timings=p.get('timings')))
                gold = 'abcd'.index(targets[q['example_id']])
                row = dict(example_id=q['example_id'], persona_id=q['persona_id'], question_type=q['question_type'],
                           gold_index=gold, predicted_label=shifts[0]['predicted_label'],
                           correct=shifts[0]['predicted_label'] == 'abcd'[gold], shifts=shifts, **selection)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                rows.append(row)
                if len(rows) % 25 == 0:
                    print(f'{len(rows)}/{len(questions)} scored; {(time.perf_counter()-started)/60:.1f} minutes', flush=True)
    report(rows)


if __name__ == '__main__':
    main()
