"""Frozen fine-tuned 57-feature ranker with a relative score cutoff on the packed selection.

python scripts/evaluate_embedding_ab_cutoff.py --datasets v1 --cutoff 3.0
python scripts/evaluate_embedding_ab_cutoff.py --datasets v1 --cutoff 3.0 --memory-tokens 8192
python scripts/evaluate_embedding_ab_cutoff.py --datasets v1 --cutoff 3.0 --summarize-only
python scripts/evaluate_embedding_ab_cutoff.py --datasets v2 v1 locomo --cutoff 3.0 --memory-tokens 8192 --neighbour-margin 5.0

--neighbour-margin M (capped path only): references in V2 sit in one consecutive run, so after the cutoff
packing each kept message's chronological neighbours (message index +-1) are added when their own ranker
score is within M of the top, and the set is repacked under the same cap. Questions whose set is unchanged
reuse the plain cutoff run's reads (no GPU drift). Outputs: {dataset}_cut{D}_mem{cap}_nb{M}/. Result at
M = 5.0 (chosen by a CPU projection on V2 validation): V2 52.10% (-2.14), V1 68.42% (+2.72), LoCoMo
69.89% (-0.50) against the plain 8,192-cap cutoff; not the default.

Same frozen encoder, adapter, ranker scores, Qwen prompt and answer-order policy as
evaluate_embedding_ab_transfer.py. With the default 2,048-token budget, the selection is packed
exactly as there and then only the messages scoring within --cutoff of the question's top score
are shown. With a larger --memory-tokens cap, packing walks the whole visible pool in rank order,
accepts every whole message within --cutoff of the top score that still fits the cap, and stops
at the first message below it (evaluate_qwen.pack_selector_messages's score_cutoff rule); the
server context grows with the cap. The 2,048 path is the same selection as packing to 2,048 and
filtering, since greedy rank-order packing reaches every in-cutoff message first. Either way the
chosen messages keep chronological order, no Qwen judgment is used and no labels are read. The
distance was fixed from a coverage analysis on V2 validation (reader_validation.py's
filter_cut3); nothing is tuned here.

Outputs under results/embedding_ab/transfer_cutoff/{dataset}_cut{distance}/ (with a larger cap:
{dataset}_cut{distance}_mem{cap}/): predictions.jsonl (resumable; one row per question, its
predicted_label/correct fields are answer order 0), protocol.json, report.json, runtime.json and
the server log. Compare with the unfiltered run in results/embedding_ab/transfer/{dataset}/
using compare_selector_runs.py --rotation.
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
from evaluate_qwen import (build_messages, evaluation_config, order_for_cache, pack_selector_messages, runtime_metadata,
                           selected_history)
from option_rotation_pilot import bootstrap, rotated, summarize
from raw_datasets import ROOT, load_dataset, sha256
from tune_ranker55 import write_json
import qwen_llamacpp_server as qwen_server

OUTPUT = ROOT / 'results/embedding_ab/transfer_cutoff'


class CutoffHistories(FrozenHistories):
    def __init__(self, questions, contexts, data, scores, cutoff, budget=MEMORY_TOKENS, neighbour_margin=None):
        super().__init__(questions, contexts, data, scores)
        self.cutoff, self.budget, self.neighbour_margin = cutoff, budget, neighbour_margin

    def get(self, question, server, config):
        if self.budget != MEMORY_TOKENS:
            return self.get_capped(question, server, config)
        history, selection = super().get(question, server, config)
        selected = selection['selected_messages']
        top = max(s['score'] for s in selected)
        kept_ids = {s['message_id'] for s in selected if s['score'] >= top - self.cutoff}
        members = self.pools[question['history_id'], question['source_end_index']]
        history = selected_history(members, [j for j, m in enumerate(members) if m['message_id'] in kept_ids])
        kept = [s for s in selected if s['message_id'] in kept_ids]
        return history, dict(selection, selected_memory_count=len(history), selected_messages=kept,
                             messages_before_cutoff=len(selected), packed_memory_tokens_before_cutoff=selection['packed_memory_tokens'])

    def get_capped(self, question, server, config):
        key = question['history_id'], question['source_end_index']
        members = self.pools[key]
        i = self.groups[question['example_id']]
        a, b = map(int, self.data.offsets[i:i+2])
        scores = np.asarray(self.scores[a:b], dtype=float)
        ranked = np.argsort(-scores, kind='stable').tolist()
        chosen, packing = pack_selector_messages(server, question, config, members, ranked, self.budget,
                                                 blocks=self.blocks[key], scores=scores, score_cutoff=self.cutoff)
        expansion = {}
        if self.neighbour_margin is not None:
            # References sit in one consecutive run: add a chronological neighbour (message index +-1) of a kept
            # message when its own score is within neighbour_margin of the top, then repack under the same cap.
            at = {int(m['message_index']): j for j, m in enumerate(members)}
            lowest, kept = scores.max() - self.neighbour_margin, set(chosen)
            extras = sorted({n for j in chosen for o in (-1, 1)
                             if (n := at.get(int(members[j]['message_index']) + o)) is not None and n not in kept and scores[n] >= lowest},
                            key=lambda n: (-scores[n], n))
            if extras:
                chosen, packing = pack_selector_messages(server, question, config, members, chosen + extras, self.budget,
                                                         blocks=self.blocks[key])
            expansion = dict(neighbours_added=len(set(chosen) - kept), neighbour_candidates=len(extras))
        history = selected_history(members, chosen)
        return history, dict(**packing, **expansion, selected_memory_count=len(history),
                             messages_within_cutoff_in_pool=int((scores >= scores.max() - self.cutoff).sum()),
                             selected_messages=[dict(message_id=members[j]['message_id'], message_index=members[j]['message_index'],
                                                     score=float(scores[j])) for j in chosen])


def report(out, dataset, rows, labels, cutoff, budget=MEMORY_TOKENS):
    split, rotations = DATASETS[dataset]
    result = summarize(rows, labels) if rotations > 1 else dict(
        questions=len(rows), accuracy={'one_pass': float(np.mean([r['correct'] for r in rows]))})
    result.update(status='complete', dataset=dataset, split=split, rotations=rotations, memory_tokens=budget, score_cutoff=cutoff,
                  mean_selected_messages=float(np.mean([r['selected_memory_count'] for r in rows])))
    if budget == MEMORY_TOKENS:
        result['mean_messages_before_cutoff'] = float(np.mean([r['messages_before_cutoff'] for r in rows]))
    else:
        result.update(mean_memory_tokens=float(np.mean([r['packed_memory_tokens'] for r in rows])),
                      mean_messages_within_cutoff_in_pool=float(np.mean([r['messages_within_cutoff_in_pool'] for r in rows])),
                      share_questions_capped=float(np.mean([r['budget_skipped_messages'] > 0 for r in rows])))
    if rows and 'neighbours_added' in rows[0]:
        result.update(neighbour_margin=rows[0].get('neighbour_margin'),
                      share_questions_expanded=float(np.mean([r['neighbours_added'] > 0 for r in rows])),
                      mean_neighbours_added=float(np.mean([r['neighbours_added'] for r in rows])),
                      share_reused_cutoff_reads=float(np.mean([r['reused_cutoff_read'] for r in rows])))
    write_json(out / 'report.json', result)
    print(f'COMPLETE {dataset}: {result["questions"]} questions; {result["accuracy"]}', flush=True)


def score(dataset, cutoff, summarize_only=False, budget=MEMORY_TOKENS, neighbour_margin=None):
    started = time.perf_counter()
    _, record, frozen = frozen_models()
    if budget != MEMORY_TOKENS:
        frozen = dict(frozen, memory_tokens=budget)
    if neighbour_margin is not None and budget == MEMORY_TOKENS:
        raise ValueError('Neighbour expansion is implemented for the capped path only (--memory-tokens above 2,048)')
    split, rotations = DATASETS[dataset]
    base = OUTPUT / (f'{dataset}_cut{cutoff:g}' if budget == MEMORY_TOKENS else f'{dataset}_cut{cutoff:g}_mem{budget}')
    out = base if neighbour_margin is None else base.with_name(f'{base.name}_nb{neighbour_margin:g}')
    out.mkdir(parents=True, exist_ok=True)
    reuse = {}
    if neighbour_margin is not None:
        # Questions whose expanded set equals the plain cutoff set reuse that run's reads, so the comparison has no GPU drift.
        reuse_path = base / 'predictions.jsonl'
        reuse = {r['example_id']: r for r in (json.loads(line) for line in reuse_path.open(encoding='utf-8') if line.strip())}
    predictions = out / 'predictions.jsonl'
    rows = [json.loads(line) for line in predictions.open(encoding='utf-8') if line.strip()] if predictions.exists() else []
    config = evaluation_config(dataset, 'selector')
    labels = config['answer_labels']
    if summarize_only:
        return report(out, dataset, rows, labels, cutoff, budget)
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
    histories = CutoffHistories(questions, contexts, data, scores, cutoff, budget, neighbour_margin)
    if any([o['label'] for o in q['options']] != labels for q in questions):
        raise ValueError('Raw option labels differ from the configured scorer')
    settings = runtime_metadata(dataset, 'selector', memory_tokens=budget)
    config.update(memory_tokens=budget, input_policy='raw_complete_messages_v1', server_settings=settings,
                  model=settings['Model'], context_tokens=settings['ContextTokens'])
    protocol = dict(dataset=dataset, split=split, rotations=rotations, config=config, frozen_model=frozen, score_cutoff=cutoff,
                    scoring_sha256=sha256(path.parent / 'scoring.json'), raw_sources=sources, question_ids=data.qids,
                    questions_sha256=builder.fingerprint(questions), targets_sha256=builder.fingerprint(targets),
                    script_sha256=sha256(Path(__file__)), scorer_sha256=sha256(Path(qwen_server.__file__)),
                    selection_policy=(f'greedy rank order under native token budget, then keep messages within {cutoff:g} of the top score; '
                                      'whole messages, chronological display') if budget == MEMORY_TOKENS else
                                     (f'greedy rank order over the whole visible pool, accepting messages within {cutoff:g} of the top score '
                                      f'that fit a {budget:,}-token cap, stopping at the first message below; whole messages, chronological display'),
                    uses_evaluation_labels_for_selection=False)
    if neighbour_margin is not None:
        protocol.update(neighbour_margin=neighbour_margin,
                        selection_policy=protocol['selection_policy'] + (
                            f'; then add each kept message\'s chronological neighbours (message index +-1) scoring within '
                            f'{neighbour_margin:g} of the top, repacked under the same cap'),
                        reused_cutoff_predictions=str(reuse_path.relative_to(ROOT)), reused_cutoff_predictions_sha256=sha256(reuse_path),
                        reused_cutoff_protocol_sha256=sha256(base / 'protocol.json'))
    done = accept_protocol(out / 'protocol.json', protocol, rows, data.qids)
    todo = [q for q in questions if q['example_id'] not in done]
    print(f'SCORING {dataset} with score cutoff {cutoff:g}, {budget:,}-token cap: {len(questions)} questions, {len(done)} complete, '
          f'{rotations} answer order(s)', flush=True)
    if todo:
        with qwen_server.LlamaCppServer(out, config) as server, predictions.open('a', encoding='utf-8', buffering=1) as stream:
            for q in todo:
                history, selection = histories.get(q, server, config)
                shifts = []
                if neighbour_margin is not None:
                    old = reuse.get(q['example_id'])
                    same = (old is not None and [m['message_id'] for m in old['selected_messages']]
                            == [m['message_id'] for m in selection['selected_messages']])
                    selection = dict(selection, neighbour_margin=neighbour_margin, reused_cutoff_read=same)
                    if same:
                        shifts = old['shifts']
                for shift in range(rotations if not shifts else 0):
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
    report(out, dataset, rows, labels, cutoff, budget)
    write_json(out / 'runtime.json', dict(seconds_this_process=time.perf_counter() - started, resumed_questions=len(done)))
    data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=['v1'])
    parser.add_argument('--cutoff', type=float, default=3.0, help='keep messages within this distance of the top score')
    parser.add_argument('--memory-tokens', type=int, default=MEMORY_TOKENS,
                        help='token cap; above 2,048 the cutoff decides the size and packing stops at it')
    parser.add_argument('--neighbour-margin', type=float, default=None,
                        help='also add chronological neighbours of kept messages scoring within this distance of the top')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.memory_tokens < 1:
        parser.error('--memory-tokens must be positive')
    for dataset in args.datasets:
        score(dataset, args.cutoff, args.summarize_only, args.memory_tokens, args.neighbour_margin)


if __name__ == '__main__':
    main()
