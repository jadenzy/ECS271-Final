"""Reader-side experiments on V2 validation: prompt wording and a Qwen filter step.

python scripts/reader_validation.py --arms oracle_current oracle_clean ranker_current ranker_clean
python scripts/reader_validation.py --stage select                     # save the ranker's selection per question
python scripts/reader_validation.py --stage judge                      # grade each ranker-selected message
python scripts/reader_validation.py --stage policies                   # reference coverage of filter rules (CPU)
python scripts/reader_validation.py --arms ranker_clean filter_useful_clean
python scripts/reader_validation.py --summarize-only

Validation only (909 questions; personas disjoint from the encoder's A cohort and the ranker's B
cohort); the test split is never read. Every arm of a question is scored on the same Qwen server
with four cyclic answer orders, so arms are paired and free of GPU drift.

Memory sources
  oracle   the annotated reference snippets (ReferenceHistories)
  ranker   the frozen fine-tuned 57-feature ranker's 2,048-token selection (saved validation scores)
  filter_* a subset of the ranker's selection chosen from Qwen's per-message judgment
           (label_reference_messages' judge: question and one message, useful or not; no options)
Templates
  current  configs/evaluate_qwen.json: the question message starts with an empty "Memories:" heading
  clean    the same message without that heading

Outputs: results/reader_validation/{arm}/predictions.jsonl (option_rotation_pilot row format, so
compare_selector_runs.py --rotation applies), judgments.jsonl, policies.json and report.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from embedding_ab_math import CachedInputs
from evaluate_embedding_ab_transfer import FrozenHistories, answer_scores
from evaluate_qwen import ReferenceHistories, build_messages, evaluation_config, order_for_cache, runtime_metadata, selected_history
from label_reference_messages import judge_candidate, judge_config
from option_rotation_pilot import bootstrap, rotated, summarize
from raw_datasets import ROOT, load_dataset
import qwen_llamacpp_server as qwen_server

OUTPUT = ROOT / 'results/reader_validation'
FEATURES = ROOT / 'data/ranker55/embedding_ab/finetuned/v2/val/features.npz'
SCORES = ROOT / 'models/embedding_ab/finetuned_math/validation_scores.npy'
MEMORY_TOKENS = 2048
CLEAN_TEMPLATE = 'Question: {question}\nOptions:\n{options}\nAnswer:'


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def load_selections():
    return {r['example_id']: r for r in read_rows(OUTPUT / 'selections.jsonl')}


def select():
    """Pack the ranker's 2,048-token selection for every validation question (no answer scoring)."""
    questions, _, _, config, ranker, _ = setup()
    done = load_selections()
    todo = [q for q in questions if q['example_id'] not in done]
    print(f'Selecting for {len(todo)} of {len(questions)} questions', flush=True)
    if todo:
        (OUTPUT / 'select').mkdir(parents=True, exist_ok=True)
        stream = (OUTPUT / 'selections.jsonl').open('a', encoding='utf-8', buffering=1)
        with qwen_server.LlamaCppServer(OUTPUT / 'select', config) as server, stream:
            for q in todo:
                _, selection = ranker.get(q, server, config)
                stream.write(json.dumps({'example_id': q['example_id'], 'selected_messages': selection['selected_messages'],
                                         'packed_memory_tokens': selection['packed_memory_tokens']}) + chr(10))
    print(f'Selections saved for {len(load_selections())} questions', flush=True)


def load_judgments():
    return {r['example_id']: r['messages'] for r in read_rows(OUTPUT / 'judgments.jsonl')}


# Filter rules: each returns the kept message ids, given the ranker's selection (best score first)
# and Qwen's judgment per message: {'useful': log p(a), 'not_useful': log p(b)}.
def keep_useful(selection, judged, floor=2):
    """Messages Qwen calls useful; if fewer than `floor`, fill with the ranker's best."""
    kept = [s['message_id'] for s in selection if judged[s['message_id']]['useful'] > judged[s['message_id']]['not_useful']]
    for s in selection:
        if len(kept) >= floor:
            break
        if s['message_id'] not in kept:
            kept.append(s['message_id'])
    return kept


def keep_top(count):
    def rule(selection, judged):
        margin = lambda s: judged[s['message_id']]['useful'] - judged[s['message_id']]['not_useful']
        ranked = sorted(range(len(selection)), key=lambda j: (-margin(selection[j]), j))
        return [selection[j]['message_id'] for j in ranked[:count]]
    return rule


def keep_within(distance):
    """Messages of the ranker's selection scoring within `distance` of its top score. Needs no Qwen judgment."""
    def rule(selection, judged=None):
        top = max(s['score'] for s in selection)
        return [s['message_id'] for s in selection if s['score'] >= top - distance]
    return rule


FILTERS = {'useful': keep_useful, 'top4': keep_top(4), 'top6': keep_top(6), 'top8': keep_top(8), 'cut3': keep_within(3.0)}


def arm_parts(arm):
    memory, _, template = arm.rpartition('_')
    if template not in ('current', 'clean') or not (memory in ('oracle', 'ranker') or memory.removeprefix('filter_') in FILTERS):
        raise ValueError(f'Unknown arm {arm}: use oracle|ranker|filter_<{"|".join(FILTERS)}> + _current|_clean')
    return memory, template


def setup():
    questions, contexts, targets, sources = load_dataset('v2', ['val'], root=ROOT)
    questions = order_for_cache(questions)
    config = evaluation_config('v2', 'selector', root=ROOT)
    settings = runtime_metadata('v2', 'selector', memory_tokens=MEMORY_TOKENS)
    config.update(memory_tokens=MEMORY_TOKENS, input_policy='raw_complete_messages_v1', server_settings=settings,
                  model=settings['Model'], context_tokens=settings['ContextTokens'])
    data = CachedInputs(FEATURES)
    scores = np.load(SCORES, mmap_mode='r')
    ranker = FrozenHistories(questions, contexts, data, scores)
    oracle = ReferenceHistories(questions, contexts, sources, root=ROOT)
    return questions, contexts, targets, config, ranker, oracle


def score_arms(arms, limit=0):
    questions, contexts, targets, config, ranker, oracle = setup()
    labels = config['answer_labels']
    if limit:
        questions = questions[:limit]
    judgments = load_judgments() if any(a.startswith('filter_') for a in arms) else {}
    # Arms sharing a memory block run back to back, so the prompt cache carries it across templates.
    arms = sorted(arms, key=lambda a: (arm_parts(a)[0], arm_parts(a)[1]))
    templates = {'current': config, 'clean': {**config, 'user_template': CLEAN_TEMPLATE}}
    streams, done, rows = {}, {}, {}
    for arm in arms:
        (OUTPUT / arm).mkdir(parents=True, exist_ok=True)
        rows[arm] = read_rows(OUTPUT / arm / 'predictions.jsonl')
        done[arm] = {r['example_id'] for r in rows[arm]}
    todo = [q for q in questions if any(q['example_id'] not in done[a] for a in arms)]
    print(f'V2 validation: {len(questions)} questions, {len(todo)} to score; arms {", ".join(arms)}', flush=True)
    started = time.perf_counter()
    if todo:
        with qwen_server.LlamaCppServer(OUTPUT, config) as server:
            for arm in arms:
                streams[arm] = (OUTPUT / arm / 'predictions.jsonl').open('a', encoding='utf-8', buffering=1)
            try:
                for count, q in enumerate(todo, 1):
                    selection_history, selection = None, None
                    for arm in arms:
                        if q['example_id'] in done[arm]:
                            continue
                        memory, template = arm_parts(arm)
                        extra = {}
                        if memory == 'oracle':
                            history, source = oracle.get(q)
                            extra = dict(reference_message_count=source['reference_message_count'])
                        else:
                            if selection is None:
                                selection_history, selection = ranker.get(q, server, config)
                            history = selection_history
                            extra = dict(selected_messages=selection['selected_messages'], packed_memory_tokens=selection['packed_memory_tokens'])
                            if memory.startswith('filter_'):
                                kept = set(FILTERS[memory.removeprefix('filter_')](selection['selected_messages'], judgments[q['example_id']]))
                                members = ranker.pools[q['history_id'], q['source_end_index']]
                                history = selected_history(members, [j for j, m in enumerate(members) if m['message_id'] in kept])
                                extra['selected_messages'] = [s for s in selection['selected_messages'] if s['message_id'] in kept]
                        shifts = []
                        for shift in range(len(labels)):
                            p = server.score(build_messages(rotated(q, shift), templates[template], history),
                                             request_id=f'{q["example_id"]}|{arm}|shift{shift}', history_messages=len(history))
                            conditional = p['label_conditional_logprobs']
                            shifts.append(dict(shift=shift, predicted_label=p['predicted_label'], label_conditional_logprobs=conditional,
                                               answer_logprobs=answer_scores(conditional, labels, shift), prompt_tokens=p['prompt_tokens'],
                                               inference_seconds=p['inference_seconds'], timings=p.get('timings')))
                        gold = labels.index(targets[q['example_id']])
                        row = dict(example_id=q['example_id'], persona_id=q['persona_id'], question_type=q['question_type'], gold_index=gold,
                                   predicted_label=shifts[0]['predicted_label'], correct=shifts[0]['predicted_label'] == labels[gold],
                                   history_messages=len(history), selected_memory_count=len(history), shifts=shifts, **extra)
                        streams[arm].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                        rows[arm].append(row)
                    if count % 25 == 0:
                        print(f'{count}/{len(todo)} questions; {(time.perf_counter() - started) / 60:.1f} min', flush=True)
            finally:
                for stream in streams.values():
                    stream.close()
    summarize_arms(arms)


def judge(parallel=8):
    """Qwen's useful / not-useful judgment for every message of the ranker's validation selection."""
    selection_rows = load_selections()
    questions, contexts, _, _, ranker, _ = setup()
    if not {q['example_id'] for q in questions} <= set(selection_rows):
        raise SystemExit('Run --stage select first: its saved selections are what the judge grades')
    done = load_judgments()
    todo = [q for q in questions if q['example_id'] not in done]
    print(f'Judging {len(todo)} of {len(questions)} questions', flush=True)
    started = time.perf_counter()
    config = judge_config(ROOT, parallel=parallel)
    out = OUTPUT / 'judge'
    out.mkdir(parents=True, exist_ok=True)
    if todo:
        with qwen_server.LlamaCppServer(out, config) as server, (OUTPUT / 'judgments.jsonl').open('a', encoding='utf-8', buffering=1) as stream, \
                ThreadPoolExecutor(parallel * 2) as executor:
            for count, q in enumerate(todo, 1):
                members = {m['message_id']: m for m in ranker.pools[q['history_id'], q['source_end_index']]}
                selected = [s['message_id'] for s in selection_rows[q['example_id']]['selected_messages']]
                results = list(executor.map(lambda mid: judge_candidate(server, q, members[mid]), selected))
                messages = {mid: {'useful': r['judge']['label_raw_logprobs']['a'], 'not_useful': r['judge']['label_raw_logprobs']['b']}
                            for mid, r in zip(selected, results)}
                stream.write(json.dumps({'example_id': q['example_id'], 'messages': messages}) + '\n')
                if count % 50 == 0:
                    print(f'{count}/{len(todo)} judged; {(time.perf_counter() - started) / 60:.1f} min', flush=True)
    print(f'Judgments saved for {len(load_judgments())} questions', flush=True)


def policies():
    """Reference coverage of each filter rule on validation (uses reference labels for measurement only)."""
    from label_reference_messages import load_references
    from message_dataset import load_message_dataset
    selection_rows = load_selections()
    judgments = load_judgments()
    questions, memories, sources = load_message_dataset('v2', ['val'], root=ROOT)
    references = load_references(questions, sources, ROOT)
    report = {}
    rules = {'ranker selection (no filter)': lambda selection, judged: [s['message_id'] for s in selection], **FILTERS}
    for name, rule in rules.items():
        every, some, count, share = [], [], [], []
        for q in questions:
            if q['example_id'] not in judgments:
                continue
            keys = {k for k in references[q['example_id']] if k[1].strip()}
            pool = {(memories[m]['role'], memories[m]['text']): m for m in q['memory_ids']}
            visible = {pool[k] for k in keys if k in pool}
            if not visible:
                continue
            kept = set(rule(selection_rows[q['example_id']]['selected_messages'], judgments[q['example_id']]))
            every.append(visible <= kept); some.append(bool(visible & kept)); count.append(len(kept))
            share.append(len(visible & kept) / max(1, len(kept)))
        report[name] = dict(questions=len(every), every_reference=float(np.mean(every)), any_reference=float(np.mean(some)),
                            messages=float(np.mean(count)), share_that_are_references=float(np.mean(share)))
        print(f'{name:<30} every reference {np.mean(every):6.1%} | any {np.mean(some):6.1%} | messages {np.mean(count):5.1f} '
              f'| share that are references {np.mean(share):5.1%}', flush=True)
    (OUTPUT / 'policies.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def summarize_arms(arms=None):
    labels = evaluation_config('v2', 'selector', root=ROOT)['answer_labels']
    arms = arms or sorted(p.name for p in OUTPUT.iterdir() if (p / 'predictions.jsonl').exists())
    report = {}
    for arm in arms:
        rows = read_rows(OUTPUT / arm / 'predictions.jsonl')
        if rows:
            s = summarize(rows, labels)
            report[arm] = dict(questions=s['questions'], one_pass=s['accuracy']['one_pass'], four_orders=s['accuracy']['rotation_sum_logprob'],
                               letter_share_all_orders=s['letter_share_all_orders'], mean_prompt_tokens=s['mean_prompt_tokens'],
                               mean_messages=float(np.mean([r['history_messages'] for r in rows])))
            (OUTPUT / arm / 'report.json').write_text(json.dumps({'arm': arm, **s}, indent=2) + '\n', encoding='utf-8')
            print(f"{arm:<26} n={s['questions']:<4} one pass {s['accuracy']['one_pass']:.2%} | four orders {s['accuracy']['rotation_sum_logprob']:.2%} "
                  f"| prompt tokens {s['mean_prompt_tokens']:.0f} | messages {report[arm]['mean_messages']:.1f}", flush=True)
    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['score', 'select', 'judge', 'policies'], default='score')
    parser.add_argument('--arms', nargs='+', default=['oracle_current', 'oracle_clean', 'ranker_current', 'ranker_clean'])
    parser.add_argument('--limit', type=int, default=0, help='first N questions only (smoke test)')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        arm_parts(arm)
    if args.summarize_only:
        summarize_arms()
    elif args.stage == 'select':
        select()
    elif args.stage == 'judge':
        judge()
    elif args.stage == 'policies':
        policies()
    else:
        score_arms(args.arms, args.limit)


if __name__ == '__main__':
    main()
