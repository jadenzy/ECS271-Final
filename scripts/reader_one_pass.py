"""Compare three frozen memory policies on all V2 validation questions, one answer order.

python scripts/reader_one_pass.py
python scripts/reader_one_pass.py --check-only

Reuses reader_validation's saved selections and message judgments. No training, judging,
option rotation, MMR, prompt changes, or test-set evaluation is performed.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

import reader_validation as reader
from evaluate_embedding_ab_transfer import accept_protocol, answer_scores
from evaluate_qwen import build_messages, selected_history
from option_rotation_pilot import bootstrap
from raw_datasets import ROOT, sha256

OUTPUT = ROOT / 'results/reader_validation/one_pass'
ARMS = ('ranker_current', 'ranker_top8_current', 'filter_top8_current')


def choose_messages(selection, judged, arm):
    if arm == 'ranker_current':
        return [s['message_id'] for s in selection]
    if arm == 'ranker_top8_current':
        return [s['message_id'] for s in selection[:8]]
    if arm == 'filter_top8_current':
        return reader.keep_top(8)(selection, judged)
    raise ValueError(f'Unknown arm: {arm}')


def checked_inputs(questions, ranker, selection_rows, judgment_rows):
    ids = {q['example_id'] for q in questions}
    selections = {r['example_id']: r for r in selection_rows}
    judgments = {r['example_id']: r['messages'] for r in judgment_rows}
    if (len(selection_rows) != len(selections) or len(judgment_rows) != len(judgments)
            or set(selections) != ids or set(judgments) != ids):
        raise ValueError('Missing, duplicate or foreign saved questions')
    for q in questions:
        qid = q['example_id']
        selected = selections[qid]['selected_messages']
        mids = [s['message_id'] for s in selected]
        members = ranker.pools[q['history_id'], q['source_end_index']]
        i = ranker.groups[qid]
        a, b = map(int, ranker.data.offsets[i:i+2])
        scores = dict(zip((m['message_id'] for m in members), ranker.scores[a:b]))
        if (len(mids) != len(set(mids)) or not set(mids) <= set(scores)
                or set(mids) != set(judgments[qid])):
            raise ValueError('Selection or judgment IDs differ from the visible pool')
        values = [s['score'] for s in selected]
        if values != sorted(values, reverse=True) or any(s['score'] != scores[s['message_id']] for s in selected):
            raise ValueError('Saved selection differs from frozen ranker scores/order')
        if not 0 <= selections[qid]['packed_memory_tokens'] <= reader.MEMORY_TOKENS:
            raise ValueError('Saved selection exceeds the memory budget')
        if any(set(v) != {'useful', 'not_useful'} or not np.isfinite(list(v.values())).all()
               for v in judgments[qid].values()):
            raise ValueError('Missing/nonfinite judge scores')
    return selections, judgments


def prepare():
    questions, _, targets, config, ranker, _ = reader.setup()
    if len(questions) != 909:
        raise ValueError('Expected the complete 909-question validation cohort')
    selections, judgments = checked_inputs(questions, ranker,
        reader.read_rows(reader.OUTPUT / 'selections.jsonl'), reader.read_rows(reader.OUTPUT / 'judgments.jsonl'))
    record = json.loads((ROOT / 'results/embedding_ab/v2/val/finetuned_ranker57.json').read_text())
    if (sha256(reader.FEATURES) != record['validation_features_sha256']
            or sha256(reader.SCORES) != record['score_file_sha256']):
        raise ValueError('Frozen validation features or scores changed')
    files = [Path(__file__), ROOT / 'scripts/reader_validation.py', ROOT / 'scripts/evaluate_qwen.py',
             ROOT / 'scripts/qwen_llamacpp_server.py', ROOT / 'scripts/evaluate_embedding_ab_transfer.py',
             reader.OUTPUT / 'selections.jsonl', reader.OUTPUT / 'judgments.jsonl', reader.SCORES]
    content = json.dumps(dict(questions=questions, targets=targets, pools=list(ranker.pools.values())),
                         sort_keys=True, ensure_ascii=False).encode('utf-8')
    protocol = dict(dataset='v2', split='val', questions=len(questions), rotations=1,
                    question_ids=[q['example_id'] for q in questions], arms=list(ARMS), config=config,
                    memory_tokens=reader.MEMORY_TOKENS, prompt='current', mmr=False, training_or_tuning=False,
                    ranker_sha256=record['model_sha256'], encoder=record['embedding_recipe']['model'],
                    input_content_sha256=hashlib.sha256(content).hexdigest(),
                    files_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in files})
    return questions, targets, config, ranker, selections, judgments, protocol


def score_one(server, question, config, members, selection, judged, arm, target):
    kept = set(choose_messages(selection, judged, arm))
    history = selected_history(members, [i for i, m in enumerate(members) if m['message_id'] in kept])
    if len(history) != len(kept):
        raise ValueError('Selected messages missing from visible history')
    messages = build_messages(question, config, history)
    bare = [messages[0], messages[-1]]
    server.verify_selector_prompt(messages, server.render_prompt(messages), bare, server.render_prompt(bare))
    prediction = server.score(messages, request_id=f'{question["example_id"]}|{arm}|shift0', history_messages=len(history))
    labels = config['answer_labels']
    conditional = prediction['label_conditional_logprobs']
    shift = dict(shift=0, predicted_label=prediction['predicted_label'], label_conditional_logprobs=conditional,
                 answer_logprobs=answer_scores(conditional, labels, 0), prompt_tokens=prediction['prompt_tokens'],
                 inference_seconds=prediction['inference_seconds'], timings=prediction.get('timings'))
    return dict(example_id=question['example_id'], persona_id=question['persona_id'], question_type=question['question_type'],
                gold_index=labels.index(target), predicted_label=prediction['predicted_label'],
                correct=prediction['predicted_label'] == target, shifts=[shift],
                selected_messages=[s for s in selection if s['message_id'] in kept],
                history_messages=len(history), selected_memory_count=len(history),
                packed_memory_tokens=prediction['memory_tokens'], prompt_sha256=prediction['prompt_sha256'])


def summarize(rows, questions, targets, config, selections, judgments, output):
    labels = config['answer_labels']
    ids = [q['example_id'] for q in questions]
    groups = np.array([q['persona_id'] for q in questions])
    report = dict(dataset='v2', split='val', questions=len(ids), personas=len(set(groups)), rotations=1,
                  prompt='current', status='complete', arms={}, paired={})
    correct = {}
    for arm in ARMS:
        by_id = {r['example_id']: r for r in rows[arm]}
        if len(rows[arm]) != len(ids) or set(by_id) != set(ids):
            raise ValueError('Incomplete or duplicate result cohort')
        ordered = [by_id[qid] for qid in ids]
        for q, r in zip(questions, ordered):
            expected = set(choose_messages(selections[r['example_id']]['selected_messages'], judgments[r['example_id']], arm))
            if (len(r['shifts']) != 1 or r['shifts'][0]['shift'] != 0
                    or r['gold_index'] != labels.index(targets[r['example_id']])
                    or r['persona_id'] != q['persona_id'] or r['question_type'] != q['question_type']
                    or r['correct'] != (r['predicted_label'] == targets[r['example_id']])
                    or r['predicted_label'] != labels[int(np.argmax(r['shifts'][0]['answer_logprobs']))]
                    or not 0 <= r['packed_memory_tokens'] <= reader.MEMORY_TOKENS
                    or {s['message_id'] for s in r['selected_messages']} != expected
                    or len(r['selected_messages']) != len(expected) or r['history_messages'] != len(expected)):
                raise ValueError('Prediction integrity check failed')
        correct[arm] = np.array([r['correct'] for r in ordered], dtype=float)
        report['arms'][arm] = dict(correct=int(correct[arm].sum()), accuracy=float(correct[arm].mean()),
            mean_messages=float(np.mean([r['history_messages'] for r in ordered])),
            mean_memory_tokens=float(np.mean([r['packed_memory_tokens'] for r in ordered])),
            mean_prompt_tokens=float(np.mean([r['shifts'][0]['prompt_tokens'] for r in ordered])))
    for newer, older in [(ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])]:
        delta = correct[newer] - correct[older]
        report['paired'][f'{newer} - {older}'] = dict(difference=float(delta.mean()),
            ci95_persona_bootstrap=bootstrap(delta, groups), gained=int((delta > 0).sum()), lost=int((delta < 0).sum()))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    questions, targets, config, ranker, selections, judgments, protocol = prepare()
    print(f'PRECHECK PASSED: {len(questions)} validation questions; all saved selections and judgments verified.', flush=True)
    if args.check_only:
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, done = {}, {}
    for arm in ARMS:
        (OUTPUT / arm).mkdir(exist_ok=True)
        rows[arm] = reader.read_rows(OUTPUT / arm / 'predictions.jsonl')
        done[arm] = accept_protocol(OUTPUT / 'protocol.json', protocol, rows[arm], protocol['question_ids'])
    todo = [q for q in questions if any(q['example_id'] not in done[a] for a in ARMS)]
    if not args.summarize_only and todo:
        started = time.perf_counter()
        print(f'SCORING: {len(todo)} questions remaining, three policies, ONE answer order each.', flush=True)
        with reader.qwen_server.LlamaCppServer(OUTPUT, config) as server, ExitStack() as stack:
            streams = {arm: stack.enter_context((OUTPUT / arm / 'predictions.jsonl').open('a', encoding='utf-8', buffering=1)) for arm in ARMS}
            for count, q in enumerate(todo, 1):
                qid = q['example_id']
                members = ranker.pools[q['history_id'], q['source_end_index']]
                for arm in ARMS:
                    if qid in done[arm]:
                        continue
                    row = score_one(server, q, config, members, selections[qid]['selected_messages'], judgments[qid], arm, targets[qid])
                    if arm == ARMS[0] and row['packed_memory_tokens'] != selections[qid]['packed_memory_tokens']:
                        raise ValueError('Unfiltered prompt differs from saved packing')
                    streams[arm].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    rows[arm].append(row)
                if count % 25 == 0 or count == len(todo):
                    elapsed = (time.perf_counter() - started) / 60
                    print(f'PROGRESS {count}/{len(todo)} questions; {elapsed:.1f} min elapsed; ETA {elapsed * (len(todo) - count) / count:.1f} min', flush=True)
        (OUTPUT / 'runtime.json').write_text(json.dumps(dict(new_questions=len(todo), elapsed_seconds=time.perf_counter()-started), indent=2) + '\n')
    summarize(rows, questions, targets, config, selections, judgments, OUTPUT)
    print('VERIFIED COMPLETE: 909 validation questions, 2,727 single-pass answers.', flush=True)


if __name__ == '__main__':
    main()
