"""Pilot: does rotating the answer options change Qwen's answers? (V2 or V1; selector, full history or oracle)

python scripts/option_rotation_pilot.py --questions 200
python scripts/option_rotation_pilot.py --questions 200 --label qwen_graded --model models/qwen_graded_all/ranker55.txt
python scripts/option_rotation_pilot.py --mode full_history --split test --questions 0
python scripts/option_rotation_pilot.py --mode full_history --dataset v1 --questions 0
python scripts/option_rotation_pilot.py --mode reference_only --split test --questions 0
python scripts/option_rotation_pilot.py --mode reference_only --split test --questions 0 --prompt-format reference
python scripts/option_rotation_pilot.py --split test --questions 0 --prompt-format reference --features data/ranker55/v2/test/features.npz

Each sampled question is scored once per cyclic shift of its options (four shifts for four
options) with the same memory: the selected messages, in full_history mode every visible
user and assistant message (source system/persona messages excluded, as in the evaluator),
or in reference_only mode (V2 only) the annotated reference snippets (the oracle).
The selector and the oracle share one prompt format ("Selected conversation messages",
"Message <source index>"). --prompt-format reference swaps in the oracle's earlier wording
("Dataset reference messages", "Reference message 0, 1, ...") without changing the messages
or their order (V2 snippet order is chronological source order); it is kept for the label test.
Shift 0 is exactly the evaluator's normal single pass. The rotated answer adds, for every
answer text, the log-probability it received under each letter and picks the highest total:
every answer sits once under every letter, so a preference for a letter or position cancels.
V2 validation questions by default; nothing is tuned here. The memory part of the prompt is
identical across shifts, so the server's prompt cache only re-reads the question and options.

Outputs under results/option_rotation/{dataset}/{split}/{label}/: predictions.jsonl (one row
per question, resumable; its predicted_label/correct fields are the shift-0 pass, so the folder
also works as a one-pass run for compare_selector_runs.py) and report.json.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

import qwen_llamacpp_server as qwen_server
from evaluate_qwen import (ReferenceHistories, SelectorHistories, VisibleHistories, build_messages, evaluation_config,
                           order_for_cache, runtime_metadata)
from raw_datasets import ROOT, load_dataset

def oracle_labels(history):
    """Drop source positions so messages are numbered 0, 1, ... as in the earlier oracle prompt."""
    return [{'role': m['role'], 'content': m['content']} for m in history]


def rotated(question, shift):
    """Letters stay a, b, c, ...; the answer texts move so letter i shows original answer (i + shift)."""
    options = question['options']
    n = len(options)
    return {**question, 'options': [{'label': options[i]['label'], 'text': options[(i + shift) % n]['text']} for i in range(n)]}


def bootstrap(diff, groups, draws=5000, seed=271):
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    values = [diff[np.concatenate([members[j] for j in rng.integers(0, len(unique), len(unique))])].mean() for _ in range(draws)]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def summarize(rows, labels):
    n = len(labels)
    gold = np.array([r['gold_index'] for r in rows])
    groups = np.array([r['persona_id'] for r in rows])
    # logp[q, shift, answer]: log-probability of each original answer text at each shift.
    logp = np.array([[[r['shifts'][s]['answer_logprobs'][a] for a in range(n)] for s in range(n)] for r in rows])
    letters = np.array([[labels.index(r['shifts'][s]['predicted_label']) for s in range(n)] for r in rows])
    single = logp.argmax(2)                                   # answer picked at each shift
    one_pass = single[:, 0]
    rotation = logp.sum(1).argmax(1)                          # product of probabilities across shifts
    mean_prob = np.exp(logp).mean(1).argmax(1)
    # Free correction on the single pass: remove each letter's average log-probability.
    first = np.array([[r['shifts'][0]['label_conditional_logprobs'][l] for l in labels] for r in rows])
    balanced = (first - first.mean(0)).argmax(1)              # shift 0: letter index equals answer index
    acc = lambda picks: float(np.mean(picks == gold))
    report = {
        'questions': len(rows), 'personas': int(len(np.unique(groups))),
        'accuracy': {'one_pass': acc(one_pass), 'rotation_sum_logprob': acc(rotation), 'rotation_mean_prob': acc(mean_prob),
                     'one_pass_letter_balanced': acc(balanced)},
        'accuracy_of_each_single_order': [acc(single[:, s]) for s in range(n)],
        'letter_share_per_order': [{labels[i]: float(np.mean(letters[:, s] == i)) for i in range(n)} for s in range(n)],
        'letter_share_all_orders': {labels[i]: float(np.mean(letters == i)) for i in range(n)},
        'correct_letter_share_one_pass': {labels[i]: float(np.mean(gold == i)) for i in range(n)},
        'same_answer_in_all_orders': float(np.mean((single == single[:, :1]).all(1))),
        'one_pass_changed_by_rotation': float(np.mean(one_pass != rotation)),
        'rotation_agrees_with_letter_balanced': float(np.mean(rotation == balanced)),
        'rotation_minus_one_pass': {'difference': acc(rotation) - acc(one_pass),
                                    'ci95_persona_bootstrap': bootstrap((rotation == gold).astype(float) - (one_pass == gold), groups),
                                    'gained': int(((rotation == gold) & (one_pass != gold)).sum()),
                                    'lost': int(((rotation != gold) & (one_pass == gold)).sum())},
        'accuracy_by_type': {t: {'questions': int(m.sum()), 'one_pass': float(np.mean(one_pass[m] == gold[m])),
                                 'rotation': float(np.mean(rotation[m] == gold[m]))}
                             for t in sorted({r['question_type'] for r in rows})
                             for m in [np.array([r['question_type'] == t for r in rows])]},
        'seconds_per_order': [float(np.mean([r['shifts'][s]['inference_seconds'] for r in rows])) for s in range(n)],
        'cached_prompt_tokens_per_order': [float(np.mean([(r['shifts'][s].get('timings') or {}).get('cache_n', 0) for r in rows])) for s in range(n)],
        'evaluated_prompt_tokens_per_order': [float(np.mean([(r['shifts'][s].get('timings') or {}).get('prompt_n', 0) for r in rows])) for s in range(n)],
        'mean_prompt_tokens': float(np.mean([r['shifts'][0]['prompt_tokens'] for r in rows])),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['v2', 'v1'], default='v2')
    parser.add_argument('--split', choices=['val', 'test', 'comparison'], default=None,
                        help='default: val for v2, comparison (the only split) for v1')
    parser.add_argument('--mode', choices=['selector', 'full_history', 'reference_only'], default='selector')
    parser.add_argument('--prompt-format', choices=['native', 'reference'], default='native',
                        help="memory heading/labels: the evaluator's, or the oracle's earlier wording (selector or reference_only mode)")
    parser.add_argument('--questions', type=int, default=200, help='random questions (seed-fixed); 0 = the whole split')
    parser.add_argument('--seed', type=int, default=271)
    parser.add_argument('--label', default=None, help='default: references_only for selector, otherwise the mode name; '
                                                      'plus _reference_labels with --prompt-format reference')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/subtlex_mix/alpha_0.5/ranker55.txt')
    parser.add_argument('--features', type=Path, default=None, help='default: data/ranker55/subtlex/{dataset}/{split}/features.npz')
    parser.add_argument('--memory-tokens', type=int, default=2048)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/option_rotation')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args.split = args.split or ('val' if args.dataset == 'v2' else 'comparison')
    if (args.dataset == 'v2') != (args.split in ('val', 'test')):
        parser.error('v2 uses --split val or test; v1 has only the comparison split')
    if args.mode == 'reference_only' and args.dataset != 'v2':
        parser.error('reference_only (oracle snippets) exists for V2 only')
    if args.prompt_format == 'reference' and args.mode == 'full_history':
        parser.error('--prompt-format reference applies to selector and reference_only modes')
    if not args.label:
        args.label = 'references_only' if args.mode == 'selector' else args.mode
        if args.prompt_format != 'native':
            args.label += f'_{args.prompt_format}_labels'
    selector = args.mode == 'selector'
    features = args.features or ROOT / f'data/ranker55/subtlex/{args.dataset}/{args.split}/features.npz'
    model = args.model if args.model.is_absolute() else ROOT / args.model
    out = args.output_root / args.dataset / args.split / args.label
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'predictions.jsonl'

    config = evaluation_config(args.dataset, args.mode, root=ROOT)
    labels = config['answer_labels']
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
    if not args.summarize_only:
        all_questions, contexts, targets, sources = load_dataset(args.dataset, [args.split], root=ROOT)
        if args.questions:
            pick = np.sort(np.random.default_rng(args.seed).choice(len(all_questions), args.questions, replace=False))
            questions = [all_questions[i] for i in pick]
        else:
            questions = all_questions
        # Questions sharing a history run back to back, so the prompt cache also carries across questions.
        questions = order_for_cache(questions)
        if any([o['label'] for o in q['options']] != labels for q in questions):
            raise ValueError('Question option labels differ from the configured answer labels')
        memory_tokens = args.memory_tokens if selector else None
        settings = runtime_metadata(args.dataset, args.mode, memory_tokens=memory_tokens)
        config.update(memory_tokens=memory_tokens, input_policy='raw_complete_messages_v1', server_settings=settings,
                      model=settings['Model'], context_tokens=settings['ContextTokens'])
        if selector:
            histories = SelectorHistories(all_questions, contexts, sources, features, model, None,
                                          memory_tokens=args.memory_tokens, selected_ids=[q['example_id'] for q in questions])
        elif args.mode == 'reference_only':
            histories = ReferenceHistories(questions, contexts, sources, root=ROOT)
        else:
            histories = VisibleHistories(questions, contexts, exclude_system=True)
        # The server config keeps the run's own mode (context size, selector budget check); only
        # the prompt text uses the swapped heading and labels.
        prompt_config = {**config, 'memory_labels': 'reference'} if args.prompt_format == 'reference' else config
        done = {r['example_id'] for r in rows}
        todo = [q for q in questions if q['example_id'] not in done]
        print(f'{args.dataset.upper()} {args.split}: {len(questions)} questions, {len(done)} already scored; '
              f'{args.mode} {args.label} (prompt format {args.prompt_format}); output {out}', flush=True)
        spent = 0.0
        if todo:
            with qwen_server.LlamaCppServer(out, config) as server, path.open('a', encoding='utf-8', buffering=1) as stream:
                if args.prompt_format == 'reference' and selector:
                    # Packing still uses `config` (selector labels, 2,048-token budget), so the messages
                    # match the native run; the oracle labels can lengthen that memory by a token. Drop
                    # only the scoring-time budget check; the context-size check stays.
                    server.config = {**config, 'memory_tokens': None}
                for q in todo:
                    history, source = histories.get(q, server=server, config=config) if selector else histories.get(q)
                    shown = oracle_labels(history) if args.prompt_format == 'reference' else history
                    gold = labels.index(targets[q['example_id']])
                    shifts = []
                    for shift in range(len(labels)):
                        prediction = server.score(build_messages(rotated(q, shift), prompt_config, shown),
                                                  request_id=f"{q['example_id']}|shift{shift}", history_messages=len(history))
                        conditional = prediction['label_conditional_logprobs']
                        # Letter i shows original answer (i + shift): read the scores back per answer text.
                        answers = [None] * len(labels)
                        for i, label in enumerate(labels):
                            answers[(i + shift) % len(labels)] = conditional[label]
                        if any(v is None or not math.isfinite(v) for v in answers):
                            raise ValueError(f"{q['example_id']}: missing answer score at shift {shift}")
                        shifts.append({'shift': shift, 'predicted_label': prediction['predicted_label'],
                                       'label_conditional_logprobs': conditional, 'answer_logprobs': answers,
                                       'prompt_tokens': prediction['prompt_tokens'],
                                       'inference_seconds': prediction['inference_seconds'], 'timings': prediction.get('timings')})
                    row = {'example_id': q['example_id'], 'persona_id': q['persona_id'], 'question_type': q['question_type'],
                           'gold_index': gold, 'predicted_label': shifts[0]['predicted_label'],
                           'correct': shifts[0]['predicted_label'] == labels[gold],
                           'history_messages': len(history), 'shifts': shifts}
                    if selector:
                        row.update(selected_memory_count=source['selected_memory_count'],
                                   packed_memory_tokens=source.get('packed_memory_tokens'))
                    if args.mode == 'reference_only':
                        row.update(reference_message_count=source['reference_message_count'],
                                   reference_messages_missing_from_visible_history=source['reference_messages_missing_from_visible_history'])
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    rows.append(row)
                    spent += sum(s['inference_seconds'] for s in shifts)
                    print(f'{len(rows)}/{len(questions)} questions; {spent / (len(rows) - len(done)):.2f}s Qwen time per question', flush=True)
    if not rows:
        raise SystemExit('No predictions to summarize')
    report = {'dataset': args.dataset, 'split': args.split, 'mode': args.mode, 'prompt_format': args.prompt_format, 'ranker': args.label,
              'model': str(model) if selector else None, 'memory_tokens': args.memory_tokens if selector else None,
              'seed': args.seed, **summarize(rows, labels)}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    a = report['accuracy']
    print(f"\n{args.label} on {args.dataset.upper()} {args.split}, {report['questions']} questions")
    print(f"  one pass {a['one_pass']:.2%} | four orders {a['rotation_sum_logprob']:.2%} | one pass, letter-balanced {a['one_pass_letter_balanced']:.2%}")
    d = report['rotation_minus_one_pass']
    print(f"  four orders minus one pass: {100 * d['difference']:+.2f} points ({100 * d['ci95_persona_bootstrap'][0]:+.2f} to {100 * d['ci95_persona_bootstrap'][1]:+.2f}); +{d['gained']}/-{d['lost']}")
    print('  accuracy of each single order:', ' '.join(f'{v:.1%}' for v in report['accuracy_of_each_single_order']))
    print('  letters picked (all orders):', ' '.join(f'{k}:{v:.0%}' for k, v in report['letter_share_all_orders'].items()))
    print(f"  same answer in all four orders: {report['same_answer_in_all_orders']:.1%}")
    print('  Qwen seconds per order:', ' '.join(f'{v:.2f}' for v in report['seconds_per_order']),
          '| prompt tokens read per order:', ' '.join(f'{v:.0f}' for v in report['evaluated_prompt_tokens_per_order']))


if __name__ == '__main__':
    main()
