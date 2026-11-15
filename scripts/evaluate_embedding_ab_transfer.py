"""Frozen fine-tuned 57-feature ranker: V2/V1 four rotations, LoCoMo one pass.

python scripts/evaluate_embedding_ab_transfer.py --stage features
python scripts/evaluate_embedding_ab_transfer.py --stage score
python scripts/evaluate_embedding_ab_transfer.py --stage summarize
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

import build_ranker_features as builder
from embedding_ab_experiment import ENCODER, MODELS, OUTPUT as TRAINING_RECORDS, verify_encoder
from embedding_ab_math import CachedInputs
from math_ranker_experiment import NAMES, extra_features, validate_cosine, vectors_on_gpu
from raw_datasets import ROOT, load_dataset, memories_for_question, sha256
from tune_ranker55 import Bundle, write_json
from evaluate_qwen import (SelectorMessageBlocks, build_messages, evaluation_config, order_for_cache,
                           pack_selector_messages, runtime_metadata, selected_history)
from option_rotation_pilot import bootstrap, rotated, summarize
import qwen_llamacpp_server as qwen_server

OUTPUT = ROOT / 'results/embedding_ab/transfer'
CACHE = ROOT / 'data/ranker55/embedding_ab/transfer'
MODEL_ROOT = MODELS / 'finetuned_math'
DATASETS = {'v2': ('test', 4), 'v1': ('comparison', 4), 'locomo': ('comparison', 1)}
MEMORY_TOKENS = 2048


def frozen_models():
    info = verify_encoder()
    record_path = TRAINING_RECORDS / 'finetuned_ranker57.json'
    record = json.loads(record_path.read_text())
    for name, key in [('ranker57.txt', 'model_sha256'), ('adapter.npz', 'adapter_sha256')]:
        if sha256(MODEL_ROOT / name) != record[key]:
            raise ValueError('Frozen ranker/adapter changed')
    if record['feature_names'] != NAMES or record['embedding_recipe']['model'] != info:
        raise ValueError('Saved encoder/ranker feature identity mismatch')
    frozen = dict(encoder=info, ranker_sha256=record['model_sha256'], adapter_sha256=record['adapter_sha256'],
                  training_record_sha256=sha256(record_path), feature_names=NAMES, mathematical_settings=record['settings'],
                  memory_tokens=MEMORY_TOKENS, redundancy_selection_used=False, training_or_tuning=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'frozen_model.json'
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError('This evaluation already has different frozen settings')
    write_json(path, frozen)
    return info, record, frozen


def score_features(path, record):
    import torch
    import lightgbm as lgb
    data = CachedInputs(path)
    if data.meta['embedding_recipe'] != record['embedding_recipe'] or data.meta['feature_recipe'] != record['feature_recipe']:
        raise ValueError('Transfer features differ from the trained encoder/lexical recipe')
    people = json.loads((TRAINING_RECORDS / 'partition.json').read_text())
    if set(data.personas) & set(people['a_personas'] + people['b_personas']):
        raise ValueError('Evaluation personas overlap training A/B')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    memory, query = vectors_on_gpu(data)
    alignment = validate_cosine(data, memory, query)
    with np.load(MODEL_ROOT / 'adapter.npz') as adapter:
        U, V, bank = [torch.tensor(adapter[k], device='cuda') for k in ['U', 'V', 'bank']]
    extra = np.full((len(data.X), 2), np.nan, dtype=np.float32)
    extra_features(data, memory, query, U, V, bank, np.arange(len(data.qids)), record['settings'], extra)
    del memory, query, U, V, bank
    gc.collect()
    torch.cuda.empty_cache()
    if not np.isfinite(extra).all():
        raise ValueError('Nonfinite transfer features')
    model = lgb.Booster(model_file=str(MODEL_ROOT / 'ranker57.txt'))
    if model.feature_name() != NAMES:
        raise ValueError('Wrong ranker feature order')
    scores = model.predict(np.column_stack([data.X, extra]), num_threads=4)
    np.save(path.parent / 'mathematical_features.npy', extra)
    np.save(path.parent / 'ranker_scores.npy', scores)
    score_record = dict(features_sha256=sha256(path), scores_sha256=sha256(path.parent / 'ranker_scores.npy'),
                        ranker_sha256=record['model_sha256'], adapter_sha256=record['adapter_sha256'],
                        question_ids=data.qids, row_count=len(scores), cosine_alignment_max_error=alignment,
                        inference_only=True, uses_evaluation_labels_or_answers=False)
    write_json(path.parent / 'scoring.json', score_record)
    data.close()
    return score_record


def features(datasets):
    info, record, frozen = frozen_models()
    counts, total, lock = builder.subtlex.load_prior(ROOT / 'data/external/subtlex_us/word_frequencies.json',
                                                   ROOT / 'configs/subtlex_us.lock.json')
    mixture = builder.subtlex.Mixture(counts, total, builder.subtlex.configuration(.5, lock))
    statistics = {'schema_version': 2, 'feature_recipe': mixture.config}
    encoder = builder.Encoder(ENCODER, 'cuda')
    for dataset in datasets:
        split, _ = DATASETS[dataset]
        output = CACHE / dataset / split
        path = output / 'features.npz'
        print(f'FEATURES: {dataset}/{split}', flush=True)
        if not path.exists() or not (output / 'report.json').exists():
            task = builder.prepare_inputs(root=ROOT, datasets=[dataset], splits=[split])[0]
            builder.build_bundle(task, statistics, encoder, info, output,
                                 batch_size=128, batch_tokens=16000, feature_workers=4, mixture=mixture)
        else:
            completed = json.loads((output / 'report.json').read_text())
            if completed['embedding_recipe']['model'] != info or completed['files_sha256']['features.npz'] != sha256(path):
                raise ValueError('Completed transfer bundle identity changed')
        if (output / 'scoring.json').exists():
            cached = json.loads((output / 'scoring.json').read_text())
            if (cached['features_sha256'] != sha256(path) or cached['ranker_sha256'] != frozen['ranker_sha256']
                    or cached['adapter_sha256'] != frozen['adapter_sha256']
                    or cached['scores_sha256'] != sha256(output / 'ranker_scores.npy')):
                raise ValueError('Cached ranker scores have changed')
        else:
            score_features(path, record)
        print(f'READY: {dataset}/{split}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['features', 'score', 'summarize', 'all'], default='all')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    if args.stage in ['features', 'all']:
        features(args.datasets)
    if args.stage in ['score', 'all']:
        for dataset in args.datasets:
            score(dataset)
    if args.stage in ['summarize', 'all']:
        summarize_runs(args.datasets)


def answer_scores(conditional, labels, shift):
    values = [conditional[labels[(a-shift) % len(labels)]] for a in range(len(labels))]
    if not np.isfinite(values).all():
        raise ValueError('Missing/nonfinite answer probabilities')
    return values


def accept_protocol(path, protocol, rows, question_ids):
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Resume protocol differs; refusing mixed predictions')
    elif rows:
        raise ValueError('Predictions exist without their frozen protocol')
    done = {r['example_id'] for r in rows}
    if len(done) != len(rows) or not done <= set(question_ids):
        raise ValueError('Duplicate or foreign resumed questions')
    if any(len(r['shifts']) != protocol['rotations'] for r in rows):
        raise ValueError('Resumed answer-order count differs')
    write_json(path, protocol)
    return done


class FrozenHistories:
    def __init__(self, questions, contexts, data, scores):
        self.data, self.scores = data, scores
        self.groups = {qid: i for i, qid in enumerate(data.qids)}
        if len(questions) != len(data.qids) or {q['example_id'] for q in questions} != set(data.qids):
            raise ValueError('Evaluation cohort differs from score cache')
        if scores.shape != (int(data.offsets[-1]),) or not np.isfinite(scores).all():
            raise ValueError('Invalid cached ranker scores')
        self.pools, self.blocks = {}, {}
        for q in questions:
            key = q['history_id'], q['source_end_index']
            i = self.groups[q['example_id']]
            a, b = map(int, data.offsets[i:i+2])
            members = memories_for_question(q, contexts)
            if [m['message_id'] for m in members] != data.bundle.read('message_ids')[a:b].tolist():
                raise ValueError('Score rows differ from the raw visible message pool')
            if key not in self.pools:
                self.pools[key] = members
                self.blocks[key] = SelectorMessageBlocks(q['dataset'], members)

    def get(self, question, server, config):
        key = question['history_id'], question['source_end_index']
        members = self.pools[key]
        i = self.groups[question['example_id']]
        a, b = map(int, self.data.offsets[i:i+2])
        scores = self.scores[a:b]
        ranked = np.argsort(-scores, kind='stable').tolist()
        chosen, packing = pack_selector_messages(server, question, config, members, ranked, MEMORY_TOKENS,
                                                 blocks=self.blocks[key])
        history = selected_history(members, chosen)
        return history, dict(**packing, selected_memory_count=len(history),
             selected_messages=[dict(message_id=members[j]['message_id'], message_index=members[j]['message_index'],
                                     score=float(scores[j])) for j in chosen])


def run_summary(dataset, rows, config):
    if len(rows) != len(set(r['example_id'] for r in rows)):
        raise ValueError('Duplicate predictions')
    if DATASETS[dataset][1] == 4:
        result = summarize(rows, config['answer_labels'])
    else:
        correct = np.array([r['correct'] for r in rows], dtype=float)
        people = np.array([r['persona_id'] for r in rows])
        result = dict(questions=len(rows), personas=int(len(np.unique(people))),
                      accuracy={'one_pass': float(correct.mean())},
                      ci95_persona_bootstrap=bootstrap(correct, people),
                      mean_prompt_tokens=float(np.mean([r['shifts'][0]['prompt_tokens'] for r in rows])))
    result.update(status='complete', dataset=dataset, split=DATASETS[dataset][0], rotations=DATASETS[dataset][1],
                  memory_tokens=MEMORY_TOKENS, redundancy_selection_used=False,
                  mean_selected_messages=float(np.mean([r['selected_memory_count'] for r in rows])),
                  mean_memory_tokens=float(np.mean([r['packed_memory_tokens'] for r in rows])))
    write_json(OUTPUT / dataset / 'report.json', result)
    print(f'COMPLETE {dataset}: {result["questions"]} questions; {result["accuracy"]}', flush=True)
    return result


def score(dataset):
    started = time.perf_counter()
    _, record, frozen = frozen_models()
    split, rotations = DATASETS[dataset]
    out = OUTPUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    questions, contexts, targets, sources = load_dataset(dataset, [split], root=ROOT)
    questions = order_for_cache(questions)
    path = CACHE / dataset / split / 'features.npz'
    scoring = json.loads((path.parent / 'scoring.json').read_text())
    if (scoring['ranker_sha256'] != frozen['ranker_sha256'] or scoring['adapter_sha256'] != frozen['adapter_sha256']
            or scoring['features_sha256'] != sha256(path)
            or scoring['scores_sha256'] != sha256(path.parent / 'ranker_scores.npy')):
        raise ValueError('Feature/score cache differs from frozen models')
    data = CachedInputs(path)
    if data.meta['embedding_recipe'] != record['embedding_recipe']:
        raise ValueError('Feature encoder differs from trained ranker')
    for source, expected in sources.items():
        if data.meta['source_sha256'].get(source) != expected:
            raise ValueError('Current raw source or split differs from feature cache')
    scores = np.load(path.parent / 'ranker_scores.npy', mmap_mode='r')
    histories = FrozenHistories(questions, contexts, data, scores)
    config = evaluation_config(dataset, 'selector')
    labels = config['answer_labels']
    if any([o['label'] for o in q['options']] != labels for q in questions):
        raise ValueError('Raw option labels differ from configured scorer')
    settings = runtime_metadata(dataset, 'selector', memory_tokens=MEMORY_TOKENS)
    config.update(memory_tokens=MEMORY_TOKENS, input_policy='raw_complete_messages_v1', server_settings=settings,
                  model=settings['Model'], context_tokens=settings['ContextTokens'])
    predictions = out / 'predictions.jsonl'
    rows = [json.loads(line) for line in predictions.open(encoding='utf-8')] if predictions.exists() else []
    protocol = dict(dataset=dataset, split=split, rotations=rotations, config=config, frozen_model=frozen,
                    scoring_sha256=sha256(path.parent / 'scoring.json'), raw_sources=sources,
                    question_ids=data.qids, questions_sha256=builder.fingerprint(questions),
                    targets_sha256=builder.fingerprint(targets), script_sha256=sha256(Path(__file__)),
                    scorer_sha256=sha256(Path(qwen_server.__file__)),
                    selection_policy='greedy rank order under native token budget; whole messages, chronological display',
                    uses_evaluation_labels_for_selection=False)
    done = accept_protocol(out / 'protocol.json', protocol, rows, data.qids)
    todo = [q for q in questions if q['example_id'] not in done]
    print(f'SCORING {dataset}: {len(questions)} questions, {len(done)} complete, {rotations} answer order(s)', flush=True)
    if todo:
        with qwen_server.LlamaCppServer(out, config) as server, predictions.open('a', encoding='utf-8', buffering=1) as stream:
            for q in todo:
                history, selection = histories.get(q, server, config)
                shifts = []
                for shift in range(rotations):
                    p = server.score(build_messages(rotated(q, shift), config, history),
                                     request_id=f'{q["example_id"]}|shift{shift}', history_messages=len(history))
                    conditional = p['label_conditional_logprobs']
                    shifts.append(dict(shift=shift, predicted_label=p['predicted_label'],
                                       label_conditional_logprobs=conditional, answer_logprobs=answer_scores(conditional, labels, shift),
                                       prompt_tokens=p['prompt_tokens'], inference_seconds=p['inference_seconds'], timings=p.get('timings')))
                # Targets are accessed after feature scoring, memory selection, and all answer scoring.
                gold = labels.index(targets[q['example_id']])
                row = dict(example_id=q['example_id'], persona_id=q['persona_id'], question_type=q['question_type'],
                           gold_index=gold, predicted_label=shifts[0]['predicted_label'],
                           correct=shifts[0]['predicted_label'] == labels[gold], shifts=shifts, **selection)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                rows.append(row)
                if len(rows) % 25 == 0:
                    print(f'{dataset}: {len(rows)}/{len(questions)} complete; {(time.perf_counter()-started)/60:.1f} min', flush=True)
    if {r['example_id'] for r in rows} != set(data.qids):
        raise ValueError('Incomplete evaluation')
    run_summary(dataset, rows, config)
    write_json(out / 'runtime.json', dict(seconds_this_process=time.perf_counter()-started, resumed_questions=len(done)))
    data.close()


def summarize_runs(datasets):
    from compare_selector_runs import load
    rng = np.random.default_rng(271)
    report, parts = dict(datasets={}, bootstrap_draws=5000, memory_tokens=MEMORY_TOKENS,
                         frozen_model=json.loads((OUTPUT / 'frozen_model.json').read_text())), []
    for dataset in datasets:
        split, rotations = DATASETS[dataset]
        folder = OUTPUT / dataset
        data_report = json.loads((folder / 'report.json').read_text())
        active = set(json.loads((CACHE / dataset / split / 'scoring.json').read_text())['question_ids'])
        if data_report['questions'] != len(active):
            raise ValueError('Dataset evaluation is incomplete')
        if dataset in ['v1', 'v2']:
            base = ROOT / f'results/option_rotation/{dataset}/{split}'
            paths = dict(reference=base / 'references_only', graded=base / 'qwen_graded', full=base / 'full_history')
        else:
            base = ROOT / 'results/qwen35_4bselector_raw/locomo_mc10/full'
            paths = dict(reference=base / 'comparison_memory_2048_references_only',
                         graded=base / 'comparison_memory_2048_qwen_graded_all',
                         full=ROOT / 'results/qwen35_4bfull_history_raw/locomo_mc10/full/comparison')
        paths['finetuned'] = folder
        loaded = {name: load(path, active, rotation=rotations == 4) for name, path in paths.items()}
        if any(set(rows) != active for rows in loaded.values()):
            raise ValueError('Compared runs cover different active questions')
        ids = sorted(active)
        groups = np.array([loaded['finetuned'][q][0] for q in ids])
        if any([loaded[name][q][0] for q in ids] != groups.tolist() for name in loaded):
            raise ValueError('Compared persona/conversation identities differ')
        correct = {name: np.array([rows[q][1] for q in ids], dtype=float) for name, rows in loaded.items()}
        members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        draws = [np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))]) for _ in range(5000)]
        entry = dict(questions=len(ids), groups=len(members), answer_orders=rotations,
                     accuracy={name: float(c.mean()) for name, c in correct.items()}, paired={})
        for name in ['reference', 'graded', 'full']:
            diff = correct['finetuned'] - correct[name]
            boot = [diff[d].mean() for d in draws]
            entry['paired'][f'finetuned - {name}'] = dict(difference=float(diff.mean()),
                 ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                 gained=int((diff > 0).sum()), lost=int((diff < 0).sum()))
        report['datasets'][dataset] = entry
        parts.append((correct, draws))
    for kind in ['macro', 'pooled']:
        def combine(values):
            return float(np.mean([v.mean() for v in values])) if kind == 'macro' else float(np.concatenate(values).mean())
        entry = dict(accuracy={name: combine([c[name] for c, _ in parts]) for name in ['finetuned', 'reference', 'graded', 'full']}, paired={})
        for name in ['reference', 'graded', 'full']:
            boot = [combine([(c['finetuned'] - c[name])[d[k]] for c, d in parts]) for k in range(5000)]
            entry['paired'][f'finetuned - {name}'] = dict(
                difference=combine([c['finetuned'] - c[name] for c, _ in parts]),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
        report[kind] = entry
    write_json(OUTPUT / 'comparison.json', report)
    lines = ['# Frozen fine-tuned ranker transfer evaluation', '',
             'V1 and V2 use four cyclic answer orders; LoCoMo uses one pass. Selector memory budget: 2,048 tokens. MMR is not applied.', '',
             '| Dataset | Questions | Orders | Fine-tuned 57 | Reference 55 | Graded 55 | Full history |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, e in report['datasets'].items():
        a = e['accuracy']
        lines.append(f"| {name} | {e['questions']} | {e['answer_orders']} | {a['finetuned']:.2%} | {a['reference']:.2%} | {a['graded']:.2%} | {a['full']:.2%} |")
    for kind in ['macro', 'pooled']:
        a = report[kind]['accuracy']
        lines.append(f"| {kind} | | mixed | {a['finetuned']:.2%} | {a['reference']:.2%} | {a['graded']:.2%} | {a['full']:.2%} |")
    lines += ['', 'Intervals and paired gained/lost question counts are in comparison.json. Baselines are the saved original runs, on identical active question IDs.',
              'The new encoder was fine-tuned on A and the ranker/adapter on B; older 55-feature baselines used the full original training cohort. This comparison does not isolate component effects.']
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
