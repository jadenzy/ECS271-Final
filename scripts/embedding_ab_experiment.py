"""Persona-disjoint embedding/ranker study: fine-tune A, fit both rankers B, compare val.

python scripts/embedding_ab_experiment.py --stage prepare
python scripts/embedding_ab_experiment.py --stage finetune
python scripts/embedding_ab_experiment.py --stage features
python scripts/embedding_ab_experiment.py --stage compare
python scripts/embedding_ab_experiment.py --stage all
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import build_ranker_features as builder
import ranker55 as recipe
from raw_datasets import ROOT, sha256
from tune_ranker55 import Bundle, aggregate, score_questions, validation_targets, write_json

OUTPUT = ROOT / 'results/embedding_ab/v2/val'
FEATURES = ROOT / 'data/ranker55/embedding_ab'
MODELS = ROOT / 'models/embedding_ab'
ENCODER = MODELS / 'encoder_half_a'
SEED = 271


def partition(qids, personas, *, count=9000, seed=SEED):
    """Whole-persona halves; sample A questions only after assigning all personas."""
    unique = np.unique(personas)
    if len(unique) < 2:
        raise ValueError('Need at least two personas')
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    a = np.isin(personas, order[:len(order)//2])
    if a.sum() < count and (~a).sum() >= count:
        a = ~a
    if a.sum() < count:
        raise ValueError('Neither persona half has enough questions')
    selected = np.sort(rng.choice(np.flatnonzero(a), count, replace=False))
    qids = np.asarray(qids)
    return dict(seed=seed, fine_tune_question_ids=qids[selected].tolist(),
                a_question_ids=qids[a].tolist(), b_question_ids=qids[~a].tolist(),
                a_personas=np.unique(personas[a]).tolist(), b_personas=np.unique(personas[~a]).tolist())


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with Bundle(ROOT / 'data/ranker55/v2/train/features.npz') as data:
        split = partition(data.read('question_ids'), data.read('persona_ids'))
        split['training_bundle_sha256'] = sha256(data.path)
    active = json.loads((ROOT / 'data/splits/v2/train_ids.json').read_text())
    if set(active) != set(split['a_question_ids'] + split['b_question_ids']):
        raise ValueError('Training bundle differs from active training questions')
    if set(split['a_personas']) & set(split['b_personas']):
        raise ValueError('Persona halves overlap')
    path = OUTPUT / 'partition.json'
    if path.exists() and json.loads(path.read_text()) != split:
        raise ValueError('An existing experiment has a different partition')
    write_json(path, split)
    write_json(OUTPUT / 'a_finetune_ids.json', split['fine_tune_question_ids'])
    write_json(OUTPUT / 'b_ranker_ids.json', split['b_question_ids'])
    print(f"A: {len(split['a_personas'])} personas, {len(split['a_question_ids'])} questions; "
          f"fine-tune {len(split['fine_tune_question_ids'])}. B: {len(split['b_personas'])} personas, "
          f"{len(split['b_question_ids'])} questions", flush=True)
    return split


def finetune():
    split = prepare()
    complete = OUTPUT / 'encoder.json'
    if complete.exists():
        verify_encoder()
        print('Reusing completed saved A encoder', flush=True)
        return
    builder.verified_model(ROOT)
    command = [sys.executable, '-u', '-B', str(ROOT / 'scripts/finetune_embedding_pilot.py'),
               '--train-questions', '9000', '--question-ids', str(OUTPUT / 'a_finetune_ids.json'),
               '--batch-size', '12', '--epochs', '1', '--evaluate-every', '0', '--skip-evaluation',
               '--label', 'ab_half_a', '--save-model', str(ENCODER),
               '--checkpoint', str(ROOT / '.cache/embedding_ab/training.pt')]
    before = time.perf_counter()
    subprocess.run(command, check=True, cwd=ROOT)
    files = {p.relative_to(ENCODER).as_posix(): sha256(p) for p in sorted(ENCODER.rglob('*')) if p.is_file()}
    info = dict(repo='local/embeddinggemma2-v2-half-a', revision=builder.fingerprint(files))
    write_json(complete, dict(model=info, files=files, partition_sha256=sha256(OUTPUT / 'partition.json'),
                              fine_tune_questions=len(split['fine_tune_question_ids']),
                              elapsed_seconds=time.perf_counter()-before,
                              source_model=recipe.EMBEDDING_RECIPE, command=command,
                              pilot_script_sha256=sha256(ROOT / 'scripts/finetune_embedding_pilot.py')))


def verify_encoder():
    record = json.loads((OUTPUT / 'encoder.json').read_text())
    if record['partition_sha256'] != sha256(OUTPUT / 'partition.json'):
        raise ValueError('Encoder partition changed')
    for path, expected in record['files'].items():
        if sha256(ENCODER / path) != expected:
            raise ValueError('Saved encoder changed')
    actual = json.loads((ROOT / 'results/embedding_finetune_pilot/ab_half_a/report.json').read_text())
    if actual['training']['questions'] != 9000 or not actual['config']['skip_evaluation']:
        raise ValueError('Fine-tuning did not complete the specified 9,000-question run')
    return record['model']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'finetune', 'baseline', 'features', 'compare', 'all'], default='all')
    args = parser.parse_args()
    for name, operation in [('prepare', prepare), ('finetune', finetune), ('baseline', baseline),
                            ('features', features), ('compare', compare)]:
        if args.stage in [name, 'all']:
            operation()


def subset_bundle(source, destination, wanted):
    """Copy complete B groups verbatim; no statistics or labels are refitted."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Bundle(source) as data:
        qids = data.read('question_ids')
        if not set(wanted) <= set(qids.tolist()):
            raise ValueError('Unknown B question')
        groups = np.flatnonzero(np.isin(qids, wanted))
        offsets = data.read('offsets')
        rows = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in groups])
        new_offsets = np.r_[0, np.cumsum(np.diff(offsets)[groups])].astype(np.int64)
        metadata = {**data.metadata, 'questions': len(groups), 'row_count': len(rows),
                    'experiment_cohort': 'persona half B', 'parent_features_sha256': sha256(source)}
        arrays = {key: data.read(key)[rows] for key in
                  ['X', 'y', 'reference_labels', 'judged_labels', 'message_ids', 'message_indices', 'content_token_lengths']}
        arrays.update(offsets=new_offsets, question_ids=qids[groups], persona_ids=data.read('persona_ids')[groups],
                      feature_names=data.read('feature_names'), metadata_json=np.asarray(json.dumps(metadata)))
        temporary = destination.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)
        temporary.replace(destination)


def baseline():
    split = prepare()
    path = FEATURES / 'original/v2/train_b/features.npz'
    if not path.exists():
        subset_bundle(ROOT / 'data/ranker55/v2/train/features.npz', path, split['b_question_ids'])


def features():
    prepare()
    info = verify_encoder()
    counts, total, lock = builder.subtlex.load_prior(ROOT / 'data/external/subtlex_us/word_frequencies.json',
                                                   ROOT / 'configs/subtlex_us.lock.json')
    mixture = builder.subtlex.Mixture(counts, total, builder.subtlex.configuration(.5, lock))
    statistics = {'schema_version': 2, 'feature_recipe': mixture.config}
    encoder = builder.Encoder(ENCODER, 'cuda')
    for split in ['train', 'val']:
        output = FEATURES / 'finetuned/v2' / split
        if (output / 'features.npz').exists() and (output / 'report.json').exists():
            record = json.loads((output / 'report.json').read_text())
            if record['embedding_recipe']['model'] != info or record['files_sha256']['features.npz'] != sha256(output / 'features.npz'):
                raise ValueError('Completed fine-tuned bundle has changed')
            print(f'Reusing completed {split} feature bundle', flush=True)
            continue
        tasks = builder.prepare_inputs(root=ROOT, datasets=['v2'], splits=[split],
                   question_ids=OUTPUT / 'b_ranker_ids.json' if split == 'train' else None)
        task = tasks[0]
        needed = {mid for q in task[2] for mid in q['memory_ids']}
        task = (*task[:3], {mid: m for mid, m in task[3].items() if mid in needed}, *task[4:])
        # The normal CLI remains pinned to the base model. The lower-level builder
        # records this local checkpoint's content hash as a distinct encoder identity.
        builder.build_bundle(task, statistics, encoder, info, output, batch_size=128,
                             batch_tokens=16000, feature_workers=4, mixture=mixture)
    print('Fine-tuned B and validation features complete', flush=True)


def compare():
    baseline()
    fine_train = FEATURES / 'finetuned/v2/train/features.npz'
    from embedding_ab_math import fit
    partition_record = json.loads((OUTPUT / 'partition.json').read_text())
    for path in [FEATURES / 'original/v2/train_b/features.npz', fine_train]:
        with Bundle(path) as data:
            if (set(data.read('question_ids')) != set(partition_record['b_question_ids'])
                    or set(data.read('persona_ids')) != set(partition_record['b_personas'])):
                raise ValueError('Mathematical adapters/rankers must fit exactly B')
    for name, train, val in [('original', FEATURES / 'original/v2/train_b/features.npz', ROOT / 'data/ranker55/v2/val/features.npz'),
                             ('finetuned', fine_train, FEATURES / 'finetuned/v2/val/features.npz')]:
        if not (OUTPUT / f'{name}_ranker57.json').exists():
            fit(train, val, MODELS / f'{name}_math', OUTPUT, name)
    with ExitStack() as stack:
        original_train = stack.enter_context(Bundle(FEATURES / 'original/v2/train_b/features.npz'))
        tuned_train = stack.enter_context(Bundle(fine_train))
        original = stack.enter_context(Bundle(ROOT / 'data/ranker55/v2/val/features.npz'))
        tuned = stack.enter_context(Bundle(FEATURES / 'finetuned/v2/val/features.npz'))
        partition_ids = json.loads((OUTPUT / 'b_ranker_ids.json').read_text())
        if set(original_train.read('question_ids')) != set(partition_ids):
            raise ValueError('Ranker inputs differ from the fixed B partition')
        for key in ['question_ids', 'persona_ids', 'message_ids', 'offsets', 'y']:
            if not np.array_equal(original_train.read(key), tuned_train.read(key)):
                raise ValueError(f'B cohorts/targets differ: {key}')
        for key in ['question_ids', 'persona_ids', 'message_ids', 'offsets']:
            if not np.array_equal(original.read(key), tuned.read(key)):
                raise ValueError(f'Validation identities differ: {key}')
        # All lexical, length, role and position features must remain bit-identical.
        dependent = [name.startswith(('cosine', 'dense_chunk')) or 'cosine' in name
                     or name in ['winning_fusion', 'qweight_fusion'] for name in recipe.FEATURE_NAMES]
        for a, b in [(original_train, tuned_train), (original, tuned)]:
            for start in range(0, len(a.read('X')), 65536):
                if not np.array_equal(a.read('X')[start:start+65536, np.logical_not(dependent)],
                                      b.read('X')[start:start+65536, np.logical_not(dependent)]):
                    raise ValueError('An embedding-independent feature changed')
        targets = validation_targets(original, ROOT, 'val')
        split = json.loads((OUTPUT / 'partition.json').read_text())
        if set(original.read('persona_ids')) & set(split['a_personas'] + split['b_personas']):
            raise ValueError('Validation personas overlap A/B')
        scores = {}
        for name, data in [('original', original), ('finetuned', tuned)]:
            meta = json.loads((OUTPUT / f'{name}_ranker57.json').read_text())
            path = MODELS / f'{name}_math/ranker57.txt'
            score_path = MODELS / f'{name}_math/validation_scores.npy'
            if (sha256(path) != meta['model_sha256'] or data.metadata['embedding_recipe'] != meta['embedding_recipe']
                    or sha256(score_path) != meta['score_file_sha256']
                    or sha256(data.path) != meta['validation_features_sha256']):
                raise ValueError('Ranker/validation encoder identity mismatch')
            scores[name + '_cosine'] = np.asarray(data.read('X')[:, 0])
            scores[name + '_ranker'] = np.load(score_path)
        report = dict(status='complete', dataset='v2', evaluation_split='val', validation_questions=len(targets['questions']),
                      test_used=False, fine_tune_questions=9000, ranker_training_questions=len(split['b_question_ids']),
                      cohort='Both 57-feature rankers fit the same B personas, at fixed identical settings.',
                      mathematical_features=['learned_bilinear_similarity', 'background_corrected_cosine'],
                      redundancy_selection_used=False, methods={})
        per_question = []
        for name, values in scores.items():
            report['methods'][name], rows = retrieval_metrics(values, targets)
            per_question.extend({'method': name, **r} for r in rows)
        from option_rotation_pilot import bootstrap
        groups = np.array([q['persona_id'] for q in targets['questions']])
        report['paired_persona_bootstrap'] = {}
        for a, b in [('finetuned_cosine', 'original_cosine'), ('finetuned_ranker', 'original_ranker')]:
            ar = [r for r in per_question if r['method'] == a]
            br = [r for r in per_question if r['method'] == b]
            report['paired_persona_bootstrap'][f'{a} - {b}'] = {}
            for metric in ['hit_at_10', 'ndcg_at_10', 'reference_recall_at_15', 'every_visible_reference_top15']:
                difference = np.array([float(x[metric]) - float(y[metric]) for x, y in zip(ar, br)])
                report['paired_persona_bootstrap'][f'{a} - {b}'][metric] = dict(
                    difference=float(difference.mean()), ci95=bootstrap(difference, groups))
        write_json(OUTPUT / 'report.json', report)
        with (OUTPUT / 'per_question.jsonl').open('w', encoding='utf-8') as stream:
            for row in per_question:
                stream.write(json.dumps(row) + '\n')
        lines = ['# Persona-disjoint embedding/ranker comparison', '',
                 'Fine-tune 9,000 questions from A; both 57-feature rankers train on B; evaluate on 909 validation questions. No test run.',
                 'Both rankers include learned question/message similarity and generic-message correction, with five persona-held-out folds inside B. Redundancy-aware selection is deferred.', '',
                 '| Method | Hit@10 | NDCG@10 | Recall@15 | Every visible reference@15 | Non-forget type-macro Hit@10 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, m in report['methods'].items():
            lines.append(f"| {name} | {m['all_questions']['hit_at_10']:.2%} | {m['all_questions']['ndcg_at_10']:.2%} | "
                         f"{m['reference_recall_at_15']:.2%} | {m['every_visible_reference_top15']:.2%} | "
                         f"{m['nonforget_type_macro']['hit_at_10']:.2%} |")
        (OUTPUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print('\n'.join(lines), flush=True)


def retrieval_metrics(scores, targets):
    rows = score_questions(scores, scores, targets)
    recalls, complete, worst = [], [], []
    for i in range(len(targets['questions'])):
        a, b = map(int, targets['offsets'][i:i+2])
        positive = targets['positive'][a:b]
        order = np.argsort(-scores[a:b], kind='stable')
        ranks = np.empty(b-a, dtype=int)
        ranks[order] = np.arange(1, b-a+1)
        ref = ranks[positive]
        recalls.append(float(np.mean(ref <= 15)))
        complete.append(bool(np.all(ref <= 15)))
        worst.append(int(ref.max()))
        rows[i].update(reference_recall_at_15=recalls[-1], every_visible_reference_top15=complete[-1])
    metrics = aggregate(rows)
    # 'incremental over cosine' is not comparable when each method supplies its own
    # comparator; omit those convenience fields rather than reporting a fake zero.
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in ['incremental_hit_over_cosine', 'tail_rescue_over_cosine_at_10']}
        return value
    metrics = clean(metrics)
    metrics.update(reference_recall_at_15=float(np.mean(recalls)), every_visible_reference_top15=float(np.mean(complete)),
                   median_lowest_reference_rank=float(np.median(worst)))
    for row in rows:
        row.pop('cosine_hit_at_10', None)
    return metrics, rows


if __name__ == '__main__':
    main()
