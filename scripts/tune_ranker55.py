"""Fit the reference-only 55-feature ranker and select settings on V2 val.

python scripts/tune_ranker55.py 

Validation annotations are used for evaluation/selection, never for fitting.
"""
import argparse
from contextlib import ExitStack
import csv
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np

import ranker55 as recipe
from label_reference_messages import load_references
from message_dataset import load_message_dataset
from raw_datasets import ROOT, sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class Bundle:
    """Map the builder's uncompressed NPZ arrays without copying the full matrix."""
    def __init__(self, path):
        self.path, self.arrays = Path(path), {}
        self.archive = zipfile.ZipFile(path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for value in self.arrays.values():
            if getattr(value, '_mmap', None) is not None:
                value._mmap.close()
        self.archive.close()

    def read(self, name):
        if name in self.arrays:
            return self.arrays[name]
        info = self.archive.getinfo(name + '.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            with np.load(self.path, allow_pickle=False) as data:
                value = data[name]
        else:
            with self.path.open('rb') as stream:
                stream.seek(info.header_offset)
                header = stream.read(30)
                if header[:4] != b'PK\x03\x04':
                    raise ValueError('Invalid local NPZ entry')
                nname, nextra = struct.unpack('<HH', header[26:30])
                stream.seek(nname + nextra, 1)
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError('Unsupported NPZ array version')
                shape, fortran, dtype = reader(stream)
                if dtype.hasobject:
                    raise ValueError('Object arrays are not permitted')
                value = np.memmap(self.path, dtype=dtype, mode='r', offset=stream.tell(),
                                  shape=shape, order='F' if fortran else 'C')
        self.arrays[name] = value
        return value

    @property
    def metadata(self):
        return json.loads(str(self.read('metadata_json').item()))


def parameter_grid(minimums, shapes, penalties):
    baseline = dict(min_child_samples=100, num_leaves=31, max_depth=5, reg_lambda=5.0)
    trials, seen = [], set()
    for item in [baseline, *(dict(min_child_samples=n, num_leaves=l, max_depth=d, reg_lambda=r)
                            for n, (l, d), r in itertools.product(minimums, shapes, penalties))]:
        recipe.training_parameters(**item)
        key = tuple(item[k] for k in baseline)
        if key not in seen:
            trials.append(item)
            seen.add(key)
    return trials


def validation_targets(bundle, root):
    """Bind every visible row to raw annotations, retaining incomplete questions."""
    questions, memories, sources = load_message_dataset('v2', ['val'], root=root)
    metadata = bundle.metadata
    if (metadata.get('dataset'), metadata.get('split'), metadata.get('memory_unit')) != ('v2', 'val', 'single_message'):
        raise ValueError('Validation bundle must be V2 val with single-message features')
    if any(metadata.get('source_sha256', {}).get(k) != v for k, v in sources.items()):
        raise ValueError('Validation features differ from current raw sources/splits')
    qids = bundle.read('question_ids').tolist()
    by_id = {q['example_id']: q for q in questions}
    if len(set(qids)) != len(qids) or set(qids) != set(by_id):
        raise ValueError('Validation bundle must cover every active validation question exactly once')
    questions = [by_id[qid] for qid in qids]
    refs = load_references(questions, sources, root)
    mids, offsets = bundle.read('message_ids'), bundle.read('offsets')
    if len(offsets) != len(questions) + 1 or offsets[0] != 0 or offsets[-1] != len(mids) or np.any(np.diff(offsets) <= 0):
        raise ValueError('Invalid validation offsets')
    positive = np.zeros(len(mids), dtype=bool)
    ideals, complete, missing = [], [], []
    for i, q in enumerate(questions):
        start, end = int(offsets[i]), int(offsets[i+1])
        if mids[start:end].tolist() != q['memory_ids']:
            raise ValueError(f"{q['example_id']}: validation rows differ from the complete visible pool")
        keys = {key for key in refs[q['example_id']] if key[1].strip()}
        found = set()
        for j, mid in enumerate(q['memory_ids'], start):
            key = memories[mid]['role'], memories[mid]['text']
            if key in keys:
                positive[j] = True
                found.add(key)
        absent = len(keys - found)
        ideals.append(int(positive[start:end].sum()) + absent)
        complete.append(absent == 0)
        missing.append(absent)
    return dict(questions=questions, offsets=offsets, positive=positive,
                ideal_counts=np.array(ideals), complete=np.array(complete), missing=np.array(missing))


def score_questions(scores, cosine, targets, k=10):
    discounts = 1 / np.log2(np.arange(2, k + 2))
    rows = []
    for i, q in enumerate(targets['questions']):
        start, end = map(int, targets['offsets'][i:i+2])
        positive = targets['positive'][start:end]
        order = np.argsort(-scores[start:end], kind='stable')
        dense = np.argsort(-cosine[start:end], kind='stable')
        dcg = float(np.sum(discounts[:min(k, end-start)] * positive[order[:k]]))
        ideal = float(discounts[:min(k, targets['ideal_counts'][i])].sum())
        rows.append(dict(example_id=q['example_id'], persona_id=q['persona_id'], question_type=q['question_type'],
                         complete_reference=bool(targets['complete'][i]),
                         missing_reference_keys=int(targets['missing'][i]),
                         hit_at_10=float(positive[order[:k]].any()), ndcg_at_10=dcg/ideal if ideal else 0.,
                         cosine_hit_at_10=float(positive[dense[:k]].any())))
    return rows


def aggregate(rows):
    def mean(selected):
        if not selected:
            return {'questions': 0, 'hit_at_10': None, 'ndcg_at_10': None,
                    'incremental_hit_over_cosine': None, 'tail_rescue_over_cosine_at_10': None}
        failures = [r for r in selected if not r['cosine_hit_at_10']]
        hit = float(np.mean([r['hit_at_10'] for r in selected]))
        return dict(questions=len(selected), hit_at_10=hit,
                    ndcg_at_10=float(np.mean([r['ndcg_at_10'] for r in selected])),
                    incremental_hit_over_cosine=hit-float(np.mean([r['cosine_hit_at_10'] for r in selected])),
                    tail_rescue_over_cosine_at_10=float(np.mean([r['hit_at_10'] for r in failures])) if failures else None)
    nonforget = [r for r in rows if r['question_type'] != 'ask_to_forget']
    by_type = {kind: mean([r for r in rows if r['question_type'] == kind]) for kind in sorted({r['question_type'] for r in rows})}
    def macro(selected):
        groups = [mean([r for r in selected if r['question_type'] == kind]) for kind in sorted({r['question_type'] for r in selected})]
        return {**{'questions': len(selected)}, **{key: float(np.mean([g[key] for g in groups if g[key] is not None]))
                if any(g[key] is not None for g in groups) else None for key in mean([]) if key != 'questions'}}
    return dict(all_questions=mean(rows), nonforget=mean(nonforget), by_type=by_type,
                nonforget_type_macro=macro(nonforget),
                complete_reference_nonforget_type_macro=macro([r for r in nonforget if r['complete_reference']]))


def run(train_features, val_features, output_model, output_root, trials, *, root=ROOT, threshold=.01):
    import lightgbm as lgb
    started = time.perf_counter()
    root, output, model_path = Path(root).resolve(), Path(output_root).resolve(), Path(output_model).resolve()
    if not output.is_relative_to(root / 'results') or not model_path.is_relative_to(root / 'models'):
        raise ValueError('Reports must be under results/ and model outputs under models/')
    recipe._model_output_path(model_path, train_features)
    recipe._model_output_path(model_path, val_features)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Adoption threshold must be nonnegative')
    output.mkdir(parents=True, exist_ok=True)
    trial_root = model_path.parent / 'ranker55_trials'
    trial_root.mkdir(parents=True, exist_ok=True)
    protocol = dict(status='running', dataset='v2', fit_split='train', tuning_split='val', test_used=False,
                    label_policy='binary exact-reference membership; no Qwen-generated labels',
                    baseline_parameters=recipe.PARAMETERS,
                    selection_metric='nonforget_type_macro.hit_at_10', adoption_threshold=threshold,
                    adoption_rule='strictly greater than threshold over default and no NDCG decrease',
                    selection_bias='Validation selects hyperparameters; these are development results, not final test performance.',
                    grid=trials, trials=[], output_model=str(model_path),
                    tie_policy='descending score then original chronological message order',
                    validation_missing_reference_policy='Keep all questions. Each absent unique reference key adds one ideal positive; absent evidence cannot produce a hit.',
                    script_sha256=sha256(Path(__file__)), recipe_sha256=sha256(Path(recipe.__file__)))
    write_json(output / 'report.json', protocol)
    with ExitStack() as stack:
        train = stack.enter_context(Bundle(train_features))
        val = stack.enter_context(Bundle(val_features))
        tm, vm = train.metadata, val.metadata
        if (tm.get('dataset'), tm.get('split'), tm.get('memory_unit')) != ('v2', 'train', 'single_message'):
            raise ValueError('Training bundle must be V2 train with single-message features')
        if tm.get('statistics_fingerprint') != vm.get('statistics_fingerprint') or not tm.get('statistics_fingerprint'):
            raise ValueError('Train and validation must use identical frozen training corpus statistics')
        for key, digest in tm.get('source_sha256', {}).items():
            if sha256(root / key) != digest:
                raise ValueError('Training sources changed; rebuild training features')
        train_qids = train.read('question_ids').tolist()
        expected = json.loads((root / 'data/splits/v2/train_ids.json').read_text(encoding='utf-8'))
        if len(set(train_qids)) != len(train_qids) or set(train_qids) != set(expected):
            raise ValueError('Training bundle must contain all active training questions exactly once')
        if set(train.read('persona_ids').tolist()) & set(val.read('persona_ids').tolist()):
            raise ValueError('Training and validation personas overlap')
        X, y, offsets = recipe.validate_training_data(train.read('X'), train.read('y'), train.read('offsets'), train.read('feature_names'))
        VX = val.read('X')
        if val.read('feature_names').tolist() != recipe.FEATURE_NAMES or VX.ndim != 2 or VX.shape[1] != 55:
            raise ValueError('Validation features must follow the complete 55-feature schema')
        if any(not np.isfinite(VX[i:i+65536]).all() for i in range(0, len(VX), 65536)):
            raise ValueError('Nonfinite validation features')
        targets = validation_targets(val, root)
        if len(VX) != len(targets['positive']):
            raise ValueError('Validation feature row count differs from row identities')
        protocol.update(train_questions=len(train_qids), train_rows=len(y), validation_questions=len(targets['questions']),
                        complete_reference_questions=int(targets['complete'].sum()),
                        questions_without_visible_reference=int(sum(not targets['positive'][a:b].any() for a,b in zip(targets['offsets'][:-1], targets['offsets'][1:]))),
                        missing_reference_keys=int(targets['missing'].sum()), statistics_fingerprint=tm['statistics_fingerprint'],
                        inputs_sha256={str(Path(p).resolve()): sha256(p) for p in (train_features, val_features)})
        cosine = np.asarray(VX[:, 0])
        # Reuse one binned training dataset across trials. Disable feature
        # prefiltering because min_child_samples varies between configurations.
        binning_parameters = {k:v for k,v in recipe.PARAMETERS.items() if k != 'n_estimators'}
        binning_parameters['feature_pre_filter'] = False
        dataset = lgb.Dataset(X, label=y, group=np.diff(offsets), feature_name=recipe.FEATURE_NAMES,
                              params=binning_parameters, free_raw_data=True)
        dataset.construct()
        selected = None
        best_score = None
        for number, tuning in enumerate(trials):
            before = time.perf_counter()
            params = recipe.training_parameters(**tuning)
            rounds = params.pop('n_estimators')
            params['feature_pre_filter'] = False
            print(f'Trial {number+1}/{len(trials)}: {tuning}', flush=True)
            booster = lgb.train(params, dataset, num_boost_round=rounds)
            scores = booster.predict(VX, num_threads=4, device_type='cpu')
            rows = score_questions(scores, cosine, targets)
            stats = aggregate(rows)
            primary = stats['nonforget_type_macro']
            path = trial_root / f'trial_{number:02d}.txt'
            recipe._save_model_atomic(booster, path)
            entry = dict(trial=number, parameters=tuning, metrics=stats, model_path=str(path),
                         elapsed_seconds=time.perf_counter()-before, model_sha256=sha256(path))
            protocol['trials'].append(entry)
            if number == 0:
                baseline = primary
                selected = entry
                best_score = (primary['hit_at_10'], primary['ndcg_at_10'])
                recipe._save_model_atomic(booster, model_path)
            gain = primary['hit_at_10'] - baseline['hit_at_10']
            entry['gain_over_default'] = gain
            entry['eligible_for_adoption'] = number == 0 or (gain > threshold + 1e-12 and primary['ndcg_at_10'] >= baseline['ndcg_at_10'] - 1e-12)
            score = (primary['hit_at_10'], primary['ndcg_at_10'])
            if entry['eligible_for_adoption'] and score > best_score:
                selected, best_score = entry, score
            protocol.update(completed_trials=number+1, elapsed_seconds=time.perf_counter()-started)
            write_json(output / 'report.json', protocol)
            print(f"  non-forget type-macro Hit@10={primary['hit_at_10']:.4%}, NDCG@10={primary['ndcg_at_10']:.4%}, gain={gain*100:+.3f} points", flush=True)
        winner = lgb.Booster(model_str=Path(selected['model_path']).read_text(encoding='utf-8'))
        recipe._save_model_atomic(winner, model_path)
        chosen_rows = score_questions(winner.predict(VX, num_threads=4), cosine, targets)
        temporary = output / 'per_question.jsonl.tmp'
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            for row in chosen_rows:
                stream.write(json.dumps(row) + '\n')
        temporary.replace(output / 'per_question.jsonl')
    protocol.update(status='complete', selected_trial=selected['trial'], selected_parameters=selected['parameters'],
                    selected_metrics=selected['metrics'], output_model_sha256=sha256(model_path),
                    elapsed_seconds=time.perf_counter()-started)
    write_json(output / 'report.json', protocol)
    with (output / 'trials.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['trial','min_child_samples','num_leaves','max_depth','reg_lambda','nonforget_macro_hit10','nonforget_macro_ndcg10','gain_pp','adoptable'])
        for t in protocol['trials']:
            m = t['metrics']['nonforget_type_macro']
            writer.writerow([t['trial'], *[t['parameters'][k] for k in ('min_child_samples','num_leaves','max_depth','reg_lambda')], m['hit_at_10'],m['ndcg_at_10'],t['gain_over_default']*100,t['eligible_for_adoption']])
    lines = ['# Ranker55 V2 validation tuning', '', f"Selected trial: {selected['trial']}. Model: {model_path}",
             '', 'Training uses binary original-reference labels. Validation selects hyperparameters; test was not used.',
             f"Validation: {protocol['validation_questions']} questions, {protocol['complete_reference_questions']} with complete visible references.",
             '', '| Trial | Min child | Leaves/depth | L2 | Non-forget macro Hit@10 | NDCG@10 | Gain (pp) |', '|---|---:|---|---:|---:|---:|---:|']
    for t in protocol['trials']:
        p, m = t['parameters'], t['metrics']['nonforget_type_macro']
        lines.append(f"| {t['trial']} | {p['min_child_samples']} | {p['num_leaves']}/{p['max_depth']} | {p['reg_lambda']} | {m['hit_at_10']:.2%} | {m['ndcg_at_10']:.2%} | {t['gain_over_default']*100:+.2f} |")
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-features', type=Path, default=ROOT/'data/ranker55/v2/train/features.npz')
    parser.add_argument('--val-features', type=Path, default=ROOT/'data/ranker55/v2/val/features.npz')
    parser.add_argument('--output', type=Path, default=ROOT/'models/ranker55.txt')
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/ranker55_tuning/v2/val')
    parser.add_argument('--min-child-samples', nargs='+', type=int, default=[50,100,200])
    parser.add_argument('--tree-shapes', nargs='+', default=['15:4','31:5','63:6'], help='Paired num_leaves:max_depth settings')
    parser.add_argument('--reg-lambda', nargs='+', type=float, default=[1.,5.,10.])
    parser.add_argument('--adoption-threshold', type=float, default=.01)
    args = parser.parse_args(argv)
    try:
        shapes = [tuple(map(int, text.split(':'))) for text in args.tree_shapes]
        if any(len(shape) != 2 for shape in shapes):
            raise ValueError('Each tree shape must be num_leaves:max_depth')
        trials = parameter_grid(args.min_child_samples, shapes, args.reg_lambda)
        deadline = time.monotonic()
        statistics = args.train_features.parent / 'statistics.json'
        while (not args.train_features.is_file() or (args.train_features.parent / '.features_work.npy').exists()
               or (not statistics.is_file())):
            if time.monotonic() >= deadline:
                raise ValueError('Training feature bundle is not finished; let the existing build finish first')
            print('Waiting for the existing training feature build to finish (read-only).', flush=True)
            time.sleep(min(30, max(0, deadline-time.monotonic())))
        if not args.val_features.exists():
            if args.val_features.resolve() != (ROOT/'data/ranker55/v2/val/features.npz').resolve():
                raise ValueError('Validation features must be located at the standard path: data/ranker55/v2/val/features.npz')
            command = [sys.executable, '-B', '-u', str(ROOT/'scripts/build_ranker_features.py'), '--splits', 'val',
                       '--statistics', str(statistics), '--batch-size', '16', '--batch-tokens', '2400']
            print('Building validation features with frozen training statistics.', flush=True)
            subprocess.run(command, check=True, cwd=ROOT)
        report = run(args.train_features, args.val_features, args.output, args.output_root, trials,
                     threshold=args.adoption_threshold)
        print(json.dumps({k:report[k] for k in ('status','selected_parameters','output_model','elapsed_seconds')}, indent=2))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        path = args.output_root / 'report.json'
        if path.is_file():
            state = json.loads(path.read_text(encoding='utf-8'))
            if state.get('status') == 'running':
                write_json(path, {**state, 'status': 'failed', 'error': str(error)})
        parser.error(str(error))


if __name__ == '__main__':
    main()
