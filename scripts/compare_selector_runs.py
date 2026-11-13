"""Paired Qwen-accuracy comparison of selector runs across datasets.

python scripts/compare_selector_runs.py --runs results/reference_hit/runs_overall.json --output results/reference_hit/overall.json --active-only
runs.json: {"dataset": {"label": "results/.../run_dir", ...}, ...}. Each run dir holds
predictions.jsonl. Intervals resample each dataset's groups (persona_id: personas or
conversations) with 5,000 draws; macro averages datasets equally.
--active-only keeps only questions still listed under data/splits, so runs saved before
questions were removed from a split (V2 test sensitive_info) compare on the current set.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from raw_datasets import ROOT


def load(run, active=None):
    rows = {}
    for line in (ROOT / run / 'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            if active is None or r['example_id'] in active:
                rows[r['example_id']] = (r['persona_id'], int(r['correct']), r['predicted_label'])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=271)
    parser.add_argument('--active-only', action='store_true', help='drop questions no longer in data/splits')
    args = parser.parse_args(argv)
    spec = json.loads(args.runs.read_text(encoding='utf-8'))
    active = None
    if args.active_only:
        active = {i for path in (ROOT / 'data/splits').glob('*/*_ids.json') for i in json.loads(path.read_text(encoding='utf-8'))}
    labels = list(next(iter(spec.values())))
    rng = np.random.default_rng(args.seed)
    data, report = {}, {'datasets': {}, 'draws': args.draws, 'active_only': args.active_only}
    for dataset, runs in spec.items():
        loaded = {label: load(path, active) for label, path in runs.items()}
        ids = sorted(set.intersection(*(set(v) for v in loaded.values())))
        if any(len(v) != len(ids) for v in loaded.values()):
            raise ValueError(f'{dataset}: runs cover different questions')
        groups = np.array([loaded[labels[0]][i][0] for i in ids])
        correct = {label: np.array([loaded[label][i][1] for i in ids], dtype=float) for label in labels}
        unique = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in unique]
        # Each draw resamples groups; record the question indices it selects.
        draws = [np.concatenate([members[j] for j in rng.integers(0, len(unique), len(unique))])
                 for _ in range(args.draws)]
        data[dataset] = (correct, draws)
        entry = {'questions': len(ids), 'groups': len(unique), 'accuracy': {l: float(c.mean()) for l, c in correct.items()},
                 'paired': {}}
        for a in labels:
            for b in labels:
                if a < b or a == b:
                    continue
                diff = correct[a] - correct[b]
                boot = [diff[d].mean() for d in draws]
                entry['paired'][f'{a} - {b}'] = {
                    'difference': float(diff.mean()), 'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    'gained': int((diff > 0).sum()), 'lost': int((diff < 0).sum()),
                    'same_prediction': float(np.mean([loaded[a][i][2] == loaded[b][i][2] for i in ids]))}
        report['datasets'][dataset] = entry
    for kind in ('macro', 'pooled'):
        acc, paired = {}, {}
        for l in labels:
            parts = [(c[l], d) for c, d in data.values()]
            acc[l] = float(np.mean([c.mean() for c, _ in parts])) if kind == 'macro' else \
                float(np.concatenate([c for c, _ in parts]).mean())
        for a in labels:
            for b in labels:
                if a < b or a == b:
                    continue
                def combine(vals):
                    return float(np.mean([v.mean() for v in vals])) if kind == 'macro' else float(np.concatenate(vals).mean())
                boot = [combine([(c[a] - c[b])[d[k]] for c, d in data.values()]) for k in range(args.draws)]
                paired[f'{a} - {b}'] = {'difference': combine([c[a] - c[b] for c, _ in data.values()]),
                                        'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        report[kind] = {'accuracy': acc, 'paired': paired}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for dataset, entry in report['datasets'].items():
        print(dataset, entry['questions'], {l: f'{v:.2%}' for l, v in entry['accuracy'].items()})
        for k, v in entry['paired'].items():
            print(f"   {k}: {v['difference']:+.2%} ({v['ci95'][0]:+.2%} to {v['ci95'][1]:+.2%}); +{v['gained']}/-{v['lost']}")
    for kind in ('macro', 'pooled'):
        print(kind, {l: f'{v:.2%}' for l, v in report[kind]['accuracy'].items()})
        for k, v in report[kind]['paired'].items():
            print(f"   {k}: {v['difference']:+.2%} ({v['ci95'][0]:+.2%} to {v['ci95'][1]:+.2%})")


if __name__ == '__main__':
    main()
