"""Fuse the current V2 training rankings and measure exact-reference retrieval.

python scripts/diagnose_retrieval.py
RRF scores: data/rrf/v2/scores/train.jsonl
Metrics, positive ranks and histogram: results/selector_diagnostics_single_message/v2/train/
All input files are read-only. No models or GPU are used.
"""
import argparse
from collections import Counter
import csv
from importlib.metadata import version
from itertools import zip_longest
import json
import math
import os
from pathlib import Path

import numpy as np

from raw_datasets import ROOT, sha256

METHODS = ('bm25', 'cosine', 'rrf')
DEFAULT_K = (5, 10, 20, 30, 50)


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'{path}:{number}: invalid JSON') from error


def validate_constant(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError('RRF constant must be finite and nonnegative')


def fuse_question(labels, bm25, cosine, constant=60):
    """Use all candidates; labels select diagnostic positives, never fusion scores."""
    validate_constant(constant)
    for field in ('example_id', 'dataset', 'split', 'history_id', 'memory_count'):
        if not labels[field] == bm25[field] == cosine[field]:
            raise ValueError(f'Question metadata/order mismatch for {field}: {labels["example_id"]}')
    if labels['dataset'] != 'v2' or labels['split'] != 'train':
        raise ValueError('Expected current V2 training rows')
    members = labels['memories']
    count = len(members)
    ids = [m['message_id'] for m in members]
    if not count or count != labels['memory_count'] or len(set(ids)) != count:
        raise ValueError('Duplicate or incomplete labeled candidate pool')
    schema = labels.get('schema_version', 2)
    if type(schema) is not int or schema not in (2, 3):
        raise ValueError('Unsupported reference-label schema')
    positive_label = 2 if schema == 3 else 1
    if any(type(m['label']) is not int or m['label'] not in (0, positive_label) for m in members):
        raise ValueError(f'Memory labels must be integer 0 or {positive_label} for schema {schema}')
    positives = [m['message_id'] for m in members if m['label'] == positive_label]
    if not positives:
        raise ValueError(f'{labels["example_id"]}: no positive reference memories')
    source_order = {mid: i for i, mid in enumerate(ids)}
    source_indices = {m['message_id']: m['message_index'] for m in members}
    source_roles = {m['message_id']: m['role'] for m in members}
    if (any(type(index) is not int or index < 0 for index in source_indices.values())
            or len(set(source_indices.values())) != count
            or any(role not in ('user', 'assistant') for role in source_roles.values())):
        raise ValueError('Expected one original user or assistant message per memory')
    if ids != sorted(ids, key=source_indices.get):
        raise ValueError('Labeled candidates must preserve chronological source order')
    rankings, indexed = {}, {}
    for method, row, score_field in (('bm25', bm25, 'bm25_score'),
                                      ('cosine', cosine, 'embeddinggemma2_cosine')):
        candidates = row['memories']
        by_id = {m['message_id']: m for m in candidates}
        if len(candidates) != count or len(by_id) != count or set(by_id) != set(ids):
            raise ValueError(f'{method}: duplicate or missing candidates; full rankings are required')
        if (any(type(m['rank']) is not int for m in candidates)
                or [m['rank'] for m in candidates] != list(range(1, count + 1))):
            raise ValueError(f'{method}: ranks must be contiguous and one-based')
        if any(not math.isfinite(m[score_field]) for m in candidates):
            raise ValueError(f'{method}: nonfinite score')
        for mid, memory in by_id.items():
            if memory['message_index'] != source_indices[mid] or memory['role'] != source_roles[mid]:
                raise ValueError(f'{method}: inconsistent source indices for {mid}')
        expected = sorted(ids, key=lambda mid: (-by_id[mid][score_field], source_order[mid]))
        if [m['message_id'] for m in candidates] != expected:
            raise ValueError(f'{method}: ranking differs from descending scores/source-order ties')
        rankings[method] = {mid: by_id[mid]['rank'] for mid in ids}
        indexed[method] = by_id
    fused_scores = {mid: 1 / (constant + rankings['bm25'][mid])
                          + 1 / (constant + rankings['cosine'][mid]) for mid in ids}
    fused_ids = sorted(ids, key=lambda mid: (-fused_scores[mid], source_order[mid]))
    rankings['rrf'] = {mid: rank for rank, mid in enumerate(fused_ids, 1)}
    fused = [{key: indexed['bm25'][mid][key] for key in
              ('message_id', 'message_index', 'role')}
             | {'rank': rankings['rrf'][mid], 'rrf_score': fused_scores[mid],
                'bm25_rank': rankings['bm25'][mid], 'cosine_rank': rankings['cosine'][mid]}
             for mid in fused_ids]
    output = {field: labels[field] for field in ('example_id', 'dataset', 'split', 'history_id', 'memory_count')}
    output['memories'] = fused
    positive_rows = [{'example_id': labels['example_id'], 'history_id': labels['history_id'], 'message_id': mid,
                      **{f'{method}_rank': rankings[method][mid] for method in METHODS}}
                     for mid in positives]
    return output, positive_rows


def question_metrics(positive_ranks, k_values):
    if not positive_ranks or any(rank < 1 for rank in positive_ranks):
        raise ValueError('Positive ranks must be a nonempty list of one-based positions')
    result = {}
    for k in k_values:
        found = sum(rank <= k for rank in positive_ranks)
        result[str(k)] = {'recall': found / len(positive_ranks), 'hit': int(found > 0),
                          'full_reference_coverage': int(found == len(positive_ranks))}
    return result


class RankSummary:
    def __init__(self, k_values):
        self.k_values, self.questions, self.positions = k_values, 0, []
        self.sums = {str(k): Counter(recall=0.0, hit=0, full_reference_coverage=0) for k in k_values}

    def add(self, positive_ranks):
        metrics = question_metrics(positive_ranks, self.k_values)
        self.questions += 1
        self.positions.extend(positive_ranks)
        for k, values in metrics.items():
            self.sums[k].update(values)
        return metrics

    def report(self):
        if not self.questions:
            raise ValueError('No eligible questions to summarize')
        return {'positive_memories': len(self.positions), 'positive_rank_median': float(np.median(self.positions)),
                'positive_rank_p90': float(np.percentile(self.positions, 90, method='linear')),
                'positive_rank_min': min(self.positions), 'positive_rank_max': max(self.positions),
                'at_k': {k: {name: value / self.questions for name, value in values.items()}
                         for k, values in self.sums.items()}}


def plot_histogram(path, summaries, root):
    cache = root / '.cache/matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    maximum = max(max(summary.positions) for summary in summaries.values())
    bins = np.arange(0.5, maximum + 1.5)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, sharey=True, layout='constrained')
    for ax, method, color in zip(axes, METHODS, ('#476eab', '#198b76', '#b56a29'), strict=True):
        summary = summaries[method]
        stats = summary.report()
        ax.hist(summary.positions, bins=bins, color=color, linewidth=0)
        ax.axvline(stats['positive_rank_median'], color='#1e293b', linestyle='--', linewidth=1.2,
                   label=f"Median {stats['positive_rank_median']:g}")
        ax.axvline(stats['positive_rank_p90'], color='#be3455', linestyle=':', linewidth=1.5,
                   label=f"90th percentile {stats['positive_rank_p90']:g}")
        ax.set_title(method.upper(), loc='left', fontweight='bold')
        ax.set_ylabel('Positive memories')
        ax.set_xlim(0.5, maximum + 0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper right', frameon=False)
    axes[-1].set_xlabel('Rank within the question (1 is best); one bin per rank')
    fig.suptitle(f'V2 training: all {len(summaries["rrf"].positions):,} positive memory positions', fontweight='bold')
    fig.savefig(path, format='png', dpi=160)
    plt.close(fig)


def markdown_report(report):
    lines = ['# Training retrieval diagnostics', '',
             f"{report['questions']:,} questions, {report['candidate_memories']:,} question-memory examples, "
             f"{report['positive_memories']:,} positive memories. RRF constant: {report['rrf_constant']:g}.", '',
             'Each positive is one complete original reference message. Metrics below average questions equally. '
             'Rank percentiles pool every positive occurrence; percentile interpolation is linear.', '',
             '| Method | Median positive rank | 90th percentile |', '| --- | ---: | ---: |']
    for method, stats in report['methods'].items():
        lines.append(f"| {method} | {stats['positive_rank_median']:g} | {stats['positive_rank_p90']:g} |")
    lines += ['', '| Method | k | Recall@k | Hit@k | Full-reference coverage@k |',
              '| --- | ---: | ---: | ---: | ---: |']
    for method, stats in report['methods'].items():
        for k, values in stats['at_k'].items():
            lines.append(f"| {method} | {k} | {values['recall']:.2%} | {values['hit']:.2%} | {values['full_reference_coverage']:.2%} |")
    lines += ['', 'Recall@k is the fraction of a question\'s positive memories retrieved in its top k; '
              'Hit@k indicates at least one positive; full-reference coverage indicates all positive memories. '
              'Cutoffs larger than the candidate pool include the whole pool.', '',
              'Score ties use original chronological memory order. RRF uses all candidates and depends only on '
              'the two input rankings. Labels are applied after ranking. Duplicate/missing candidates, '
              'nonfinite scores, inconsistent ranks, missing questions and zero-positive questions cause errors.', '',
              'Files: metrics.csv, per_question.jsonl, positive_ranks.jsonl, positive_rank_histogram.csv, summary.json.']
    if report['histogram_plot']:
        lines += ['', '![Positive-rank histograms](positive_rank_histogram.png)']
    return '\n'.join(lines) + '\n'


def run(*, root=ROOT, constant=60, k_values=DEFAULT_K, output_root=None, rrf_root=None, make_plot=True,
        feature_root=None):
    validate_constant(constant)
    k_values = sorted(set(k_values))
    if not k_values or any(type(k) is not int or k <= 0 for k in k_values):
        raise ValueError('Evaluation cutoffs must be positive integers')
    root = Path(root).resolve()
    output = ((Path(output_root) if output_root is not None else root / 'results/selector_diagnostics_single_message') / 'v2/train').resolve()
    rrf_path = ((Path(rrf_root) if rrf_root is not None else root / 'data/rrf') / 'v2/scores/train.jsonl').resolve()
    if not output.is_relative_to((root / 'results/selector_diagnostics_single_message').resolve()):
        raise ValueError('Diagnostic outputs must stay under results/selector_diagnostics_single_message/')
    if not rrf_path.is_relative_to((root / 'data/rrf').resolve()):
        raise ValueError('RRF scores must stay under data/rrf/')
    feature_root = Path(feature_root).resolve() if feature_root is not None else root / 'data/ranker55/v2/train'
    paths = {'train_ids': root / 'data/splits/v2/train_ids.json',
             'labels': feature_root / 'reference_labels/train.jsonl',
             'labels_report': feature_root / 'reference_labels/report.json',
             'bm25': feature_root / 'bm25/scores/train.jsonl',
             'bm25_report': feature_root / 'bm25/report.json',
             'cosine': feature_root / 'cosine/scores/train.jsonl',
             'cosine_report': feature_root / 'cosine/report.json'}
    fingerprints = {name: sha256(path) for name, path in paths.items()}
    active_list = json.loads(paths['train_ids'].read_text(encoding='utf-8'))
    active = set(active_list)
    if not active or len(active) != len(active_list):
        raise ValueError('Training IDs must be nonempty and unique')
    parents = {name: json.loads(paths[f'{name}_report'].read_text(encoding='utf-8')) for name in METHODS[:2] + ('labels',)}
    if any(parent.get('memory_unit') != 'single_message' for parent in parents.values()):
        raise ValueError('All rankings and labels must use single_message memories; regenerate stale inputs')
    if parents['labels']['files_sha256']['train.jsonl'] != fingerprints['labels']:
        raise ValueError('Label file hash differs from its report')
    for name, parent in parents.items():
        if (parent['source_sha256'].get('data/splits/v2/train_ids.json') != fingerprints['train_ids']
                or parent['splits']['train']['questions'] != len(active)):
            raise ValueError(f'{name}: stale training membership or question count')
    for first, second in (('labels', 'bm25'), ('labels', 'cosine'), ('bm25', 'cosine')):
        a, b = parents[first]['source_sha256'], parents[second]['source_sha256']
        if any(a[name] != b[name] for name in a.keys() & b.keys()):
            raise ValueError(f'{first}/{second}: source provenance differs')
    output.mkdir(parents=True, exist_ok=True)
    rrf_path.parent.mkdir(parents=True, exist_ok=True)
    staged = {}

    def target(path):
        temporary = path.with_suffix(path.suffix + '.tmp')
        staged[path] = temporary
        return temporary

    summaries = {method: RankSummary(k_values) for method in METHODS}
    seen, candidate_count, zero_scores, zero_questions = set(), 0, 0, 0
    try:
        with target(rrf_path).open('w', encoding='utf-8', newline='\n') as fused_stream, \
             target(output / 'positive_ranks.jsonl').open('w', encoding='utf-8', newline='\n') as positive_stream, \
             target(output / 'per_question.jsonl').open('w', encoding='utf-8', newline='\n') as question_stream:
            for label_row, bm25_row, cosine_row in zip_longest(rows(paths['labels']), rows(paths['bm25']), rows(paths['cosine'])):
                if any(row is None for row in (label_row, bm25_row, cosine_row)):
                    raise ValueError('Input ranking/label files have different question counts')
                example_id = label_row['example_id']
                if example_id not in active or example_id in seen:
                    raise ValueError(f'Duplicate or inactive training question: {example_id}')
                fused, positives = fuse_question(label_row, bm25_row, cosine_row, constant)
                seen.add(example_id)
                candidate_count += fused['memory_count']
                zeros = sum(m['bm25_score'] == 0 for m in bm25_row['memories'])
                zero_scores += zeros
                zero_questions += int(zeros > 0)
                question = {'example_id': example_id, 'history_id': fused['history_id'],
                            'candidate_count': fused['memory_count'], 'positive_count': len(positives), 'methods': {}}
                for method in METHODS:
                    positions = [row[f'{method}_rank'] for row in positives]
                    question['methods'][method] = {'positive_ranks': positions, 'at_k': summaries[method].add(positions)}
                fused_stream.write(json.dumps(fused, allow_nan=False) + '\n')
                question_stream.write(json.dumps(question, allow_nan=False) + '\n')
                for row in positives:
                    positive_stream.write(json.dumps(row, allow_nan=False) + '\n')
                if len(seen) % 2000 == 0:
                    print(f'{len(seen):,}/{len(active):,} training questions processed.', flush=True)
        if seen != active:
            raise ValueError('Input files do not cover every active training question')
        if (candidate_count != parents['bm25']['splits']['train']['saved_bm25_scores']
                or candidate_count != parents['cosine']['splits']['train']['saved_cosine_scores']
                or len(summaries['rrf'].positions) != parents['labels']['splits']['train']['positive_memories']):
            raise ValueError('Processed candidate/positive counts differ from input reports')
        report = {'schema_version': 2, 'memory_unit': 'single_message', 'dataset': 'v2', 'split': 'train', 'questions': len(seen),
                  'candidate_memories': candidate_count, 'positive_memories': len(summaries['rrf'].positions),
                  'rrf_constant': constant, 'rrf_formula': '1/(c+bm25_rank) + 1/(c+cosine_rank)',
                  'cutoffs': k_values, 'rank_base': 1, 'tie_policy': 'original chronological source memory order',
                  'metric_aggregation': 'unweighted mean across questions; every question has at least one positive',
                  'positive_rank_aggregation': 'pooled positions of all positive memory occurrences',
                  'percentile_method': 'numpy linear interpolation', 'histogram_bin_width': 1,
                  'bm25_zero_score_candidates': zero_scores, 'bm25_questions_with_zero_scores': zero_questions,
                  'bm25_definition': parents['bm25'].get('score_definition', 'unspecified in parent report'),
                  'methods': {method: summary.report() for method, summary in summaries.items()},
                  'histogram_plot': make_plot, 'script_sha256': sha256(Path(__file__)),
                  'source_sha256': {path.relative_to(root).as_posix(): fingerprints[name] for name, path in paths.items()},
                  'packages': {'numpy': np.__version__}}
        with target(output / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('method', 'k', 'recall_at_k', 'hit_at_k', 'full_reference_coverage_at_k', 'questions'))
            for method, stats in report['methods'].items():
                for k, values in stats['at_k'].items():
                    writer.writerow((method, k, values['recall'], values['hit'], values['full_reference_coverage'], len(seen)))
        histograms = {method: Counter(summary.positions) for method, summary in summaries.items()}
        with target(output / 'positive_rank_histogram.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('rank', *METHODS))
            for rank in range(1, max(max(hist) for hist in histograms.values()) + 1):
                writer.writerow((rank, *(histograms[method][rank] for method in METHODS)))
        if make_plot:
            plot_histogram(target(output / 'positive_rank_histogram.png'), summaries, root)
            report['packages']['matplotlib'] = version('matplotlib')
        target(output / 'REPORT.md').write_text(markdown_report(report), encoding='utf-8')
        report['files_sha256'] = {path.relative_to(root).as_posix(): sha256(temporary) for path, temporary in staged.items()}
        target(output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if any(sha256(path) != fingerprints[name] for name, path in paths.items()):
            raise ValueError('An input changed during processing; outputs were not published')
        for path, temporary in staged.items():
            temporary.replace(path)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    print(f'Completed {len(seen):,} questions and {candidate_count:,} RRF scores. Report: {output / "REPORT.md"}', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rrf-constant', type=float, default=60)
    parser.add_argument('--k', nargs='+', type=int, default=list(DEFAULT_K))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--rrf-root', type=Path)
    parser.add_argument('--feature-root', type=Path, help='Bundle from build_ranker_features.py; default data/ranker55/v2/train')
    parser.add_argument('--no-plot', action='store_true', help='Write the exact histogram CSV without a PNG')
    args = parser.parse_args(argv)
    run(constant=args.rrf_constant, k_values=args.k, output_root=args.output_root,
        rrf_root=args.rrf_root, make_plot=not args.no_plot, feature_root=args.feature_root)


if __name__ == '__main__':
    main()
