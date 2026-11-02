"""Join Qwen candidate judgments to the V2 training features for graded ranker training.

python scripts/build_graded_training.py --all-rows
python scripts/tune_ranker55.py --train-features data/ranker55/v2/train_graded_all/features.npz --output models/qwen_graded_all/ranker55.txt --output-root results/ranker55_tuning_graded_all/v2/val

Reads data/ranker55/v2/train/features.npz and data/qwen_labels/v2/train.jsonl. Grades are original
references (2), Qwen-useful candidates (1) and Qwen-not-useful candidates (0). By default only judged
rows are kept: unjudged messages are dropped and question offsets are rebuilt. On V2 validation
that collapses retrieval (the judged negatives were picked for high scores), so --all-rows keeps
every row and labels unjudged messages 0, as reference-only training does; judged_labels keeps -1
for them. Every training question must be judged; --allow-partial keeps the judged questions only,
for ranker55.py smoke fits (the tuner requires every active training question). No model is trained here.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

import ranker55
from feature_contract import validate_recipe
from label_reference_messages import validate_judgment
from message_dataset import write_json
from raw_datasets import ROOT, sha256
from tune_ranker55 import Bundle

REQUIRED_ROW_ARRAYS = ('X', 'message_ids', 'reference_labels')
OPTIONAL_ROW_ARRAYS = ('message_indices', 'content_token_lengths')
QUESTION_ARRAYS = ('question_ids', 'persona_ids')
GRADES = (-1, 0, 1, 2)


def judged_rows(labels, bundle, all_rows=False):
    """Map each judged question to its kept bundle rows and grades (-1 unjudged), checking row identities."""
    qids = bundle.read('question_ids').tolist()
    position = {qid: i for i, qid in enumerate(qids)}
    offsets = np.asarray(bundle.read('offsets'))
    mids, reference = bundle.read('message_ids'), bundle.read('reference_labels')
    judged = {}
    with labels.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get('example_id')
            if qid not in position or qid in judged:
                raise ValueError(f'{labels.name} line {number}: unknown or duplicate training question {qid!r}')
            start, end = map(int, offsets[position[qid]:position[qid] + 2])
            memories = row.get('memories') or []
            if [m.get('message_id') for m in memories] != mids[start:end].tolist():
                raise ValueError(f'{qid}: judged memories differ from the feature rows or their order')
            if any(type(m.get('label')) is not int or m['label'] not in GRADES for m in memories):
                raise ValueError(f'{qid}: memory labels must be -1, 0, 1 or 2')
            grades = np.array([m['label'] for m in memories], dtype=np.int8)
            if not np.array_equal(grades == 2, np.asarray(reference[start:end]) == 2):
                raise ValueError(f'{qid}: grade 2 must mark exactly the original references')
            by_id = dict(zip(mids[start:end].tolist(), grades.tolist()))
            candidates = row.get('candidates') or []
            for entry in candidates:
                grade = validate_judgment(entry.get('judge') or {})
                if entry.get('label') != grade or by_id.get(entry.get('message_id')) != grade:
                    raise ValueError(f'{qid}: candidate grade disagrees with its Qwen scores')
            if len({e['message_id'] for e in candidates}) != len(candidates) or int(np.isin(grades, (0, 1)).sum()) != len(candidates):
                raise ValueError(f'{qid}: grades 0/1 must come from exactly the Qwen-judged candidates')
            keep = np.arange(len(grades)) if all_rows else np.flatnonzero(grades >= 0)
            judged[qid] = (start + keep, grades[keep])
    return judged


def build(*, features=None, labels=None, output_root=None, allow_partial=False, all_rows=False, root=ROOT):
    root = Path(root).resolve()
    features = Path(features or root / 'data/ranker55/v2/train/features.npz').resolve()
    labels = Path(labels or root / 'data/qwen_labels/v2/train.jsonl').resolve()
    output = Path(output_root or root / 'data/ranker55/v2' / ('train_graded_all' if all_rows else 'train_graded')).resolve()
    if not output.is_relative_to(root / 'data/ranker55') or output == features.parent:
        raise ValueError('Graded bundles must stay under data/ranker55/, separate from the source bundle')
    report_path = labels.parent / 'report.json'
    judge_report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else {}
    with Bundle(features) as bundle:
        metadata = bundle.metadata
        if (metadata.get('dataset'), metadata.get('split'), metadata.get('memory_unit')) != ('v2', 'train', 'single_message'):
            raise ValueError('Graded training needs the V2 train single-message feature bundle')
        validate_recipe(metadata.get('feature_recipe'))
        # The bundle hashes every split's histories; the judge hashes the training sources only.
        bundle_sources = metadata.get('source_sha256', {})
        if any(bundle_sources.get(k) != v for k, v in judge_report.get('source_sha256', {}).items()):
            raise ValueError('Qwen judgments and features were built from different raw sources/splits')
        judged = judged_rows(labels, bundle, all_rows)
        qids = bundle.read('question_ids').tolist()
        kept = [i for i, qid in enumerate(qids) if qid in judged]
        if len(kept) < len(qids) and not allow_partial:
            raise ValueError(f'{len(qids) - len(kept):,} of {len(qids):,} training questions are not judged yet; '
                             'finish Qwen labeling or pass --allow-partial')
        if not kept:
            raise ValueError('No judged training questions')
        rows = np.concatenate([judged[qids[i]][0] for i in kept])
        judged_labels = np.concatenate([judged[qids[i]][1] for i in kept])
        y = np.maximum(judged_labels, 0)
        offsets = np.concatenate([[0], np.cumsum([len(judged[qids[i]][1]) for i in kept])]).astype(np.int64)
        present = set(bundle.archive.namelist())
        arrays = {name: np.asarray(bundle.read(name)[rows]) for name in REQUIRED_ROW_ARRAYS + tuple(
            n for n in OPTIONAL_ROW_ARRAYS if n + '.npy' in present)}
        # Copies, not memory-map views: the bundle closes before the output is written.
        arrays.update({name: np.asarray(bundle.read(name))[kept] for name in QUESTION_ARRAYS})
        arrays['feature_names'] = np.array(bundle.read('feature_names'))
    arrays['X'], y, offsets = ranker55.validate_training_data(arrays['X'], y, offsets, arrays['feature_names'])
    grade_counts = {str(g): int(np.count_nonzero(y == g)) for g in (0, 1, 2)}
    provenance = {'source_features': str(features), 'source_features_sha256': sha256(features),
                  'qwen_labels': str(labels), 'qwen_labels_sha256': sha256(labels),
                  'qwen_report_status': judge_report.get('status'), 'judged_questions': len(kept),
                  'training_questions': len(qids), 'partial': len(kept) < len(qids),
                  'row_policy': ('all rows; unjudged messages labeled 0 as in reference-only training' if all_rows
                                 else 'judged rows only; unjudged messages dropped, never zero-filled; offsets rebuilt'),
                  'unjudged_rows': int(np.count_nonzero(judged_labels < 0)),
                  'grade_counts': grade_counts, 'builder_sha256': sha256(Path(__file__))}
    metadata.update(questions=len(kept), row_count=len(y), graded_labels=provenance,
                    y_policy='2 original reference; 1 Qwen-judged useful candidate; 0 Qwen-judged not useful candidate'
                             + ('; 0 unjudged message' if all_rows else ''),
                    judged_labels_policy=('-1 unjudged (y 0); otherwise identical to y' if all_rows
                                          else 'identical to y; every kept row is judged'))
    arrays.update(y=y, judged_labels=judged_labels, offsets=offsets,
                  metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
    output.mkdir(parents=True, exist_ok=True)
    target, temporary = output / 'features.npz', output / 'features.npz.tmp'
    try:
        # Uncompressed so tune_ranker55 can memory-map the arrays.
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    statistics = features.parent / 'statistics.json'
    if statistics.is_file():
        shutil.copyfile(statistics, output / 'statistics.json')
    report = {**provenance, 'output': str(target), 'output_sha256': sha256(target),
              'questions': len(kept), 'rows': len(y), 'rows_per_question': {
                  'min': int(np.diff(offsets).min()), 'max': int(np.diff(offsets).max())},
              'tuner_ready': len(kept) == len(qids)}
    write_json(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', type=Path, default=None,
                        help='Source training bundle (default: data/ranker55/v2/train/features.npz)')
    parser.add_argument('--labels', type=Path, default=None,
                        help='Qwen judgments (default: data/qwen_labels/v2/train.jsonl)')
    parser.add_argument('--output-root', type=Path, default=None,
                        help='Output folder under data/ranker55/ (default: data/ranker55/v2/train_graded, '
                             'or train_graded_all with --all-rows)')
    parser.add_argument('--all-rows', action='store_true',
                        help='Keep every row and label unjudged messages 0 instead of dropping them')
    parser.add_argument('--allow-partial', action='store_true',
                        help='Keep only judged questions; usable by ranker55.py but not tune_ranker55.py')
    args = parser.parse_args(argv)
    try:
        report = build(features=args.features, labels=args.labels, output_root=args.output_root,
                       allow_partial=args.allow_partial, all_rows=args.all_rows)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
