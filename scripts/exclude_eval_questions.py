"""Remove one question type from the V2 val/test splits and filter matching feature bundles.

python scripts/exclude_eval_questions.py            # dry run: report what would change
python scripts/exclude_eval_questions.py --apply

sensitive_info reference snippets are absent from every linked 32K history (an upstream
dataset issue; the training split already excludes the type, see training_exclusions.json),
so no retriever can find them. --apply:
  * rewrites data/splits/v2/{val,test}_ids.json without those questions (order kept),
  * updates the counts and ID-file hashes in data/splits/v2/persona_split.json,
  * records the removal in data/splits/v2/evaluation_exclusions.json,
  * filters each feature bundle under --bundle-root (features.npz, bm25/cosine score rows
    and their reports). A question's features depend only on its own history, so retained
    rows are copied unchanged and verified equal; nothing is re-embedded. The previous
    bundle stays beside it as features_before_exclusion.npz and removed score rows as
    scores/{split}_excluded.jsonl.
Saved Qwen runs, the training split and embedding caches are not touched.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from label_reference_messages import load_references
from raw_datasets import ROOT, load_dataset, sha256

SPLITS = ('val', 'test')
ROW_ARRAYS = ('X', 'y', 'reference_labels', 'judged_labels', 'message_ids', 'message_indices', 'content_token_lengths')
QUESTION_ARRAYS = ('question_ids', 'persona_ids')
OTHER_ARRAYS = ('offsets', 'feature_names', 'metadata_json')


def write_lf(path, text):
    # Hashes of these files are recorded in bundles and manifests: always LF bytes.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(text.encode('utf-8'))
    temporary.replace(path)


def find_excluded(root, question_type):
    """Per split: every active question, the excluded IDs and how many have a findable reference."""
    found = {}
    for split in SPLITS:
        questions, contexts, _, sources = load_dataset('v2', [split], root=root, include_answers=False)
        references = load_references(questions, sources, root)
        excluded, findable = [], 0
        for q in questions:
            if q['question_type'] != question_type:
                continue
            visible = {(m.get('role'), m.get('content')) for m in contexts[q['history_id']]}
            findable += any(key in visible for key in references[q['example_id']] if key[1].strip())
            excluded.append(q['example_id'])
        found[split] = {'questions': questions, 'excluded': excluded, 'findable': findable}
    return found


def filter_bundle(directory, split, excluded, new_hashes, note, apply):
    """Drop the excluded questions' rows from one bundle directory; return a summary or None."""
    path = directory / 'features.npz'
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    unknown = set(arrays) - set(ROW_ARRAYS) - set(QUESTION_ARRAYS) - set(OTHER_ARRAYS)
    if unknown:
        raise ValueError(f'{path}: unexpected arrays {sorted(unknown)}')
    metadata = json.loads(str(arrays['metadata_json'].item()))
    if (metadata.get('dataset'), metadata.get('split')) != ('v2', split):
        raise ValueError(f'{path}: not a v2/{split} bundle')
    qids, offsets = arrays['question_ids'].tolist(), arrays['offsets']
    keep = np.array([qid not in excluded for qid in qids])
    summary = {'bundle': path.relative_to(ROOT).as_posix(), 'questions_before': len(qids), 'questions_after': int(keep.sum()),
               'rows_before': int(offsets[-1]), 'previous_features_sha256': sha256(path)}
    if keep.all():
        return {**summary, 'rows_after': int(offsets[-1]), 'changed': False}
    sizes = np.diff(offsets)
    rows = np.repeat(keep, sizes)
    summary.update(rows_after=int(rows.sum()), changed=True)
    if not apply:
        return summary
    filtered = {}
    for name, value in arrays.items():  # original array order is kept in the archive
        if name in ROW_ARRAYS:
            filtered[name] = value[rows]
        elif name in QUESTION_ARRAYS:
            filtered[name] = value[keep]
        elif name == 'offsets':
            filtered[name] = np.concatenate([[0], np.cumsum(sizes[keep])]).astype(offsets.dtype)
        elif name == 'feature_names':
            filtered[name] = value
    sources = dict(metadata['source_sha256'])
    sources.update({name: digest for name, digest in new_hashes.items() if name in sources})
    metadata.update(questions=int(keep.sum()), row_count=int(rows.sum()), source_sha256=sources,
                    question_filter={**note, 'removed_questions': int((~keep).sum()), 'removed_rows': int((~rows).sum()),
                                     'filtered_from_features_sha256': summary['previous_features_sha256'],
                                     'policy': 'rows of excluded questions dropped; every retained row copied unchanged; '
                                               'build-time counts in report.json describe the unfiltered build'})
    filtered['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False))
    temporary = directory / 'features.filtered.tmp.npz'
    with temporary.open('wb') as stream:
        np.savez(stream, **filtered)
    # Retained rows must equal the original rows, question by question.
    with np.load(temporary, allow_pickle=False) as check:
        new_offsets, new_qids = check['offsets'], check['question_ids'].tolist()
        position = {qid: i for i, qid in enumerate(qids)}
        if new_qids != [qid for qid in qids if qid not in excluded] or new_offsets[-1] != rows.sum():
            raise ValueError(f'{path}: filtered question order or row count is wrong')
        for name in ROW_ARRAYS:
            if name not in arrays:
                continue
            new = check[name]
            for i, qid in enumerate(new_qids):
                j = position[qid]
                if not np.array_equal(new[new_offsets[i]:new_offsets[i + 1]], arrays[name][offsets[j]:offsets[j + 1]]):
                    raise ValueError(f'{path}: {name} rows of {qid} changed during filtering')
    backup = directory / 'features_before_exclusion.npz'
    if backup.exists():
        raise ValueError(f'{backup} already exists; refusing to overwrite an earlier backup')
    path.replace(backup)
    temporary.replace(path)
    # Score rows (one JSON line per question) and their reports.
    files = {'features.npz': sha256(path)}
    for name, count_key in (('bm25', 'saved_bm25_scores'), ('cosine', 'saved_cosine_scores')):
        scores = directory / name / 'scores' / f'{split}.jsonl'
        if not scores.is_file():
            continue
        kept, removed = [], []
        with scores.open('rb') as stream:
            for line in stream:
                (removed if json.loads(line)['example_id'] in excluded else kept).append(line)
        (scores.parent / f'{split}_excluded.jsonl').write_bytes(b''.join(removed))
        temporary = scores.with_suffix('.jsonl.tmp')
        temporary.write_bytes(b''.join(kept))
        temporary.replace(scores)
        files[f'{name}/scores/{split}.jsonl'] = sha256(scores)
        child = directory / name / 'report.json'
        if child.is_file():
            report = json.loads(child.read_text(encoding='utf-8'))
            report['source_sha256'] = sources
            report['splits'][split] = {'questions': len(kept), count_key: int(rows.sum())}
            report['files_sha256'] = {f'{split}.jsonl': files[f'{name}/scores/{split}.jsonl']}
            report['question_filter'] = metadata['question_filter']
            child.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            files[f'{name}/report.json'] = sha256(child)
    parent = directory / 'report.json'
    if parent.is_file():
        report = json.loads(parent.read_text(encoding='utf-8'))
        report.update(questions=metadata['questions'], row_count=metadata['row_count'], source_sha256=sources,
                      question_filter=metadata['question_filter'])
        report['files_sha256'] = {**report.get('files_sha256', {}), **files}
        parent.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {**summary, 'features_sha256': files['features.npz']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question-type', default='sensitive_info')
    parser.add_argument('--bundle-root', type=Path, nargs='*', default=[ROOT / 'data/ranker55/subtlex'],
                        help='roots holding v2/{val,test}/features.npz to filter (default: data/ranker55/subtlex)')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    split_dir = ROOT / 'data/splits/v2'
    record_path = split_dir / 'evaluation_exclusions.json'
    if record_path.exists():
        raise SystemExit(f'{record_path} exists: an evaluation exclusion was already applied')
    found = find_excluded(ROOT, args.question_type)
    reason = (f'{args.question_type}: annotated reference snippets are absent from the linked raw 32K histories '
              '(upstream dataset issue), so the questions cannot test memory retrieval. '
              'The training split already excludes this type (training_exclusions.json).')
    manifest_path = split_dir / 'persona_split.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    record = {'dataset': 'v2', 'splits': {}, 'question_type': args.question_type, 'reason': reason,
              'applied_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
              'not_changed': ['training split', 'raw sources', 'saved Qwen runs (they still contain these questions)',
                              'embedding caches and bundles outside the listed bundle roots',
                              'persona_split.json source_sha256.questions (its source file is not in this checkout)']}
    new_ids, new_hashes = {}, {}
    for split in SPLITS:
        info = found[split]
        excluded = set(info['excluded'])
        if not excluded:
            raise SystemExit(f'No {args.question_type} questions in {split}; nothing to remove')
        ids_path = split_dir / f'{split}_ids.json'
        old = json.loads(ids_path.read_text(encoding='utf-8'))
        if set(old) != {q['example_id'] for q in info['questions']} or not excluded <= set(old):
            raise ValueError(f'{ids_path}: IDs differ from the loaded {split} questions')
        new_ids[split] = [example_id for example_id in old if example_id not in excluded]
        kept = [q for q in info['questions'] if q['example_id'] not in excluded]
        entry = manifest['splits'][split]
        if entry['questions'] != len(old) or manifest['source_sha256'][f'{split}_ids'] != sha256(ids_path):
            raise ValueError(f'persona_split.json does not describe the current {split} IDs')
        per_persona = Counter(q['persona_id'] for q in kept)
        record['splits'][split] = {
            'questions_before': len(old), 'removed_questions': len(excluded), 'questions_after': len(new_ids[split]),
            'removed_with_a_reference_present_in_history': info['findable'],
            'personas_before': len(entry['personas']), 'personas_after': len(per_persona),
            'previous_ids_sha256': sha256(ids_path), 'excluded_example_ids': sorted(excluded)}
        entry.update(questions=len(kept), personas=[p for p in entry['personas'] if per_persona[p]],
                     questions_per_persona={p: per_persona[p] for p in entry['questions_per_persona'] if per_persona[p]},
                     preference_types=dict(sorted(Counter(q['question_type'] for q in kept).items())),
                     histories=len({q['history_id'] for q in kept}))
        r = record['splits'][split]
        print(f"{split}: {r['questions_before']} -> {r['questions_after']} questions ({r['removed_questions']} {args.question_type} "
              f"removed; {r['removed_with_a_reference_present_in_history']} of them had a reference present in the history); "
              f"personas {r['personas_before']} -> {r['personas_after']}")
    if not args.apply:
        for root in args.bundle_root:
            for split in SPLITS:
                summary = filter_bundle(Path(root) / 'v2' / split, split, set(found[split]['excluded']), {}, {}, False)
                if summary:
                    print(f"  would filter {summary['bundle']}: {summary['questions_before']} -> {summary['questions_after']} "
                          f"questions, {summary['rows_before']} -> {summary['rows_after']} rows")
        print('Dry run: nothing written. Pass --apply to make the change.')
        return
    # Hash the new ID files before writing anything, so the bundles (the step most likely
    # to fail) are filtered and verified first and the split files change last.
    texts = {split: json.dumps(new_ids[split], indent=2) + '\n' for split in SPLITS}
    for split in SPLITS:
        digest = hashlib.sha256(texts[split].encode('utf-8')).hexdigest()
        record['splits'][split]['ids_sha256'] = manifest['source_sha256'][f'{split}_ids'] = digest
        new_hashes[f'data/splits/v2/{split}_ids.json'] = digest
    note = {'question_type': args.question_type, 'reason': reason, 'record': 'data/splits/v2/evaluation_exclusions.json'}
    record['filtered_bundles'] = []
    for root in args.bundle_root:
        for split in SPLITS:
            summary = filter_bundle(Path(root) / 'v2' / split, split, set(found[split]['excluded']), new_hashes, note, True)
            if summary:
                record['filtered_bundles'].append(summary)
                print(f"  filtered {summary['bundle']}: {summary['questions_before']} -> {summary['questions_after']} questions, "
                      f"{summary['rows_before']} -> {summary['rows_after']} rows (retained rows verified unchanged)")
    for split in SPLITS:
        write_lf(split_dir / f'{split}_ids.json', texts[split])
    manifest['evaluation_exclusions'] = {
        'path': 'data/splits/v2/evaluation_exclusions.json', 'question_type': args.question_type, 'reason': reason,
        'removed_questions': {split: record['splits'][split]['removed_questions'] for split in SPLITS}}
    write_lf(manifest_path, json.dumps(manifest, indent=2) + '\n')
    write_lf(record_path, json.dumps(record, indent=2) + '\n')
    print(f'Applied. Record: {record_path.relative_to(ROOT).as_posix()}')


if __name__ == '__main__':
    main()
