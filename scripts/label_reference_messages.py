"""Assign exact-reference 0/2 labels to complete V2 question-visible messages.

python scripts/label_reference_messages.py --splits train
Reads active split IDs and original raw messages. Writes data/reference_labels/v2/.
No models or retrieval scores are needed. Unmatched references raise an error.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from message_dataset import SPLIT_ROOT, load_message_dataset, write_json
from raw_datasets import ROOT, literal, sha256, split_ids, v2_history_path

LABEL_ROOT = ROOT / 'data/reference_labels'
SCHEMA_VERSION = 3
REFERENCE_LABEL = 2


def reference_keys(snippet, example_id):
    """Keep the original role and content; never normalize or use substring matches."""
    if not isinstance(snippet, list) or not snippet:
        raise ValueError(f'{example_id}: reference snippet must be a nonempty message list')
    keys = []
    for message in snippet:
        if (not isinstance(message, dict) or message.get('role') not in ('user', 'assistant')
                or not isinstance(message.get('content'), str)):
            raise ValueError(f'{example_id}: invalid reference message')
        keys.append((message['role'], message['content']))
    if not any(content.strip() for _, content in keys):
        raise ValueError(f'{example_id}: reference snippet contains no nonempty text')
    return keys


def load_references(questions, sources, root):
    """Resolve IDs by their original CSV prefix, regardless of current split assignment."""
    selected = {q['example_id']: q for q in questions}
    references, checked_histories, empty_histories = {}, set(), {}
    base = root / 'data/raw/personamem_v2'
    for original_split in ('train', 'val', 'benchmark'):
        path = base / f'benchmark/text/{original_split}.csv'
        if sources.get(path.relative_to(root).as_posix()) != sha256(path):
            raise ValueError(f'Changed raw source during loading: {path}; rerun with stable input files')
        with path.open(encoding='utf-8', newline='') as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                prefix = 'test' if original_split == 'benchmark' else original_split
                example_id = f'v2:{prefix}:{index:05d}'
                if example_id not in selected:
                    continue
                question = selected[example_id]
                if question['history_id'] != f"v2:32k:persona{row['persona_id']}":
                    raise ValueError(f'{example_id}: loaded history differs from the original CSV')
                history_path = v2_history_path(base, row['chat_history_32k_link'])
                if history_path not in checked_histories:
                    if sources.get(history_path.relative_to(root).as_posix()) != sha256(history_path):
                        raise ValueError(f'Changed raw history during loading: {history_path}; rerun with stable input files')
                    checked_histories.add(history_path)
                keys = reference_keys(literal(row['related_conversation_snippet']), example_id)
                empty_keys = {key for key in keys if not key[1].strip()}
                if empty_keys:
                    # Retrieval excludes empty turns. Verify their presence in
                    # the original history rather than inventing positive memories.
                    if history_path not in empty_histories:
                        history = json.loads(history_path.read_text(encoding='utf-8'))['chat_history']
                        empty_histories[history_path] = {
                            (m.get('role'), m['content']) for m in history
                            if m.get('role') in ('user', 'assistant')
                            and isinstance(m.get('content'), str) and not m['content'].strip()}
                    if not empty_keys.issubset(empty_histories[history_path]):
                        raise ValueError(f'{example_id}: empty reference turn is absent from the raw history')
                references[example_id] = keys
    if set(references) != set(selected):
        raise ValueError('Reference CSV rows do not exactly match the selected question IDs')
    return references


def label_question(question, memories, references):
    """Label each message independently; only exact role/content matches are positive."""
    example_id = question['example_id']
    nonempty_references = {key for key in references if key[1].strip()}
    matched, reference_indices, labeled = set(), set(), []
    for memory_id in question['memory_ids']:
        memory = memories[memory_id]
        message = memory['source_message']
        if memory['role'] != message['role'] or memory['text'] != message['content']:
            raise ValueError(f'{memory_id}: expected one complete original message')
        key = (message['role'], message['content'])
        label = REFERENCE_LABEL if key in nonempty_references else 0
        if label:
            matched.add(key)
            reference_indices.add(memory['message_index'])
        labeled.append({'message_id': memory_id, 'message_index': memory['message_index'],
                        'role': memory['role'], 'label': label})
    missing = nonempty_references - matched
    if missing or not nonempty_references:
        raise ValueError(f'{example_id}: {len(missing)} reference message(s) absent from visible memories; '
                         'refusing to emit incomplete-reference labels')
    return {'schema_version': SCHEMA_VERSION,
            'example_id': example_id, 'dataset': 'v2', 'split': question['split'],
            'history_id': question['history_id'], 'memory_count': len(labeled),
            'reference_source_message_indices': sorted(reference_indices), 'memories': labeled}


def run(splits=None, *, root=ROOT, split_root=None, output_root=None):
    root = Path(root).resolve()
    splits = list(dict.fromkeys(splits or ['train']))
    if any(split not in ('train', 'val', 'test') for split in splits):
        raise ValueError('V2 supports train, val and test splits')
    split_root = Path(split_root).resolve() if split_root is not None else root / 'data/splits'
    output_root = Path(output_root) if output_root is not None else root / 'data/reference_labels'
    output = (output_root / 'v2').resolve()
    if not output.is_relative_to((root / 'data/reference_labels').resolve()):
        raise ValueError('Label outputs must stay under data/reference_labels/')
    if output.is_relative_to(split_root) or split_root.is_relative_to(output):
        raise ValueError('Label outputs must be separate from the split ID inputs')
    questions, memories, sources = load_message_dataset('v2', splits, root=root, split_root=split_root)
    owners = split_ids(root, 'v2', splits, split_root=split_root)
    if {q['example_id']: q['split'] for q in questions} != owners:
        raise ValueError('Loaded questions differ from the active split IDs; rerun with stable inputs')
    print(f'Loaded {len(questions):,} V2 questions; checking reference snippets and raw source hashes.', flush=True)
    references = load_references(questions, sources, root)
    print('Reference sources verified; assigning message and memory labels.', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    counts, temporary_files = {}, {}
    try:
        # All requested splits are staged before any completed label file is replaced.
        for split in splits:
            path = output / f'{split}.jsonl'
            temporary = path.with_suffix('.jsonl.tmp')
            temporary_files[path] = temporary
            count = Counter(questions=0, positive_memories=0, negative_memories=0,
                            positive_source_messages=0, negative_source_messages=0,
                            empty_reference_messages=0)
            with temporary.open('w', encoding='utf-8', newline='\n') as stream:
                for question in questions:
                    if question['split'] != split:
                        continue
                    refs = references[question['example_id']]
                    row = label_question(question, memories, refs)
                    count['questions'] += 1
                    count['empty_reference_messages'] += sum(not content.strip() for _, content in refs)
                    for memory in row['memories']:
                        count['positive_memories' if memory['label'] else 'negative_memories'] += 1
                        count['positive_source_messages' if memory['label'] else 'negative_source_messages'] += 1
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            counts[split] = dict(count)
        report = {
            'schema_version': SCHEMA_VERSION, 'memory_unit': 'single_message', 'dataset': 'v2', 'splits': counts,
            'source_message_rule': '2 for exact (role, content) reference matches; 0 otherwise; all matching occurrences are positive',
            'zero_label_meaning': 'annotation non-membership, not judged irrelevance; candidate selection exports these as unjudged null',
            'memory_rule': 'Each memory is one original message; its label is its exact-reference membership',
            'empty_reference_rule': 'Verified against raw histories; empty turns have no retrieval memory and contribute no positive label',
            'unmatched_reference_policy': 'error; incomplete references never produce an all-negative training example',
            'source_sha256': sources, 'labeler_sha256': sha256(Path(__file__)),
            'files_sha256': {path.name: sha256(temporary) for path, temporary in temporary_files.items()}}
        for path, temporary in temporary_files.items():
            temporary.replace(path)
        write_json(output / 'report.json', report)
    finally:
        for temporary in temporary_files.values():
            temporary.unlink(missing_ok=True)
    for split, count in counts.items():
        print(f"v2/{split}: {count['questions']:,} questions; {count['positive_memories']:,} positive and "
              f"{count['negative_memories']:,} negative memories.", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['train'])
    parser.add_argument('--split-root', '--input-root', dest='split_root', type=Path, default=SPLIT_ROOT,
                        help='Split ID directory; original messages are read directly from data/raw/')
    parser.add_argument('--output-root', type=Path, default=LABEL_ROOT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(args.splits, split_root=args.split_root, output_root=args.output_root)


if __name__ == '__main__':
    main()
