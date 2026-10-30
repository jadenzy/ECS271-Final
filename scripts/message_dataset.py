"""Read single-message retrieval inputs directly from raw histories and split IDs.

No preparation, manifest, targets or intermediate dataset files are required.
"""
import json
from pathlib import Path

from raw_datasets import ROOT, SPLITS, load_dataset, memories_for_question

SPLIT_ROOT = ROOT / 'data/splits'
MEMORY_UNIT = 'single_message'


def load_message_dataset(dataset, splits=None, *, root=ROOT, split_root=None):
    splits = list(dict.fromkeys(splits or SPLITS[dataset]))
    if any(split not in SPLITS[dataset] for split in splits):
        raise ValueError(f'Invalid splits for {dataset}: {splits}')
    questions, contexts, _, sources = load_dataset(
        dataset, splits, root=root, split_root=split_root, include_answers=False)
    visible_ids, memories = {}, {}
    for question in questions:
        history = contexts[question['history_id']]
        end = slice(None, question['source_end_index']).indices(len(history))[1]
        key = (question['history_id'], end)
        if key not in visible_ids:
            rows = memories_for_question(question, contexts)
            visible_ids[key] = [row['message_id'] for row in rows]
            memories.update((row['message_id'], row) for row in rows)
        question['memory_ids'] = visible_ids[key]
    return questions, memories, sources


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)
