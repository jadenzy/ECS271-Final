"""Read original benchmark messages and the existing split IDs without writing data."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import random
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {'v2': ('train', 'val', 'test'), 'v1': ('comparison',), 'locomo': ('comparison',)}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def literal(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def split_ids(root, dataset, splits, *, split_root=None):
    split_root = Path(split_root) if split_root is not None else Path(root) / 'data/splits'
    owners = {}
    for split in splits:
        if split not in SPLITS[dataset]:
            raise ValueError(f'{dataset} does not have a {split} split')
        for example_id in json.loads((split_root / dataset / f'{split}_ids.json').read_text(encoding='utf-8')):
            if example_id in owners:
                raise ValueError('Duplicate or overlapping split IDs')
            owners[example_id] = split
    return owners


def answer_options(example_id, correct, incorrect, seed=271):
    answers = [(correct, True), *((text, False) for text in incorrect)]
    random.Random(f'{seed}:{example_id}').shuffle(answers)
    options = [{'label': chr(97 + i), 'text': text} for i, (text, _) in enumerate(answers)]
    return options, next(options[i]['label'] for i, (_, gold) in enumerate(answers) if gold)


def v2_history_path(base, link):
    """Resolve the CSV's 32K history link to its local downloaded file."""
    relative = unquote(urlsplit(link.replace('\\', '/')).path)
    marker = 'data/chat_history_32k/'
    if marker not in relative:
        raise ValueError(f'Invalid chat_history_32k_link: {link!r}')
    suffix = relative.split(marker, 1)[1]
    folder = (base / marker).resolve()
    path = (folder / suffix).resolve()
    if not path.is_relative_to(folder) or path.suffix != '.json':
        raise ValueError(f'Invalid chat_history_32k_link: {link!r}')
    return path


def load_dataset(dataset='v2', splits=None, root=ROOT, *, include_contexts=True,
                 include_answers=True, split_root=None):
    """Return questions, untouched source messages, answer labels and provenance.

    Set include_answers=False to avoid reading/parsing answer fields for retrieval.
    V1 keeps Python slicing
    semantics for the original per-question end index, including negative ends.
    """
    root = Path(root).resolve()
    split_root = Path(split_root).resolve() if split_root is not None else root / 'data/splits'
    splits = list(splits or SPLITS[dataset])
    owners = split_ids(root, dataset, splits, split_root=split_root)
    questions, contexts, targets, source_paths = [], {}, {}, []
    if dataset == 'v2':
        base = root / 'data/raw/personamem_v2'
        history_links = {}
        for original_split in ('train', 'val', 'benchmark'):
            path = base / f'benchmark/text/{original_split}.csv'
            source_paths.append(path)
            with path.open(encoding='utf-8', newline='') as stream:
                for index, row in enumerate(csv.DictReader(stream)):
                    provenance_split = 'test' if original_split == 'benchmark' else original_split
                    example_id = f'v2:{provenance_split}:{index:05d}'
                    if example_id not in owners:
                        continue
                    persona = str(row['persona_id'])
                    cid = f'v2:32k:persona{persona}'
                    history_path = v2_history_path(base, row['chat_history_32k_link'])
                    if cid in history_links and history_links[cid] != history_path:
                        raise ValueError(f'Conflicting 32K history links for {cid}')
                    history_links[cid] = history_path
                    if include_contexts and cid not in contexts:
                        contexts[cid] = json.loads(history_path.read_text(encoding='utf-8'))['chat_history']
                        source_paths.append(history_path)
                    query = literal(row['user_query'])
                    options, gold = (answer_options(example_id, row['correct_answer'], literal(row['incorrect_answers']))
                                     if include_answers else (None, None))
                    questions.append({'example_id': example_id, 'dataset': dataset, 'split': owners[example_id],
                                      'persona_id': f'v2:{persona}', 'history_id': cid, 'version': '32k',
                                      'source_end_index': None, 'question': query['content'], 'options': options,
                                      'question_type': row['pref_type']})
                    if include_answers:
                        targets[example_id] = gold
                    else:
                        questions[-1].pop('options')
    elif dataset == 'v1':
        base = root / 'data/raw/personamem_v1'
        path = base / 'questions_32k.csv'
        source_paths.append(path)
        with path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                example_id = 'v1:comparison:' + row['question_id']
                if example_id not in owners:
                    continue
                cid = row['shared_context_id']
                options, gold = None, None
                if include_answers:
                    choices = literal(row['all_options'])
                    if isinstance(choices, dict):
                        options = [{'label': str(k).lower(), 'text': v} for k, v in choices.items()]
                    else:
                        options = [{'label': chr(97 + i), 'text': re.sub(r'^\([a-j]\)\s*', '', v)} for i, v in enumerate(choices)]
                    correct = row['correct_answer']
                    matches = [o['label'] for o in options if o['text'] == correct]
                    gold = matches[0] if len(matches) == 1 else correct.strip().strip('()').lower()
                    if gold not in {o['label'] for o in options}:
                        raise ValueError('Cannot locate the original V1 answer')
                questions.append({'example_id': example_id, 'dataset': dataset, 'split': owners[example_id],
                                  'persona_id': 'v1:' + row['persona_id'], 'history_id': cid, 'version': '32k',
                                  'source_end_index': int(row['end_index_in_shared_context']),
                                  'question': row['user_question_or_message'], 'options': options,
                                  'question_type': row['question_type']})
                if include_answers:
                    targets[example_id] = gold
                else:
                    questions[-1].pop('options')
        if include_contexts:
            needed = {q['history_id'] for q in questions}
            path = base / 'shared_contexts_32k.jsonl'
            source_paths.append(path)
            for row in read_rows(path):
                contexts.update({key: value for key, value in row.items() if key in needed})
    elif dataset == 'locomo':
        path = root / 'data/raw/locomo/locomo_mc10.json'
        source_paths.append(path)
        for row in read_rows(path):
            example_id = 'locomo:' + row['question_id']
            if example_id not in owners:
                continue
            cid = 'locomo:' + row['question_id'].rsplit('_q', 1)[0] + ':full'
            if include_contexts and cid not in contexts:
                contexts[cid] = [{**message, 'session_id': session_id, 'timestamp': timestamp}
                                 for session_id, timestamp, session in zip(row['haystack_session_ids'], row['haystack_session_datetimes'], row['haystack_sessions'], strict=True)
                                 for message in session]
            options = ([{'label': chr(97 + i), 'text': text} for i, text in enumerate(row['choices'])]
                       if include_answers else None)
            questions.append({'example_id': example_id, 'dataset': dataset, 'split': owners[example_id],
                              'persona_id': cid.removesuffix(':full'), 'history_id': cid, 'version': 'full',
                              'source_end_index': None, 'question': row['question'], 'options': options,
                              'question_type': row['question_type']})
            if include_answers:
                targets[example_id] = options[row['correct_choice_index']]['label']
            else:
                questions[-1].pop('options')
    else:
        raise ValueError('Unknown dataset')
    if {q['example_id'] for q in questions} != set(owners) or len(questions) != len(owners):
        raise ValueError('Original questions do not exactly match the preserved split IDs')
    if include_contexts and set(contexts) != {q['history_id'] for q in questions}:
        raise ValueError('A selected original conversation is missing')
    questions.sort(key=lambda q: (q['persona_id'], q['history_id'], q['example_id']))
    source_paths.extend(split_root / dataset / f'{split}_ids.json' for split in splits)
    provenance = {(p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p)): sha256(p)
                  for p in sorted(set(source_paths))}
    return questions, contexts, targets, provenance


def visible_messages(question, contexts):
    return contexts[question['history_id']][:question['source_end_index']]


def message_id(context_id, index):
    return f'{context_id}:message:{index}'


def message_memories(context_id, messages):
    """Keep each complete original dialogue message as one retrieval memory."""
    return [{'message_id': message_id(context_id, i), 'message_index': i,
             'context_id': context_id, 'role': m['role'], 'text': m['content'],
             'source_message': dict(m)}
            for i, m in enumerate(messages)
            if m.get('role') in ('user', 'assistant')
            and isinstance(m.get('content'), str) and m['content'].strip()]


def memories_for_question(question, contexts):
    return message_memories(question['history_id'], visible_messages(question, contexts))
