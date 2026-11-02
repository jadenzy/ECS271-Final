"""Select the LongBench v2 questions whose context fits PersonaMem's 32k scale in Qwen tokens.

python scripts/longbench_split.py
Reads data/raw/longbench_v2/data.json (THUDM/LongBench-v2) and keeps every question whose raw
context is at most 32,768 tokens by the evaluator's pinned Qwen tokenizer (vocabulary only; no
model weights or GPU). Writes data/splits/longbench/comparison_ids.json and persona_split.json.
A context with more whitespace words than the limit cannot fit and is not tokenized.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import qwen_llamacpp_server as qwen_server
from raw_datasets import ROOT, sha256

LIMIT = 32768


def select(root=ROOT, limit=LIMIT):
    root = Path(root).resolve()
    source = root / 'data/raw/longbench_v2/data.json'
    rows = json.loads(source.read_text(encoding='utf-8'))
    # Every server mode uses the same pinned model and runtime, so any settings give the same vocabulary.
    tokenizer = qwen_server.LocalQwenTokenizer(qwen_server.server_settings('full_history', dataset='v2', root=root),
                                               root=root)
    try:
        kept = [row for row in rows if len(row['context'].split()) <= limit
                and len(tokenizer.tokenize(row['context'], add_special=False, parse_special=False)) <= limit]
    finally:
        tokenizer.close()
    ids = ['longbench:' + row['_id'] for row in kept]
    output = root / 'data/splits/longbench'
    output.mkdir(parents=True, exist_ok=True)
    ids_path = output / 'comparison_ids.json'
    ids_path.write_text(json.dumps(ids, indent=2) + '\n', encoding='utf-8')
    manifest = {
        'dataset': 'longbench', 'version': '32k', 'subset': 'comparison',
        'policy': ('LongBench v2 is comparison only. Questions whose context is at most '
                   f'{limit:,} Qwen tokens; each question has its own document, so persona_id is the question.'),
        'question_order': 'persona_id, history_id, example_id',
        'selection': {'max_context_tokens': limit, 'source_questions': len(rows),
                      'tokenizer': 'pinned Qwen3.5-4B GGUF vocabulary; raw context, no special tokens'},
        'splits': {'comparison': {'dataset': 'longbench', 'questions': len(ids), 'personas': ids,
                                  'question_types': dict(Counter(row['domain'] for row in kept)),
                                  'sub_domains': dict(Counter(row['sub_domain'] for row in kept)),
                                  'histories': len(ids)}},
        'source_sha256': {'questions': sha256(source), 'comparison_ids': sha256(ids_path)},
    }
    (output / 'persona_split.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-context-tokens', type=int, default=LIMIT)
    args = parser.parse_args(argv)
    manifest = select(limit=args.max_context_tokens)
    split = manifest['splits']['comparison']
    print(json.dumps({'questions': split['questions'], 'question_types': split['question_types']}, indent=2))


if __name__ == '__main__':
    main()
