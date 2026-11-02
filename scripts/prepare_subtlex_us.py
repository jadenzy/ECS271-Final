"""Normalize SUBTLEX-US Word/FREQcount into a reusable token-frequency prior.

python scripts/prepare_subtlex_us.py --input PATH/TO/SUBTLEXusrawdata.dat
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://www.ugent.be/plone_portal/pp/experimentele-psychologie/en/research/documents/subtlexus/overview.htm'


def prepare(source, output, encoding='cp1252'):
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    counts = Counter()
    rows = discarded = split_rows = 0
    input_count = discarded_count = 0
    with source.open(encoding=encoding, newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {'Word', 'FREQcount'} <= set(reader.fieldnames or []):
            raise ValueError('Expected tab-separated Word and FREQcount columns')
        for row in reader:
            if row.get('Word') is None or row.get('FREQcount') is None or None in row:
                raise ValueError(f'Malformed row at line {reader.line_num}')
            frequency = int(row['FREQcount'])
            if frequency <= 0:
                raise ValueError(f'FREQcount must be positive at line {reader.line_num}')
            # Same word boundaries and normalization as ranker55.tokenize.
            tokens = re.findall(r'\w+', unicodedata.normalize('NFKC', row['Word']).casefold())
            rows += 1
            input_count += frequency
            if not tokens:
                discarded += 1
                discarded_count += frequency
            elif len(tokens) > 1:
                split_rows += 1
            for token in tokens:
                counts[token] += frequency
    total = sum(counts.values())
    if not total:
        raise ValueError('No word tokens found')
    result = {
        'schema_version': 1,
        'recipe': 'subtlex_us_nfkc_casefold_word_token_frequency_v1',
        'source': {
            'filename': source.name,
            'sha256': hashlib.sha256(raw).hexdigest(),
            'encoding': encoding,
            'url': SOURCE_URL,
            'citation': 'Brysbaert, M., & New, B. (2009). Behavior Research Methods, 41, 977-990. doi:10.3758/BRM.41.4.977',
        },
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'normalization': {
            'unicode': 'NFKC then casefold', 'token_pattern': r'\w+',
            'split_entry_policy': 'Each token occurrence receives the full entry FREQcount',
            'duplicate_policy': 'Sum all normalized token occurrences across source rows',
            'probability_denominator': 'Sum of transformed token counts; not original corpus size',
            'oov_policy': 'Absent words have probability zero; smoothing belongs to the downstream mixture',
        },
        'summary': {
            'source_rows': rows, 'source_frequency_total': input_count,
            'discarded_rows': discarded, 'discarded_frequency_total': discarded_count,
            'multi_token_rows': split_rows, 'vocabulary_size': len(counts),
            'total_token_count': total,
        },
        'words': {word: {'count': count, 'probability': count / total}
                  for word, count in sorted(counts.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'data/external/subtlex_us/word_frequencies.json')
    parser.add_argument('--encoding', default='cp1252', help='Source text encoding (default: cp1252)')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must be different files')
    result = prepare(args.input, args.output, args.encoding)
    print(json.dumps({'output': str(args.output.resolve()), **result['summary']}, indent=2))


if __name__ == '__main__':
    main()
