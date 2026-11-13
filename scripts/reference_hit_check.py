"""Reference Hit@10 / NDCG@10 of saved rankers on a V2 split (CPU only; no Qwen).

python scripts/reference_hit_check.py --split val
python scripts/reference_hit_check.py --split test --tag before_exclusion
Same targets and aggregation as tune_ranker55: the headline is the non-forget
type-macro Hit@10. Test references are read for measurement only; nothing is tuned.
Writes results/reference_hit/v2/{split}/report[_tag].json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from feature_contract import bundle_recipe
from raw_datasets import ROOT
from select_label_candidates import SavedModels
from tune_ranker55 import Bundle, aggregate, score_questions, validation_targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--features', type=Path, default=None, help='default: data/ranker55/subtlex/v2/{split}/features.npz')
    parser.add_argument('--models', nargs='+', default=['references_only=models/subtlex_mix/alpha_0.5/ranker55.txt',
                                                        'qwen_graded=models/qwen_graded_all/ranker55.txt'])
    parser.add_argument('--tag', default='', help='suffix for the report file name')
    args = parser.parse_args(argv)
    features = args.features or ROOT / f'data/ranker55/subtlex/v2/{args.split}/features.npz'
    with Bundle(features) as bundle:
        names = bundle.read('feature_names').tolist()
        X = np.array(bundle.read('X'), dtype=np.float32)  # copy: the memmap closes with the bundle
        targets = validation_targets(bundle, ROOT, args.split)
        targets['offsets'] = offsets = np.array(targets['offsets'], dtype=np.int64)
    questions = targets['questions']
    cosine = X[:, names.index('cosine')].astype(float)
    recipe = bundle_recipe(features)
    types = sorted({q['question_type'] for q in questions})
    print(f"V2 {args.split}: {len(questions)} questions, {len(X)} rows, types: {', '.join(types)}")
    report = {'split': args.split, 'features': str(features), 'questions': len(questions), 'rows': int(len(X)),
              'question_types': {t: sum(q['question_type'] == t for q in questions) for t in types}, 'models': {}}
    for spec in args.models:
        label, path = spec.split('=', 1)
        model = SavedModels(ROOT / path, None, names, feature_recipe=recipe)
        scores = np.concatenate([model.predict(q, X[int(offsets[i]):int(offsets[i + 1])])[0]
                                 for i, q in enumerate(questions)])
        stats = aggregate(score_questions(scores, cosine, targets))
        report['models'][label] = {'model': path, **stats}
        macro, forget = stats['nonforget_type_macro'], stats['by_type'].get('ask_to_forget', {})
        print(f"\n{label}: non-forget type-macro Hit@10 {macro['hit_at_10']:.2%}  NDCG@10 {macro['ndcg_at_10']:.2%}  "
              f"| all questions Hit@10 {stats['all_questions']['hit_at_10']:.2%}  "
              f"| ask_to_forget Hit@10 {forget.get('hit_at_10', float('nan')):.2%}")
        for kind, row in stats['by_type'].items():
            print(f"    {kind:<32} n={row['questions']:<4} Hit@10 {row['hit_at_10']:.2%}  NDCG@10 {row['ndcg_at_10']:.2%}")
    output = ROOT / f"results/reference_hit/v2/{args.split}/report{'_' + args.tag if args.tag else ''}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'\nSaved {output.relative_to(ROOT).as_posix()}')


if __name__ == '__main__':
    main()
