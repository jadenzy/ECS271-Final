"""Pilot: does fine-tuning the embedding model on V2 training references improve retrieval?

python scripts/finetune_embedding_pilot.py --train-questions 6000
python scripts/finetune_embedding_pilot.py --train-questions 0 --evaluate-every 0     # baseline check only

Contrastive fine-tuning of EmbeddingGemma 2 with the pipeline's own input format (same
prefixes, masked mean pooling, L2 normalization). Each training question is pulled toward
its annotated reference messages and pushed away from the non-reference messages of the
same history that the unchanged model scores highest (read from the training bundle's
cosine feature), plus every other message in the batch. Messages whose text equals one of
the question's references are never used as negatives.

Measured on V2 validation with whole-message cosine alone (no ranker): the share of
questions with every visible reference in the top 15, with any reference in the top 10,
and reference recall at 15. Validation personas are disjoint from training personas; the
test split is not read. Writes results/embedding_finetune_pilot/{label}/report.json. The
fine-tuned weights are kept only with --save-model.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

import ranker55 as recipe
from message_dataset import load_message_dataset
from raw_datasets import ROOT
from tune_ranker55 import Bundle, validation_targets

MODEL_PATH = ROOT / 'models/embeddinggemma-2'
PREFIXES = recipe.PREFIXES


def selected_indices(qids, *, question_ids=None, pick=None, seed=271):
    index = np.arange(len(qids))
    if question_ids is not None:
        requested = json.loads(Path(question_ids).read_text(encoding='utf-8'))
        if not isinstance(requested, list) or len(set(requested)) != len(requested) or not requested:
            raise ValueError('Question IDs must be a nonempty unique list')
        if not set(requested) <= set(qids):
            raise ValueError('Question IDs outside the requested split')
        wanted = set(requested)
        index = np.array([i for i, qid in enumerate(qids) if qid in wanted], dtype=np.int64)
    if pick is not None and pick < len(index):
        index = np.sort(np.random.default_rng(seed).choice(index, pick, replace=False))
    return index


def load_split(split, *, cosine=False, pick=None, seed=271, question_ids=None):
    """Questions with their visible message texts, reference rows, and optionally base cosine."""
    features = ROOT / f'data/ranker55/v2/{split}/features.npz'
    with Bundle(features) as bundle:
        names = bundle.read('feature_names').tolist()
        targets = validation_targets(bundle, ROOT, split)
        offsets = np.array(targets['offsets'], dtype=np.int64)
        positive = np.array(targets['positive'])
        questions = targets['questions']
        index = selected_indices([q['example_id'] for q in questions], question_ids=question_ids, pick=pick, seed=seed)
        base = {}
        if cosine:
            column = names.index('cosine')
            X = bundle.read('X')
            base = {int(i): np.array(X[int(offsets[i]):int(offsets[i + 1]), column], dtype=np.float32) for i in index}
    _, memories, _ = load_message_dataset('v2', [split], root=ROOT)
    rows = []
    for i in index:
        i = int(i)
        q = questions[i]
        refs = np.flatnonzero(positive[int(offsets[i]):int(offsets[i + 1])])
        if len(refs):
            rows.append({'example_id': q['example_id'], 'persona_id': q['persona_id'], 'question': q['question'],
                         'texts': [memories[m]['text'] for m in q['memory_ids']], 'references': refs.tolist(),
                         'base_cosine': base.get(i)})
    return rows


class Embedder:
    def __init__(self, path=MODEL_PATH):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        config.vision_config = config.audio_config = None
        self.model = AutoModel.from_pretrained(path, config=config, local_files_only=True, trust_remote_code=False,
                                               dtype=torch.float32, attn_implementation='sdpa').to('cuda')

    def embed(self, texts, kind, *, max_tokens=None):
        """Normalized vectors with gradients; callers choose train/eval mode and autocast."""
        torch = self.torch
        batch = self.tokenizer([PREFIXES[kind] + t for t in texts], add_special_tokens=True, padding=True,
                               truncation=max_tokens is not None, max_length=max_tokens, return_tensors='pt').to('cuda')
        states = self.model(**batch).last_hidden_state.float()
        mask = batch['attention_mask'].unsqueeze(-1)
        return torch.nn.functional.normalize((states * mask).sum(dim=1) / mask.sum(dim=1), p=2, dim=1)

    def encode(self, texts, kind, *, batch_tokens=24000):
        """Inference vectors for many texts, longest first so padding stays small. No truncation."""
        torch = self.torch
        lengths = np.array([len(ids) for ids in self.tokenizer([PREFIXES[kind] + t for t in texts], add_special_tokens=True)['input_ids']])
        order = np.argsort(-lengths, kind='stable')
        out = None
        self.model.eval()
        start = 0
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            while start < len(order):
                size = max(1, min(256, batch_tokens // int(lengths[order[start]])))
                chunk = order[start:start + size]
                vectors = self.embed([texts[i] for i in chunk], kind).float().cpu().numpy()
                if out is None:
                    out = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
                out[chunk] = vectors
                start += size
        return out


def evaluate(embedder, rows):
    """Whole-message cosine ranking per question; references are rows['references']."""
    texts = sorted({t for r in rows for t in r['texts']})
    position = {t: i for i, t in enumerate(texts)}
    memory = embedder.encode(texts, 'memory')
    questions = sorted({r['question'] for r in rows})
    qposition = {t: i for i, t in enumerate(questions)}
    query = embedder.encode(questions, 'question')
    every15, any10, any5, recall15, lowest = [], [], [], [], []
    for r in rows:
        scores = memory[[position[t] for t in r['texts']]] @ query[qposition[r['question']]]
        order = np.argsort(-scores, kind='stable')
        rank = np.empty(len(scores), dtype=int)
        rank[order] = np.arange(1, len(scores) + 1)
        ref = rank[r['references']]
        every15.append(ref.max() <= 15); any10.append(ref.min() <= 10); any5.append(ref.min() <= 5)
        recall15.append(np.mean(ref <= 15)); lowest.append(ref.max())
    return {'questions': len(rows), 'every_reference_top15': float(np.mean(every15)), 'any_reference_top10': float(np.mean(any10)),
            'any_reference_top5': float(np.mean(any5)), 'reference_recall_at_15': float(np.mean(recall15)),
            'median_rank_of_lowest_reference': float(np.median(lowest))}


def show(tag, m):
    print(f"{tag}: every reference in top 15 {m['every_reference_top15']:.1%} | any reference in top 10 {m['any_reference_top10']:.1%} "
          f"| top 5 {m['any_reference_top5']:.1%} | recall@15 {m['reference_recall_at_15']:.1%} "
          f"| median rank of lowest reference {m['median_rank_of_lowest_reference']:.0f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-questions', type=int, default=6000, help='random training questions (seed-fixed); 0 = no training')
    parser.add_argument('--question-ids', type=Path, help='Restrict training to these active question IDs before sampling')
    parser.add_argument('--checkpoint', type=Path, help='Save/resume trusted local training state every 125 steps')
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=16, help='questions per step')
    parser.add_argument('--positives', type=int, default=2, help='reference messages per question per step')
    parser.add_argument('--hard-negatives', type=int, default=5, help='per question per step, sampled from the hardest pool')
    parser.add_argument('--random-negatives', type=int, default=1, help='per question per step, sampled from the rest of its history')
    parser.add_argument('--hard-pool', type=int, default=30, help='highest-cosine non-reference messages to sample hard negatives from')
    parser.add_argument('--max-tokens', type=int, default=256, help='training truncation; evaluation never truncates')
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--evaluate-every', type=int, default=125, help='steps between validation checks; 0 = only at the start and end')
    parser.add_argument('--limit-steps', type=int, default=0, help='stop early (smoke test)')
    parser.add_argument('--skip-evaluation', action='store_true', help='smoke test: train only')
    parser.add_argument('--seed', type=int, default=271)
    parser.add_argument('--label', default='pilot')
    parser.add_argument('--save-model', type=Path, default=None, help='folder for the fine-tuned weights (not saved by default)')
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = ROOT / 'results/embedding_finetune_pilot' / args.label
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    val = [] if args.skip_evaluation else load_split('val')
    embedder = Embedder()
    report = {'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, 'model': str(MODEL_PATH),
              'validation': []}
    if val:
        baseline = evaluate(embedder, val)
        report['validation'].append({'step': 0, **baseline})
        show('step 0 (unchanged model)', baseline)

    if args.train_questions:
        train = load_split('train', cosine=True, pick=args.train_questions, seed=args.seed, question_ids=args.question_ids)
        pools = []
        for r in train:
            refs = set(r['references'])
            ref_texts = {r['texts'][j] for j in refs}
            allowed = [j for j in range(len(r['texts'])) if j not in refs and r['texts'][j] not in ref_texts]
            ranked = sorted(allowed, key=lambda j: -float(r['base_cosine'][j]))
            pools.append((ranked[:args.hard_pool], ranked[args.hard_pool:]))
        steps = math.ceil(len(train) * args.epochs / args.batch_size)
        if args.limit_steps:
            steps = min(steps, args.limit_steps)
        print(f'{len(train)} training questions, {steps} steps of {args.batch_size} questions', flush=True)
        model = embedder.model
        try:
            model.gradient_checkpointing_enable()
        except Exception as error:                      # optional memory saving only
            print(f'gradient checkpointing unavailable: {error}', flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        warm = max(1, int(steps * args.warmup))
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
        order = np.concatenate([rng.permutation(len(train)) for _ in range(math.ceil(args.epochs))])
        losses, torch.backends.cuda.matmul.allow_tf32 = [], True
        first_step = 0
        identity = {'config': report['config'], 'question_ids': [r['example_id'] for r in train]}
        if args.checkpoint and args.checkpoint.exists():
            saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            if saved['identity'] != identity:
                raise ValueError('Checkpoint configuration or training questions differ')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            schedule.load_state_dict(saved['scheduler'])
            rng.bit_generator.state = saved['numpy_rng']
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
            first_step, losses = saved['step'], saved['losses']
            print(f'Resuming saved training at step {first_step}/{steps}', flush=True)
            del saved
        for step in range(first_step, steps):
            model.train()
            batch = order[step * args.batch_size:(step + 1) * args.batch_size]
            candidates, owner, is_positive = [], [], []
            for b, i in enumerate(batch):
                r, (hard, rest) = train[i], pools[i]
                chosen = rng.choice(r['references'], min(args.positives, len(r['references'])), replace=False).tolist()
                negatives = (rng.choice(hard, min(args.hard_negatives, len(hard)), replace=False).tolist() if hard else []) + \
                            (rng.choice(rest, min(args.random_negatives, len(rest)), replace=False).tolist() if rest else [])
                for j in chosen + negatives:
                    candidates.append(r['texts'][j]); owner.append(b); is_positive.append(j in chosen)
            reference_texts = [{train[i]['texts'][j] for j in train[i]['references']} for i in batch]
            # allowed[b, c]: candidate c may sit in question b's denominator (its own positives are handled per positive)
            allowed = torch.tensor([[not (c_text in reference_texts[b]) for c_text in candidates] for b in range(len(batch))], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                q = embedder.embed([train[i]['question'] for i in batch], 'question', max_tokens=args.max_tokens)
                c = embedder.embed(candidates, 'memory', max_tokens=args.max_tokens)
            scores = (q @ c.T) / args.temperature
            negatives_only = scores.masked_fill(~allowed, float('-inf'))
            terms = []
            for col, (b, pos) in enumerate(zip(owner, is_positive)):
                if pos:
                    terms.append(torch.logsumexp(torch.cat([scores[b, col:col + 1], negatives_only[b]]), 0) - scores[b, col])
            loss = torch.stack(terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 25 == 0 or step + 1 == steps:
                print(f'step {step + 1}/{steps} loss {np.mean(losses[-25:]):.3f} | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB '
                      f'| {time.time() - started:.0f}s', flush=True)
                report['progress'] = {'step': step + 1, 'steps': steps, 'seconds_this_process': time.time() - started,
                                      'mean_loss': float(np.mean(losses[-25:]))}
                (out / 'progress.json').write_text(json.dumps(report['progress'], indent=2) + '\n', encoding='utf-8')
            if args.checkpoint and ((step + 1) % 125 == 0 or step + 1 == steps):
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.checkpoint.with_suffix('.tmp')
                torch.save(dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                scheduler=schedule.state_dict(), numpy_rng=rng.bit_generator.state,
                                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                                step=step + 1, losses=losses), temporary)
                temporary.replace(args.checkpoint)
            if val and ((args.evaluate_every and (step + 1) % args.evaluate_every == 0) or step + 1 == steps):
                metrics = evaluate(embedder, val)
                report['validation'].append({'step': step + 1, 'training_questions_seen': (step + 1) * args.batch_size,
                                             'mean_loss': float(np.mean(losses[-args.evaluate_every or None:])), **metrics})
                show(f'step {step + 1}', metrics)
        report['training'] = {'questions': len(train), 'steps': steps, 'final_mean_loss': float(np.mean(losses[-25:]))}
        if args.save_model:
            args.save_model.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.save_model)
            embedder.tokenizer.save_pretrained(args.save_model)
    report['seconds'] = time.time() - started
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {(out / "report.json").relative_to(ROOT).as_posix()} after {report["seconds"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
