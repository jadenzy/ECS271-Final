"""Evaluate frozen Qwen on PersonaMem V1/V2 or LoCoMo-MC10.

Targets are used only after inference for grading. Original raw messages and preserved split IDs are loaded directly.
With no options, resume the latest unfinished run for the current model under results; otherwise the
default is full-history PersonaMem-v2 test evaluation. V1 and LoCoMo use their full
comparison datasets. V2 also supports no-memory and original-reference-only
(label 2) baselines. Selector mode supports all three datasets using saved NPZ
features and a LightGBM model; no embedding generation or training runs here.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from feature_contract import bundle_recipe
from raw_datasets import sha256 as digest, load_dataset, visible_messages, memories_for_question
import qwen_llamacpp_server as qwen_server
EVALUATION_MODES = ('full_history', 'no_memory', 'reference_only', 'selector')
DEFAULT_SELECTOR_MEMORY_TOKENS = qwen_server.DEFAULT_SELECTOR_MEMORY_TOKENS

def read_rows(path):
    return list(iter_rows(path))

def iter_rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def read_predictions(path, *, recover=True):
    """Recover a torn final append, preserving its bytes before retrying it.

    Completed lines must parse normally. Only an invalid final line without a
    newline can be a partial write; corruption elsewhere is never skipped.
    Discovery uses recover=False to inspect progress without changing saved bytes.
    """
    if not path.exists():
        return []
    rows = []
    with path.open('r+b' if recover else 'rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if line.endswith(b'\n'):
                    raise ValueError(f'Corrupt saved prediction at byte {offset}; restore {path} before resuming') from error
                if not recover:
                    break
                backup = path.with_name(f'{path.name}.incomplete-{time.time_ns()}.bak')
                with backup.open('xb') as saved:
                    saved.write(line)
                stream.seek(offset)
                stream.truncate()
                print(f'Recovered {len(rows)} complete predictions; incomplete final bytes saved to {backup.name}. The unfinished question will be retried.', flush=True)
                break
            if not isinstance(row, dict) or not isinstance(row.get('example_id'), str):
                raise ValueError(f'Invalid saved prediction at byte {offset} in {path}')
            rows.append(row)
    return rows

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def runtime_metadata(dataset='v2', mode='full_history', *, memory_tokens=None):
    """Read native launch metadata from the dedicated server module."""
    return qwen_server.server_settings(mode, dataset=dataset, root=ROOT, memory_tokens=memory_tokens)


def evaluation_config(dataset='v2', mode='full_history', *, root=None):
    """Share the exact prompt settings with offline tokenizer audits."""
    root = Path(root) if root is not None else ROOT
    config = json.loads((root / 'configs/evaluate_qwen.json').read_text(encoding='utf-8'))
    config.update(dataset=dataset, baseline=mode)
    if dataset == 'locomo':
        config.update(answer_labels=list('abcdefghij'), system_prompt='Use the conversation history to answer the question about its speakers. Select exactly one option, including the unanswerable option when appropriate. Reply with exactly one lowercase answer letter: a, b, c, d, e, f, g, h, i, or j.', primary_nll='negative log-probability normalized over the ten permitted labels')
    elif dataset == 'longbench':
        config.update(system_prompt='Use the provided context passages to answer the question. Select exactly one option. Reply with exactly one lowercase answer letter: a, b, c, or d.')
    return config

class NoMemoryInputs:
    """Supply an empty history without opening conversation or memory files."""

    def get(self, question):
        return ([], {'history_messages': 0})

class VisibleHistories:
    """Read complete source messages directly from raw data."""

    def __init__(self, questions, contexts=None):
        datasets = {q['dataset'] for q in questions}
        if len(datasets) != 1 or not datasets <= {'v1', 'v2', 'locomo', 'longbench'}:
            raise ValueError('VisibleHistories requires one dataset')
        self.questions = {q['example_id']: dict(q) for q in questions}
        if contexts is None:
            _, contexts, _, _ = load_dataset(next(iter(datasets)), root=ROOT)
        self.contexts = contexts
        if not {q['history_id'] for q in questions} <= set(contexts):
            raise ValueError('Missing original conversation')

    def get(self, question):
        original = self.questions[question['example_id']]
        if any((original[k] != question[k] for k in ('persona_id', 'dataset', 'split', 'version', 'history_id', 'source_end_index'))):
            raise ValueError('Question/history ownership mismatch')
        messages = visible_messages(question, self.contexts)
        return (messages, {'history_id': question['history_id'], 'source_end_index': question['source_end_index'], 'effective_end_index': len(messages), 'history_messages': len(messages)})

class ReferenceHistories:
    """Oracle: verbatim annotated snippets, even when absent from visible history."""

    def __init__(self, questions, contexts, sources, *, root=ROOT):
        from label_reference_messages import load_references, REFERENCE_LABEL
        if any(q['dataset'] != 'v2' for q in questions):
            raise ValueError('Reference-only evaluation supports V2 only')
        self.visible = VisibleHistories(questions, contexts)
        self.references = load_references(questions, sources, Path(root))
        self.reference_label = REFERENCE_LABEL
        self.provenance = {
            'oracle': True, 'source': 'original related_conversation_snippet field',
            'included_labels': [REFERENCE_LABEL], 'qwen_generated_labels_used': False,
            'order': 'dataset snippet order', 'missing_reference_policy':
                'Include the original annotated message and report its absence from visible history.',
            'reference_loader_sha256': digest(Path(__file__).with_name('label_reference_messages.py')),
            'references_sha256': hashlib.sha256(json.dumps(self.references, sort_keys=True).encode()).hexdigest(),
        }

    def get(self, question):
        visible, source = self.visible.get(question)
        keys = {(m['role'], m.get('content', '')) for m in visible}
        refs = self.references[question['example_id']]
        history = [{'role': role, 'content': text} for role, text in refs]
        missing = sum(key not in keys for key in refs)
        return history, {**source, 'available_history_messages': len(visible),
                         'history_messages': len(history), 'reference_message_count': len(history),
                         'reference_label': self.reference_label,
                         'reference_messages_missing_from_visible_history': missing,
                         'oracle_reference_input': True}


class SelectorHistories:
    """Score the complete visible pool with the saved ranker; never read labels."""

    def __init__(self, questions, contexts, sources, features, model, top_k=None, *, memory_tokens=None, selected_ids=None,
                 score_cutoff=None, floor_tokens=0):
        from select_label_candidates import load_features, SavedModels
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError('top_k must be positive')
        if memory_tokens is not None and (type(memory_tokens) is not int or memory_tokens < 1):
            raise ValueError('memory_tokens must be positive')
        if top_k is not None and memory_tokens is not None:
            raise ValueError('Use either top_k or a memory token budget')
        self.visible = VisibleHistories(questions, contexts)
        self.top_k = top_k
        self.memory_tokens = (memory_tokens or DEFAULT_SELECTOR_MEMORY_TOKENS) if top_k is None else None
        if score_cutoff is not None and (top_k is not None or isinstance(score_cutoff, bool)
                                         or not isinstance(score_cutoff, (int, float)) or not math.isfinite(score_cutoff) or score_cutoff < 0):
            raise ValueError('score_cutoff must be a finite nonnegative distance below the top score, with a token budget')
        if (type(floor_tokens) is not int or floor_tokens < 0 or (floor_tokens and score_cutoff is None)
                or (self.memory_tokens is not None and floor_tokens > self.memory_tokens)):
            raise ValueError('floor_tokens must be between 0 and the memory budget and requires score_cutoff')
        self.score_cutoff, self.floor_tokens = score_cutoff, floor_tokens
        self.pools, self.questions, self.blocks = {}, {}, {}
        # Share pools for identical history boundaries; options/answers are never
        # passed to the feature reader or ranker.
        for q in questions:
            key = (q['history_id'], q['source_end_index'])
            if key not in self.pools:
                self.pools[key] = memories_for_question(q, contexts)
                self.blocks[key] = SelectorMessageBlocks(q['dataset'], self.pools[key])
            self.questions[q['example_id']] = {
                **{k: q[k] for k in ('example_id', 'dataset', 'split', 'persona_id', 'history_id', 'source_end_index', 'question')},
                'memory_ids': [m['message_id'] for m in self.pools[key]],
            }
        dataset, split = questions[0]['dataset'], questions[0]['split']
        for path in (features, model):
            if not Path(path).is_file():
                raise FileNotFoundError(f'Missing selector input: {path}. Build this dataset/split feature bundle and fit the saved ranker first.')
        self.X, names, self.offsets, qids = load_features(
            features, list(self.questions.values()), sources, dataset=dataset, split=split, require_metadata=True)
        self.groups = {qid: i for i, qid in enumerate(qids)}
        required = set(self.questions) if selected_ids is None else set(selected_ids)
        if not required <= self.groups.keys():
            raise ValueError('Selector features do not cover every requested evaluation question')
        self.models = SavedModels(model, None, names, feature_recipe=bundle_recipe(features))
        self.provenance = {
            'model_path': str(Path(model).resolve()), 'model_sha256': self.models.records[0]['sha256'],
            'features_path': str(Path(features).resolve()), 'features_sha256': digest(Path(features)),
            'feature_names': names, 'model_feature_count': self.models.records[0]['feature_count'],
            'tie_break': 'earlier source message index',
            'prompt_order': 'chronological source order', 'pool': 'all visible nonempty user/assistant messages',
            'uses_reference_labels_or_answers': False,
        }
        if self.memory_tokens is None:
            self.provenance['top_k'] = top_k
        else:
            self.provenance.update(memory_tokens=self.memory_tokens,
                packing_policy='greedy rank order; skip messages that do not fit; never truncate',
                token_cost='native Qwen full prompt minus instruction/question/options-only prompt, including history formatting and session headers',
                context_policy='reserve one answer token within the actual server context')
            if score_cutoff is not None:
                self.provenance.update(score_cutoff=float(score_cutoff), floor_tokens=floor_tokens,
                    cutoff_policy='once packed memory reaches floor_tokens, stop at the first message scoring more than score_cutoff below the top score')

    def _rank(self, question):
        _, source = self.visible.get(question)
        started = time.perf_counter()
        q = self.questions[question['example_id']]
        members = self.pools[(q['history_id'], q['source_end_index'])]
        group = self.groups[q['example_id']]
        start, end = int(self.offsets[group]), int(self.offsets[group + 1])
        scores, _ = self.models.predict(q, self.X[start:end])
        ranked = np.argsort(-scores, kind='stable').tolist()
        source['selector_ranking_seconds'] = time.perf_counter() - started
        return members, scores, ranked, source

    def preview(self, question):
        """Validate/rank on CPU; budget packing requires the active Qwen tokenizer."""
        if self.memory_tokens is None:
            return self.get(question)[1]
        members, _, _, source = self._rank(question)
        return {'available_history_messages': source['history_messages'],
                'available_memory_count': len(members), 'memory_token_budget': self.memory_tokens,
                'selector_ranking_seconds': source['selector_ranking_seconds'],
                'token_packing_performed': False, 'token_packing_requires': 'active Qwen native tokenizer'}

    def get(self, question, *, server=None, config=None):
        members, scores, ranked, source = self._rank(question)
        ranks = {index: rank for rank, index in enumerate(ranked, 1)}
        packing = {}
        if self.memory_tokens is not None:
            if server is None or config is None:
                raise ValueError('Token-budget selection requires the active Qwen server and prompt configuration')
            key = (question['history_id'], question['source_end_index'])
            chosen, packing = pack_selector_messages(server, question, config, members, ranked,
                                                    self.memory_tokens, blocks=self.blocks[key], scores=scores,
                                                    score_cutoff=self.score_cutoff, floor_tokens=self.floor_tokens)
        else:
            chosen = ranked[:self.top_k]
        chosen = sorted(chosen, key=lambda i: members[i]['message_index'])
        history = selected_history(members, chosen)
        selections = [{'message_id': members[i]['message_id'], 'message_index': members[i]['message_index'],
                       'rank': ranks[i], 'score': float(scores[i])} for i in chosen]
        return history, {**source, 'available_history_messages': source['history_messages'],
                         'history_messages': len(history), 'selected_memory_count': len(history),
                         'available_memory_count': len(members), 'selected_messages': selections,
                         **packing}


def selected_history(members, chosen):
    return [{**members[i]['source_message'], 'source_message_index': members[i]['message_index']}
            for i in sorted(chosen, key=lambda i: members[i]['message_index'])]


def message_block(message, index, *, reference=False):
    if message['role'] not in ('system', 'user', 'assistant') or not isinstance(message.get('content', ''), str):
        raise ValueError('Expected text system/user/assistant source messages')
    label = 'Reference message' if reference else 'Message'
    source_index = message.get('source_message_index', index)
    return f"[{label} {source_index} | role: {message['role']}]\n{message.get('content', '')}"


class SelectorMessageBlocks:
    """Format immutable source blocks once per conversation boundary."""

    def __init__(self, dataset, members):
        self.dataset = dataset
        self.indices = [m['message_index'] for m in members]
        self.blocks = [message_block(m['source_message'], m['message_index']) for m in members]
        self.sessions = [(m['source_message']['session_id'],
                          f"[Session {m['source_message']['session_id']} | timestamp: {m['source_message']['timestamp']}]")
                         for m in members] if dataset == 'locomo' else None

    def messages(self, chosen, bare_messages):
        if not chosen:
            return bare_messages
        blocks, previous_session = [], None
        for index in sorted(chosen, key=lambda i: self.indices[i]):
            if self.sessions is not None:
                session, header = self.sessions[index]
                if session != previous_session:
                    blocks.append(header)
                    previous_session = session
            blocks.append(self.blocks[index])
        history = 'Selected conversation messages (in chronological source order):\n\n' + '\n\n'.join(blocks)
        return [bare_messages[0], {'role': 'user', 'content': history}, bare_messages[-1]]


def pack_selector_messages(server, question, config, members, ranked, memory_tokens, *, blocks=None,
                           scores=None, score_cutoff=None, floor_tokens=0):
    """Greedily accept whole messages using exact native, templated token costs.

    With score_cutoff, packing stops at the first message scoring more than score_cutoff
    below the top score once accepted memory has reached floor_tokens.
    """
    started = time.perf_counter()
    if config.get('baseline') != 'selector':
        raise ValueError('Selector packing requires selector prompt configuration')
    blocks = blocks if blocks is not None else SelectorMessageBlocks(question['dataset'], members)
    bare_messages = build_messages(question, config, [])
    bare_rendered = server.render_prompt(bare_messages)
    _, bare_tokens = bare_rendered
    context = min(getattr(server, 'actual_context', None) or config['context_tokens'], config['context_tokens'])
    if bare_tokens + 1 > context:
        raise ValueError(f"Question/options prompt for {question['example_id']} needs {bare_tokens + 1} tokens; server context is {context}")
    effective_budget = min(memory_tokens, context - bare_tokens - 1)
    chosen, used, skipped = [], 0, 0
    final_messages, final_rendered = bare_messages, bare_rendered
    lowest = scores[ranked[0]] - score_cutoff if score_cutoff is not None and ranked else None
    stopped = False
    for index in ranked:
        if lowest is not None and used >= floor_tokens and scores[index] < lowest:
            stopped = True
            break
        trial = chosen + [index]
        messages = blocks.messages(trial, bare_messages)
        rendered = server.render_prompt(messages)
        _, tokens = rendered
        cost = tokens - bare_tokens
        if cost < 0:
            raise ValueError('Native tokenizer returned a negative history token cost')
        if cost <= effective_budget:
            chosen.append(index)
            used = cost
            final_messages, final_rendered = messages, rendered
        else:
            skipped += 1
    verify = getattr(server, 'verify_selector_prompt', None)
    if verify is not None:
        verify(final_messages, final_rendered, bare_messages, bare_rendered)
    cutoff = {'score_cutoff_stop': stopped} if score_cutoff is not None else {}
    return chosen, {'memory_token_budget': memory_tokens, 'effective_memory_token_budget': effective_budget,
                    'packed_memory_tokens': used, 'budget_skipped_messages': skipped, **cutoff,
                    'token_packing_performed': True, 'packing_seconds': time.perf_counter() - started}


def build_messages(question, config, history):
    mode = config.get('baseline', 'full_history')
    options = '\n'.join((f"{o['label']}) {o['text']}" for o in question['options']))
    messages = [{'role': 'system', 'content': config['system_prompt']}]
    if mode == 'no_memory':
        if history:
            raise ValueError('No-memory evaluation cannot include source messages or memories')
        messages.append({'role': 'user', 'content': config['user_template'].format(question=question['question'], options=options)})
        return messages
    if history is None:
        raise ValueError('Full-history evaluation requires the visible source history')
    if mode == 'selector' and not history:
        messages.append({'role': 'user', 'content': config['user_template'].format(question=question['question'], options=options)})
        return messages
    blocks = []
    previous_session = None
    for index, message in enumerate(history):
        if question['dataset'] == 'locomo' and message['session_id'] != previous_session:
            blocks.append(f"[Session {message['session_id']} | timestamp: {message['timestamp']}]")
            previous_session = message['session_id']
        blocks.append(message_block(message, index, reference=mode == 'reference_only'))
    heading = {'full_history': 'Conversation history (all visible source messages, in chronological order)',
               'selector': 'Selected conversation messages (in chronological source order)',
               'reference_only': 'Dataset reference messages (in original snippet order)'}[mode]
    messages.append({'role': 'user', 'content': heading + ':\n\n' + '\n\n'.join(blocks)})
    messages.append({'role': 'user', 'content': config['user_template'].format(question=question['question'], options=options)})
    return messages


def score_question(server, question, history, source, config):
    """Prepare dataset input, call the server module, and attach source provenance."""
    messages = build_messages(question, config, history)
    prediction = server.score(
        messages, request_id=question['example_id'], history_messages=len(history),
        selection_seconds=source.get(f"{config['baseline']}_ranking_seconds", 0) + source.get('packing_seconds', 0))
    return {**prediction, **source}


def metrics(rows, config):
    people = sorted({r['persona_id'] for r in rows})
    grouped = {p: [r for r in rows if r['persona_id'] == p] for p in people}
    counts = np.array([len(grouped[p]) for p in people])
    correct = np.array([sum((r['correct'] for r in grouped[p])) for p in people])
    losses = np.array([sum((r['correct_label_nll'] for r in grouped[p])) for p in people])
    rng = np.random.default_rng(config['bootstrap_seed'])
    indices = rng.integers(0, len(people), size=(config['bootstrap_persona_replicates'], len(people)))
    denominators = counts[indices].sum(axis=1)
    ci_acc = np.quantile(correct[indices].sum(axis=1) / denominators, [0.025, 0.975])
    ci_loss = np.quantile(losses[indices].sum(axis=1) / denominators, [0.025, 0.975])
    result = {'questions': len(rows), 'personas': len(people), 'correct': sum((r['correct'] for r in rows)), 'accuracy': statistics.mean((r['correct'] for r in rows)), 'accuracy_95ci_persona_bootstrap': ci_acc.tolist() if len(people) > 1 else None, 'macro_persona_accuracy': float(np.mean(correct / counts)), 'mean_correct_label_nll': statistics.mean((r['correct_label_nll'] for r in rows)), 'nll_95ci_persona_bootstrap': ci_loss.tolist() if len(people) > 1 else None, 'mean_raw_correct_label_nll': statistics.mean((r['raw_correct_label_nll'] for r in rows)), 'mean_prompt_tokens': statistics.mean((r['prompt_tokens'] for r in rows)), 'max_prompt_tokens': max((r['prompt_tokens'] for r in rows)), 'mean_inference_seconds': statistics.mean((r['inference_seconds'] for r in rows)), 'mean_elapsed_seconds': statistics.mean((r['elapsed_seconds'] for r in rows)), 'median_elapsed_seconds': statistics.median((r['elapsed_seconds'] for r in rows)), 'p95_elapsed_seconds': float(np.quantile([r['elapsed_seconds'] for r in rows], 0.95))}
    for resource in ('gpu_memory', 'ram_working_set'):
        means = [r[f'mean_{resource}_mib'] for r in rows if r.get(f'mean_{resource}_mib') is not None]
        peaks = [r[f'peak_{resource}_mib'] for r in rows if r.get(f'peak_{resource}_mib') is not None]
        result[f'mean_{resource}_mib'] = statistics.mean(means) if means else None
        result[f'peak_{resource}_mib'] = max(peaks) if peaks else None
        result[f'questions_with_{resource}'] = len(means)
    if any(('selected_memory_count' in row for row in rows)):
        for field in ('memory_tokens', 'selected_memory_count', 'available_memory_count', 'selector_ranking_seconds', 'packing_seconds', 'selection_seconds'):
            values = [row[field] for row in rows if field in row]
            result[f'mean_{field}'] = statistics.mean(values) if values else None
    return result

def save_summary(out, rows, config, protocol, hardware=None, status='complete', error=None):
    version = config['dataset_version']
    dataset_name = config['dataset_name']
    mode = config.get('baseline', 'full_history')
    no_memory = mode == 'no_memory'
    limitations = ['Question rows are correlated within personas; confidence intervals resample personas.']
    limitations += ['No model training or judge model is involved.']
    if no_memory:
        limitations += ['Qwen receives only the shared evaluation instructions, question, and answer options. No conversation, source persona profile, retrieved memory, or earlier evaluation answer is included.', 'This measures the question/options-only baseline on the same prepared V2 partition.']
    elif mode == 'reference_only':
        limitations += ['Oracle baseline: only original dataset reference conversation snippets (label 2) are supplied in dataset snippet order. Qwen-generated label 1 messages are never loaded.', 'Annotated reference messages absent from visible history are retained verbatim and counted separately; this is not a deployable retrieval result.']
    elif mode == 'selector':
        limitations += ['The saved LightGBM model scores every visible nonempty user/assistant message using cached features. Whole messages are passed in chronological order; ties favor earlier messages. Reference labels and correct answers are not used for selection.', 'Model and feature hashes, feature order, and selection settings are recorded in the protocol. The feature bundle and model must use the same pinned SUBTLEX recipe and alpha.']
        if 'memory_tokens' in protocol['selector']:
            limitations += ['Messages are considered in descending ranker-score order and skipped if they do not fit. The token budget counts the exact native Qwen prompt increment, including history formatting and session headers. Question/options tokens are outside the memory budget; the complete prompt plus one answer token must also fit the server context.']
        if 'score_cutoff' in protocol['selector']:
            limitations += [f"Score cutoff: once {protocol['selector']['floor_tokens']:,} memory tokens are packed, packing stops at the first message scoring more than {protocol['selector']['score_cutoff']:g} below the question's top score."]
    elif config['dataset'] == 'v2':
        limitations += ['Each question follows its entire official 32K source history, quoted as a chronological role-labelled transcript. Source system messages are preserved. CSV persona profiles, preferences, and reference snippets are not added.', 'Use training personas for selector fitting, validation personas for tuning, and test personas for final reporting.']
    elif config['dataset'] == 'v1':
        limitations += ['The entire source history before the question cutoff is used, including all repeated system persona descriptions, quoted as a chronological role-labelled transcript in a preceding user message.', 'Source cutoffs of -1 retain official Python slicing semantics and exclude the last source message.']
    if config['dataset'] == 'locomo':
        limitations += ['LoCoMo-MC10 uses ten answer options. Conversation IDs are the bootstrap unit; the ten shared conversations are correlated groups, not independent questions.', 'Both speakers and session timestamps are retained. Session summaries and answer metadata are excluded from memory inputs. Source question-type labels are preserved.', 'Evaluation preserves designated correct indices and answer text; local distractor edits, when present, are recorded in the choice-correction audit.', 'This is evaluation of the multiple-choice adaptation; its accuracy is not original LoCoMo answer F1.']
    elif config['dataset'] == 'longbench':
        limitations += ['LongBench v2 subset: the questions whose raw context is at most 32,768 Qwen tokens (data/splits/longbench). Each context is cut into passages of whole lines (at most 150 words and 1,200 characters) presented as user messages in the shared transcript format; most contexts are documents, not conversations.', 'Each question has its own context, so the question is the bootstrap unit. Question types are LongBench domains.']
    elif not no_memory:
        limitations += ['32K is measured by the source tokenizer; the actual Qwen context may need to be larger.']
    limitations += [f"The text-only model uses {config.get('model_dtype', 'Q4_K_M')} weights on CUDA without CPU offload. Runtime settings are recorded in the protocol. The configured context cap has not been benchmarked for VRAM fit."]
    summary = {'status': status, 'updated_utc': datetime.now(timezone.utc).isoformat(), 'baseline': f'frozen_qwen_{mode}', 'split': protocol['split'], 'dataset_version': version, 'selected_questions': protocol['questions'], 'overall': metrics(rows, config), 'by_question_type': {}, 'by_persona': {}, 'confidence_interval_method': f"{config['bootstrap_persona_replicates']} bootstrap draws with persona as the resampling unit; percentile 95% interval.", 'nll_definition': config['primary_nll'], 'chance_accuracy_four_options': 0.25, 'correct_label_counts': {}, 'predicted_label_counts': {}, 'limitations': limitations, 'resource_measurement': 'Memory averages are the arithmetic mean of per-question averages. Poll about once per second during scoring; requests with no poll use an immediate post-response snapshot. Missing measurements are null and excluded from averages; coverage counts are reported. GPU readings include other processes; RAM is the Qwen model server working set only. Runtime includes native template/tokenization HTTP requests, and all scoring attempts, including recovery restart overhead but excluding initial model startup and post-response memory sampling.'}
    summary['answer_option_count'] = len(config['answer_labels'])
    summary['memories_used'] = not no_memory
    if mode in ('reference_only', 'selector'):
        summary[mode] = protocol[mode]
    if mode == 'reference_only':
        missing = [r.get('reference_messages_missing_from_visible_history', 0) for r in rows]
        summary['reference_messages_missing_from_visible_history'] = sum(missing)
        summary['questions_with_references_missing_from_visible_history'] = sum(n > 0 for n in missing)
    summary['chance_accuracy'] = 1 / len(config['answer_labels'])
    if len(config['answer_labels']) != 4:
        summary.pop('chance_accuracy_four_options')
    summary['dataset'] = config['dataset']
    summary['subset'] = config.get('subset', 'full')
    if summary['subset'] == 'compact':
        summary['limitations'].append('This is a fixed whole-persona subset evaluation, not the full PersonaMem-v2 benchmark.')
    segments = protocol.get('server_runs', protocol.get('runtime_segments', []))
    if segments:
        summary['runtime_segments'] = []
        for index, segment in enumerate(segments):
            start = segment['first_prediction']
            end = segments[index + 1]['first_prediction'] if index + 1 < len(segments) else len(rows)
            selected = rows[start:end]
            values = metrics(selected, config) if selected else {}
            summary['runtime_segments'].append({'first_prediction': start, 'questions': len(selected), **{key: values.get(key) for key in ('accuracy', 'mean_elapsed_seconds', 'mean_inference_seconds', 'mean_gpu_memory_mib', 'mean_ram_working_set_mib')}})
        summary['limitations'].append('Runtime and memory averages combine evaluator invocations. Per-invocation measurements are in runtime_segments; direct server commands are recorded without interpretation in protocol.server_runs.')
    if error is not None:
        summary['error'] = repr(error)
    for field, destination in [('question_type', 'by_question_type'), ('persona_id', 'by_persona')]:
        for value in sorted({r[field] for r in rows}):
            summary[destination][str(value)] = metrics([r for r in rows if r[field] == value], config)
    for label in config['answer_labels']:
        summary['correct_label_counts'][label] = sum((r['correct_label'] == label for r in rows))
        summary['predicted_label_counts'][label] = sum((r['predicted_label'] == label for r in rows))
    if hardware is None and (out / 'summary.json').exists():
        hardware = json.loads((out / 'summary.json').read_text(encoding='utf-8')).get('hardware')
    if hardware is not None:
        summary['hardware'] = hardware
    write_json(out / 'summary.json', summary)
    overall = summary['overall']
    interval = overall['accuracy_95ci_persona_bootstrap']
    interval_text = f'**{interval[0]:.2%}–{interval[1]:.2%}**' if interval else 'not estimable from a single persona'
    title = 'full visible history'
    if no_memory:
        title = 'no memory (question/options only)'
    elif mode == 'reference_only':
        title = 'original references only (label 2, oracle)'
    elif mode == 'selector':
        selector = protocol['selector']
        limit = (f"{selector['memory_tokens']:,} Qwen memory tokens" if 'memory_tokens' in selector
                 else f"top {selector['top_k']} messages")
        if 'score_cutoff' in selector:
            limit += f", stopping {selector['score_cutoff']:g} below the top score after {selector['floor_tokens']:,} tokens"
        title = f'LightGBM selector ({limit})'

    def memory_text(resource):
        value = overall[f'mean_{resource}_mib']
        return 'unavailable' if value is None else f"**{value:,.1f} MiB** ({overall[f'questions_with_{resource}']}/{len(rows)} questions measured)"
    lines = [f'# Frozen Qwen baseline: {title}', '', f"Status: **{status}**. Scored {len(rows):,}/{protocol['questions']:,} selected {dataset_name} {version.upper()} questions from {overall['personas']} {protocol['split']} personas.", '', f"- Accuracy: **{overall['accuracy']:.2%}** ({overall['correct']}/{len(rows)}).", f'- 95% confidence interval, clustered by persona: {interval_text}.', f"- Mean correct-option NLL (four-label distribution): **{overall['mean_correct_label_nll']:.4f}**.", f"- Average runtime per question: **{overall['mean_elapsed_seconds']:.3f} seconds**.", f"- Average inference request time per question: **{overall['mean_inference_seconds']:.3f} seconds**.", f"- Average GPU memory per question (device-wide): {memory_text('gpu_memory')}.", f"- Average Qwen model server RAM per question (working set): {memory_text('ram_working_set')}.", f"- Allocated context: **{config['context_tokens']:,} tokens**; longest complete prompt: **{overall['max_prompt_tokens']:,} tokens**.", '', summary['resource_measurement'], '', '| Question type | Questions | Accuracy | Mean NLL |', '| --- | ---: | ---: | ---: |']
    if config['dataset'] == 'locomo':
        lines = [line.replace('four-label distribution', 'ten-label distribution').replace('clustered by persona', 'clustered by conversation') for line in lines]
    lines.insert(3, f"Selection: **{summary['subset']}**; questions run consecutively by persona and history. Compact results cover a subset of the benchmark.")
    for kind, value in summary['by_question_type'].items():
        lines.append(f"| {kind} | {value['questions']} | {value['accuracy']:.2%} | {value['mean_correct_label_nll']:.4f} |")
    if 'runtime_segments' in summary:
        lines.extend(['', 'Resource averages by evaluator invocation:', '', '| Saved prediction range | Questions | Mean runtime (s) | Mean GPU (MiB) | Mean RAM (MiB) |', '| --- | ---: | ---: | ---: | ---: |'])
        for segment in summary['runtime_segments']:
            if not segment['questions']:
                continue
            values = ['unavailable' if segment[key] is None else f'{segment[key]:.3f}' for key in ('mean_elapsed_seconds', 'mean_gpu_memory_mib', 'mean_ram_working_set_mib')]
            first, last = (segment['first_prediction'] + 1, segment['first_prediction'] + segment['questions'])
            lines.append(f"| {first}–{last} | {segment['questions']} | " + ' | '.join(values) + ' |')
    input_description = 'a complete chronological transcript of all visible source messages, with role labels and all repeated system persona descriptions, followed by the question and all options'
    if no_memory:
        input_description = 'only the shared evaluation instructions, question, and all options; no conversation or persona profile'
    elif mode == 'reference_only':
        input_description = 'only original dataset reference conversation snippets (label 2), in original snippet order, followed by the question and all options; Qwen-generated label 1 messages are excluded'
    elif mode == 'selector':
        input_description = f"complete LightGBM-selected source messages within {limit}, in chronological order, followed by the question and all options"
    elif config['dataset'] == 'locomo':
        input_description = 'a complete chronological transcript of both speakers with session timestamps, followed by the question and all ten options'
    truncation = 'History is never truncated.'
    if no_memory:
        truncation = 'No conversation history is loaded or passed to Qwen.'
    elif mode in ('reference_only', 'selector'):
        truncation = 'Selected messages are never truncated.'
    if mode == 'reference_only':
        lines.extend(['', f"Reference messages absent from visible histories: {summary['reference_messages_missing_from_visible_history']} across {summary['questions_with_references_missing_from_visible_history']} scored questions. These remain original label 2 annotations; no label 1 messages are added."])
    lines.extend(['', f"The frozen {config.get("model", "Qwen3.5-4B-Q4_K_M")} model receives {input_description}. Thinking and context shifting are disabled, and the complete prompt must fit the allocated context. {truncation} The answer is the argmax of the {len(config['answer_labels'])} answer-label logits; correct dataset labels grade the result after inference. The run performs no model training. Choose training settings before inspecting test results.", '', 'Per-question predictions and resource measurements are in predictions.jsonl. Aggregate metrics and hardware details are in summary.json; fixed settings and source hashes are in protocol.json.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary

def order_for_cache(questions, limit=None):
    """Preserve the limited selection, then keep identical histories consecutive."""
    questions = list(questions)
    if limit and questions:
        groups = defaultdict(list)
        for q in questions:
            groups[q['persona_id']].append(q)
        questions = [groups[person][i] for i in range(max(map(len, groups.values()))) for person in sorted(groups) if i < len(groups[person])][:limit]
    return sorted(questions, key=lambda q: (q['persona_id'], q['history_id'], q['example_id']))

def parse_args(argv=None, *, fixed_mode=None):
    parser = argparse.ArgumentParser(description='Evaluate Qwen on V2 test or full V1/LoCoMo, using full history, no memory, reference-only, or selector inputs.')
    if fixed_mode is None:
        parser.add_argument('--mode', '-m', choices=EVALUATION_MODES, default='full_history')
    else:
        parser.set_defaults(mode=fixed_mode)
    parser.add_argument('--dataset', '-d', choices=['v2'] if fixed_mode == 'no_memory' else ['v2', 'v1', 'locomo', 'longbench'], default='v2')
    parser.add_argument('--selector-model', type=Path, help='Selector mode: saved LightGBM model (default: models/subtlex_mix/alpha_0.5/ranker55.txt)')
    parser.add_argument('--selector-features', type=Path, help='Selector mode: matching dataset/split NPZ (default: data/ranker55/{dataset}/{split}/features.npz)')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--memory-tokens', '--token-budget', type=int,
                           help=f'Selector mode: Qwen memory-token budget including formatting (default: {DEFAULT_SELECTOR_MEMORY_TOKENS})')
    selection.add_argument('--top-k', type=int, help='Selector mode: explicitly use a message count instead of a token budget')
    parser.add_argument('--score-cutoff', type=float,
                        help='Selector token budget: stop at the first message scoring more than this below the top score')
    parser.add_argument('--floor-tokens', type=int,
                        help='With --score-cutoff: always pack at least this many memory tokens first (default: 0)')
    parser.add_argument('--limit', '-l', type=int)
    parser.add_argument('--output', '-o', type=Path)
    parser.add_argument('--dry-run', '-n', action='store_true')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.mode in ('no_memory', 'reference_only') and args.dataset != 'v2':
        parser.error('No-memory and reference-only evaluation support V2 only')
    if args.top_k is not None and args.top_k < 1:
        parser.error('--top-k must be positive')
    if args.memory_tokens is not None and args.memory_tokens < 1:
        parser.error('--memory-tokens must be positive')
    if args.mode != 'selector' and any(value is not None for value in (args.selector_model, args.selector_features, args.top_k, args.memory_tokens, args.score_cutoff, args.floor_tokens)):
        parser.error('Selector model/features, --top-k, --memory-tokens, --score-cutoff and --floor-tokens require --mode selector')
    if args.score_cutoff is not None and (args.top_k is not None or not math.isfinite(args.score_cutoff) or args.score_cutoff < 0):
        parser.error('--score-cutoff must be a nonnegative number and works with the token budget, not --top-k')
    if args.floor_tokens is not None and (args.score_cutoff is None or args.floor_tokens < 0):
        parser.error('--floor-tokens must be nonnegative and requires --score-cutoff')
    args.floor_tokens = args.floor_tokens or 0
    args.split = 'test' if args.dataset == 'v2' else 'comparison'
    if args.mode == 'selector':
        args.selector_model = args.selector_model or ROOT / 'models/subtlex_mix/alpha_0.5/ranker55.txt'
        args.selector_features = args.selector_features or ROOT / f'data/ranker55/{args.dataset}/{args.split}/features.npz'
        if args.top_k is None and args.memory_tokens is None:
            args.memory_tokens = DEFAULT_SELECTOR_MEMORY_TOKENS
        if args.floor_tokens > (args.memory_tokens or 0):
            parser.error('--floor-tokens cannot exceed the memory budget')
    return args

def supported_test_selection(protocol):
    """Do not automatically resume obsolete train/val or compact comparison runs."""
    config = protocol.get('config', {})
    dataset = config.get('dataset')
    subset = config.get('subset', protocol.get('selection', {}).get('subset', 'full'))
    if dataset == 'v2':
        return protocol.get('split') == 'test'
    return dataset in ('v1', 'locomo', 'longbench') and protocol.get('split') == 'comparison' and subset == 'full'

def latest_unfinished_run(baseline=None):
    """Find saved work using prediction records, never stale summary status."""
    candidates = []
    current = json.loads((ROOT / 'configs/evaluate_qwen.json').read_text(encoding='utf-8'))
    for path in (ROOT / 'results').rglob('protocol.json'):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            continue
        protocol = json.loads(path.read_text(encoding='utf-8'))
        config = protocol.get('config', {})
        if any(config.get(key) != current.get(key) for key in ('model', 'model_revision', 'model_dtype', 'backend')):
            continue
        if config.get('input_policy') != 'raw_complete_messages_v1':
            continue
        if config.get('baseline') not in EVALUATION_MODES or config.get('dataset') not in ('v1', 'v2', 'locomo', 'longbench'):
            continue
        if not supported_test_selection(protocol):
            continue
        if baseline is not None and config['baseline'] != baseline:
            continue
        if config['baseline'] in ('no_memory', 'reference_only') and config['dataset'] != 'v2':
            continue
        total = protocol.get('questions')
        if type(total) is not int or total <= 0:
            raise ValueError(f'Invalid question count in {path}')
        saved = path.with_name('predictions.jsonl')
        records = read_predictions(saved, recover=False)
        if len({row['example_id'] for row in records}) != len(records) or len(records) > total:
            raise ValueError(f'Invalid or duplicate saved predictions in {saved}')
        if len(records) < total:
            activity = max((p.stat().st_mtime_ns for p in (path, saved, path.with_name('qwen-server.log')) if p.exists()))
            candidates.append((activity, str(path), path, protocol, len(records)))
    if not candidates:
        return None
    _, _, path, protocol, count = max(candidates)
    return (path, protocol, count)

def validate_resume_model(previous, config, predictions_path):
    if not predictions_path.exists() and not previous:
        return
    before = previous.get('config', {})
    keys = ('model', 'model_revision', 'model_dtype', 'backend')
    if any(before.get(key) != config.get(key) for key in keys):
        raise ValueError('Saved run uses a different model, revision, precision or backend; choose a new output directory')


def resume_arguments(path, protocol, *, include_mode=True):
    """Restore evaluation choices; current files/settings still pass all checks."""
    if not supported_test_selection(protocol):
        raise ValueError('Only V2 test runs or full V1/LoCoMo comparison runs can be resumed')
    config = protocol['config']
    argv = ['--dataset', config['dataset'], '--output', str(path.parent)]
    if include_mode:
        argv += ['--mode', config['baseline']]
    if config['baseline'] == 'selector':
        selector = protocol['selector']
        argv += ['--selector-model', selector['model_path'], '--selector-features', selector['features_path']]
        if 'memory_tokens' in selector:
            argv += ['--memory-tokens', str(selector['memory_tokens'])]
        else:
            argv += ['--top-k', str(selector['top_k'])]
        if 'score_cutoff' in selector:
            argv += ['--score-cutoff', repr(selector['score_cutoff']), '--floor-tokens', str(selector['floor_tokens'])]
    if protocol.get('limit') is not None:
        argv += ['--limit', str(protocol['limit'])]
    return argv

def main(argv=None, *, fixed_mode=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or all((arg in ('--dry-run', '-n') for arg in argv)):
        unfinished = latest_unfinished_run(fixed_mode)
        if unfinished is not None:
            path, protocol, count = unfinished
            print(f"Resuming latest unfinished evaluation: {count}/{protocol['questions']} saved. Output: {path.parent}", flush=True)
            argv = resume_arguments(path, protocol, include_mode=fixed_mode is None) + argv
    return run_evaluation(parse_args(argv, fixed_mode=fixed_mode))

def run_evaluation(args):
    config = evaluation_config(args.dataset, args.mode)
    config['memory_tokens'] = args.memory_tokens if args.mode == 'selector' else None
    manifest = json.loads((ROOT / f'data/splits/{args.dataset}/persona_split.json').read_text(encoding='utf-8'))
    # This metadata describes the preserved IDs; it is not an evaluation switch.
    subset = manifest.get('subset', 'full') if args.dataset == 'v2' else 'full'
    all_questions, raw_contexts, targets, raw_sources = load_dataset(args.dataset, [args.split], root=ROOT, include_contexts=args.mode != 'no_memory')
    # Explicit top-k selection has no token budget, so retain the complete-history
    # capacity for that comparison. Budgeted selection derives its context cap.
    runtime_mode = 'full_history' if args.mode == 'selector' and args.top_k is not None else args.mode
    settings = runtime_metadata(args.dataset, runtime_mode, memory_tokens=args.memory_tokens)
    if 'ModelRevision' in settings:
        config['model_revision'] = settings['ModelRevision']
    no_memory = args.mode == 'no_memory'
    locomo = args.dataset == 'locomo'
    version = 'full' if locomo else '32k'
    dataset_name = {'locomo': 'LoCoMo-MC10', 'longbench': 'LongBench-v2'}.get(args.dataset, 'PersonaMem-' + args.dataset)
    config.update(input_policy='raw_complete_messages_v1', server_settings=settings, model=settings['Model'], baseline=args.mode, dataset=args.dataset, dataset_name=dataset_name, dataset_version=version, subset=subset, context_tokens=settings['ContextTokens'])
    split_dir = ROOT / 'data/splits' / args.dataset
    selection_path = split_dir / f'{args.split}_ids.json'
    selected_ids = json.loads(selection_path.read_text(encoding='utf-8'))
    selected = set(selected_ids)
    questions = [q for q in all_questions if q['example_id'] in selected]
    if len(questions) != len(selected_ids) or len(selected) != len(selected_ids) or any((q['dataset'] != args.dataset or q['split'] != args.split or q['version'] != version for q in questions)):
        raise ValueError('Selected IDs do not match the dataset/split; restore matching prepared questions and split files')
    if any(([o['label'] for o in q['options']] != config['answer_labels'] for q in questions)):
        raise ValueError('Question option labels differ from the configured answer labels')
    questions = order_for_cache(questions, args.limit)
    if not questions:
        raise ValueError('No questions selected')
    selected_personas = {q['persona_id'] for q in questions}
    if no_memory:
        histories = NoMemoryInputs()
    elif args.mode == 'reference_only':
        histories = ReferenceHistories(questions, raw_contexts, raw_sources, root=ROOT)
    elif args.mode == 'selector':
        histories = SelectorHistories(all_questions, raw_contexts, raw_sources,
                                     args.selector_features, args.selector_model, args.top_k,
                                     memory_tokens=args.memory_tokens, score_cutoff=args.score_cutoff,
                                     floor_tokens=args.floor_tokens,
                                     selected_ids=[q['example_id'] for q in questions])
        if args.memory_tokens is not None:
            histories.provenance['configured_context_tokens'] = settings['ContextTokens']
    else:
        histories = VisibleHistories(questions, raw_contexts)
    print(f"{config['dataset_name']} / {version.upper()} / {subset} / {args.mode} / {args.split}: {len(questions)} questions from {len(selected_personas)} persona/conversation groups; grouped by history for consistent ordering.", flush=True)
    print('The server module manages llama-server.exe using configs/qwen_server.json. Skipping prompt preflight.', flush=True)
    if args.dry_run:
        details = {'dry_run': True, 'model_loaded': False, 'dataset': args.dataset, 'subset': subset, 'split': args.split, 'dataset_version': version, 'baseline': args.mode, 'answer_options': len(config['answer_labels']), 'questions': len(questions), 'personas': len(selected_personas), 'histories': len({q['history_id'] for q in questions}), 'question_order': 'persona_id, history_id, example_id', 'context_tokens': config['context_tokens'], 'prompt_preflight': False, 'server_script': settings['ServerPath']}
        if no_memory:
            details.update(histories=0, memories_used=False, source_history_loaded=False)
        elif args.mode in ('reference_only', 'selector'):
            details[args.mode] = histories.provenance
            sample = histories.preview(questions[0]) if args.mode == 'selector' else histories.get(questions[0])[1]
            details['first_question_inputs'] = sample
            details['selector_model_loaded'] = args.mode == 'selector'
        print(json.dumps(details, indent=2), flush=True)
        return
    name = args.split + (f'_limit_{args.limit}' if args.limit else '')
    if args.mode == 'selector':
        name += f'_memory_{args.memory_tokens}' if args.memory_tokens is not None else f'_top_{args.top_k}'
        if args.score_cutoff is not None:
            name += f'_cutoff_{args.score_cutoff:g}_floor_{args.floor_tokens}'
    family = f'qwen35_4b{args.mode}_raw'
    output_base = ROOT / {'locomo': f'results/{family}/locomo_mc10/full',
                          'longbench': f'results/{family}/longbench_v2/32k'}.get(args.dataset, f'results/{family}/personamem_{args.dataset}/32k')
    if subset == 'compact':
        output_base /= 'compact'
    out = (args.output or output_base / name).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Output must stay inside this project')
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'config': config, 'split': args.split, 'limit': args.limit, 'questions': len(questions), 'personas': sorted(selected_personas), 'example_ids_sha256': hashlib.sha256(json.dumps([q['example_id'] for q in questions]).encode()).hexdigest(), 'questions_sha256': hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest(), 'targets_sha256': hashlib.sha256(json.dumps(targets, sort_keys=True).encode()).hexdigest(), 'raw_sources_sha256': raw_sources, 'splits_sha256': digest(selection_path), 'persona_split_sha256': digest(split_dir / 'persona_split.json'), 'training': False, 'memories_used': not no_memory}
    assets = json.loads((ROOT / 'configs/assets.lock.json').read_text(encoding='utf-8'))
    protocol['model_assets'] = {key: assets[key] for key in ('answer', 'llama_cpp') if key in assets}
    protocol['runtime_sources_sha256'] = {name: digest(ROOT / name) for name in ('configs/qwen_server.json', 'scripts/qwen_llamacpp_server.py', 'scripts/evaluate_qwen.py')}
    protocol['selection'] = {'subset': subset, 'question_order': 'persona_id, history_id, example_id', 'persona_split_sha256': digest(split_dir / 'persona_split.json')}
    if no_memory:
        protocol['no_memory'] = {'source_history_loaded': False, 'initial_system_persona_included': False, 'retrieval_used': False, 'input': 'shared evaluation instructions, question, and options only'}
    elif args.mode in ('reference_only', 'selector'):
        protocol[args.mode] = histories.provenance
    else:
        protocol['full_history'] = {'prompt_preflight': False, 'context_check': 'Full native prompt plus one answer token must fit before each scoring request; no truncation.', 'source_histories_sha256': raw_sources, 'visible_boundaries': 'original V1 end index; complete V2/LoCoMo history', 'initial_system_persona_included': not locomo}
    path = out / 'predictions.jsonl'
    protocol_path = out / 'protocol.json'
    previous = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.exists() else {}
    validate_resume_model(previous, config, path)
    records = read_predictions(path)
    done = {r['example_id'] for r in records}
    if len(done) != len(records) or not done.issubset({q['example_id'] for q in questions}):
        raise ValueError('Invalid or duplicate saved predictions')
    if records and (previous.get('questions_sha256') != protocol['questions_sha256'] or previous.get('targets_sha256') != protocol['targets_sha256']):
        raise ValueError('Saved predictions use different data or answer ordering; choose a new output directory')
    if records and (previous.get('config', {}).get('baseline') != args.mode
                    or previous.get('raw_sources_sha256') != protocol['raw_sources_sha256']
                    or previous.get('selector') != protocol.get('selector')
                    or previous.get('reference_only') != protocol.get('reference_only')):
        raise ValueError('Saved predictions use different memory inputs, selector settings or source files; choose a new output directory')
    runs = previous.get('server_runs', previous.get('runtime_segments', [])) if records else []
    if records and (not runs):
        runs = [{'first_prediction': 0, 'server_settings': previous.get('config', {}).get('server_settings', {})}]
    if len(records) < len(questions):
        runs = [*runs, {'first_prediction': len(records), 'server_settings': settings}]
    protocol['server_runs'] = runs
    write_json(protocol_path, protocol)
    initial = len(records)
    scored_seconds = 0
    server = None
    print(f"{config['dataset_name']} {version.upper()} {args.split}: {len(questions)} questions; {len(records)} already completed. Output: {out}", flush=True)
    try:
        if len(records) < len(questions):
            with qwen_server.LlamaCppServer(out, config) as server, path.open('a', encoding='utf-8', buffering=1) as f:
                if records:
                    with path.open('rb') as saved:
                        saved.seek(-1, os.SEEK_END)
                        if saved.read(1) != b'\n':
                            f.write('\n')
                for q in questions:
                    if q['example_id'] in done:
                        continue
                    history, source = (histories.get(q, server=server, config=config) if args.mode == 'selector'
                                       else histories.get(q))
                    units = 'visible messages'
                    print(f"[{len(records) + 1}/{len(questions)}] {q['example_id']}: {len(history)} {units}.", flush=True)
                    prediction = score_question(server, q, history, source, config)
                    correct_label = targets[q['example_id']]
                    row = {'example_id': q['example_id'], 'dataset': q['dataset'], 'split': q['split'], 'persona_id': q['persona_id'], 'question_type': q['question_type'], 'correct_label': correct_label, **prediction, 'correct': int(prediction['predicted_label'] == correct_label), 'correct_label_nll': -prediction['label_conditional_logprobs'][correct_label], 'raw_correct_label_nll': -prediction['label_raw_logprobs'][correct_label]}
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    f.flush()
                    records.append(row)
                    done.add(q['example_id'])
                    scored_seconds += prediction['elapsed_seconds']
                    per_question = scored_seconds / (len(records) - initial)
                    print(f'{len(records)}/{len(questions)} complete; average runtime {per_question:.3f}s/question; estimated remaining {per_question * (len(questions) - len(records)):.0f}s', flush=True)
        summary = save_summary(out, records, config, protocol, hardware=server.hardware if server else None)
        print(json.dumps({'output': str(out), **summary['overall']}, indent=2), flush=True)
    except BaseException as error:
        if records:
            save_summary(out, records, config, protocol, hardware=server.hardware if server else None, status='interrupted', error=error)
        print(f'Evaluation interrupted: {len(records)}/{len(questions)} predictions saved. Rerun this command to resume, or invoke without options to resume the latest unfinished run under results.', flush=True)
        raise
if __name__ == '__main__':
    main()
