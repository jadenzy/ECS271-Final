"""Own the native llama.cpp process, HTTP API, scoring and resource monitoring.

The evaluator passes prepared messages to LlamaCppServer and receives scores.
This module owns startup, health checks, tokenization, requests, retries and cleanup.
Run this file for a standalone server; --print-settings never loads a model.
"""
import argparse
import bisect
from collections import OrderedDict
import ctypes
from datetime import datetime, timezone
import hashlib
import math
import re
import socket
import statistics
import threading
import time

import requests
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTOR_MEMORY_TOKENS = 2048


class _ModelParams(ctypes.Structure):
    """llama_model_params from the pinned b11669 / 620d2225 llama.h."""

    _fields_ = [(name, ctypes.c_void_p) for name in ('devices', 'tensor_buft_overrides')] + [
        (name, ctypes.c_int32) for name in ('n_gpu_layers', 'split_mode', 'load_mode', 'lazy_mode', 'main_gpu')
    ] + [(name, ctypes.c_void_p) for name in (
        'tensor_split', 'progress_callback', 'progress_callback_user_data', 'kv_overrides'
    )] + [(name, ctypes.c_bool) for name in (
        'vocab_only', 'check_tensors', 'use_extra_bufts', 'no_host', 'no_alloc', 'load_mtp'
    )]


_LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_QUIET_NATIVE_LOG = _LOG_CALLBACK(lambda *_: None)


class LocalQwenTokenizer:
    """The server's pinned native tokenizer, with vocabulary only and no weights.

    Render the GGUF's own Jinja template. Startup probes and every final selector
    prompt are also checked against the server; a mismatch stops before scoring.
    """

    def __init__(self, settings, *, root=ROOT):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        self.model, self.library = None, None
        assets = json.loads((Path(root) / 'configs/assets.lock.json').read_text(encoding='utf-8'))['llama_cpp']
        if assets['git_revision'] != '620d2225cc436ab0b001e3200e590e3b286ef9ab':
            raise ValueError('Update the local tokenizer ABI before changing the pinned llama.cpp runtime')
        directory = Path(settings['ServerPath']).parent
        library = directory / 'llama.dll'
        expected = next(item['sha256'] for item in assets['files'] if item['path'].endswith('/llama.dll'))
        with library.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Local tokenizer llama.dll differs from the pinned runtime; verify model assets')
        # Keep dependency lookup active while the vocabulary is alive. No GPU
        # backends or inference contexts are initialized in this process.
        self._dll_directory = os.add_dll_directory(str(directory.resolve()))
        try:
            dll = self.library = ctypes.CDLL(str(library.resolve()))
            dll.llama_log_set.argtypes = [_LOG_CALLBACK, ctypes.c_void_p]
            dll.llama_log_set(_QUIET_NATIVE_LOG, None)
            dll.llama_model_default_params.restype = _ModelParams
            dll.llama_model_load_from_file.argtypes = [ctypes.c_char_p, _ModelParams]
            dll.llama_model_load_from_file.restype = ctypes.c_void_p
            dll.llama_model_free.argtypes = [ctypes.c_void_p]
            dll.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
            dll.llama_model_get_vocab.restype = ctypes.c_void_p
            dll.llama_model_chat_template.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            dll.llama_model_chat_template.restype = ctypes.c_char_p
            dll.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                                          ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                                          ctypes.c_bool, ctypes.c_bool]
            dll.llama_tokenize.restype = ctypes.c_int32
            params = dll.llama_model_default_params()
            params.vocab_only, params.n_gpu_layers = True, 0
            self.model = dll.llama_model_load_from_file(str(Path(settings['ModelPath']).resolve()).encode('utf-8'), params)
            if not self.model:
                raise RuntimeError('Could not load the local Qwen GGUF vocabulary')
            self.vocab = dll.llama_model_get_vocab(self.model)
            dll.llama_model_meta_val_str.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
            dll.llama_model_meta_val_str.restype = ctypes.c_int32
            pre = ctypes.create_string_buffer(128)
            dll.llama_model_meta_val_str(self.model, b'tokenizer.ggml.pre', pre, len(pre))
            dll.llama_vocab_get_add_eos.argtypes = [ctypes.c_void_p]
            dll.llama_vocab_get_add_eos.restype = ctypes.c_bool
            self.cache_block_counts = pre.value == b'qwen35' and not dll.llama_vocab_get_add_eos(self.vocab)
            self._block_counts = OrderedDict()
            self._lock = threading.Lock()
            template = dll.llama_model_chat_template(self.model, None)
            if not template:
                raise ValueError('The Qwen GGUF has no chat template')
            self.template_sha256 = hashlib.sha256(template).hexdigest()
            environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            def raise_exception(message):
                raise ValueError(message)
            environment.globals['raise_exception'] = raise_exception
            self.template = environment.from_string(template.decode('utf-8'))
        except BaseException:
            self.close()
            raise

    def tokenize(self, text, *, add_special=True, parse_special=True):
        if self.model is None:
            raise RuntimeError('Local Qwen tokenizer is closed')
        data = text.encode('utf-8')
        capacity = max(32, len(data) // 2 + 16)
        while True:
            tokens = (ctypes.c_int32 * capacity)()
            count = self.library.llama_tokenize(self.vocab, data, len(data), tokens, capacity,
                                                add_special, parse_special)
            if count >= 0:
                return list(tokens[:count])
            if count == -(2 ** 31):
                raise ValueError('Qwen tokenization exceeded the native token limit')
            capacity = -count

    def render_prompt(self, messages):
        if any(message['role'] not in ('system', 'user') or not isinstance(message['content'], str)
               for message in messages):
            raise ValueError('Local evaluator prompts must contain text system/user messages')
        prompt = self.template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
        return prompt, self.token_count(prompt)

    def token_count(self, text):
        if self.model is None:
            raise RuntimeError('Local Qwen tokenizer is closed')
        if not self.cache_block_counts:
            return len(self.tokenize(text))
        # Qwen35 BPE never merges across a newline followed by '[': its
        # punctuation/whitespace pre-tokenizer alternatives end at the newline.
        # Keep the newline on the LEFT, unlike adding independently measured
        # raw-message lengths. Internal punctuation, whitespace, and all other
        # boundaries are still tokenized together by the exact native tokenizer.
        # See the pinned llama-vocab.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35 regex.
        total = 0
        for index, block in enumerate(re.split(r'(?<=\n)(?=\[)', text)):
            key = (block, index == 0)
            with self._lock:
                if key not in self._block_counts:
                    self._block_counts[key] = len(self.tokenize(block, add_special=index == 0))
                    while len(self._block_counts) > 4096:
                        self._block_counts.popitem(last=False)
                self._block_counts.move_to_end(key)
                total += self._block_counts[key]
        return total

    def close(self):
        if self.model is not None:
            self.library.llama_model_free(self.model)
            self.model = None
        if getattr(self, '_dll_directory', None) is not None:
            self._dll_directory.close()
            self._dll_directory = None
        if hasattr(self, '_block_counts'):
            self._block_counts.clear()


def parse_args(argv=None, *, root=None):
    root = Path(root) if root is not None else ROOT
    config = json.loads((root / 'configs/qwen_server.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['v2', 'v1', 'locomo', 'longbench'], default='v2')
    parser.add_argument('--mode', choices=['full_history', 'no_memory', 'reference_only', 'selector'], default='full_history')
    parser.add_argument('--ctx-size', type=int)
    parser.add_argument('--memory-tokens', type=int,
                        help='Selector mode: memory budget; default context is this budget plus 1000 tokens')
    parser.add_argument('--host', default=config['host'])
    parser.add_argument('--port', type=int, default=config['port'])
    parser.add_argument('--print-settings', action='store_true')
    args = parser.parse_args(argv)
    if args.memory_tokens is not None and (args.mode != 'selector' or args.memory_tokens < 1):
        parser.error('--memory-tokens requires selector mode and a positive budget')
    if args.mode == 'selector':
        args.memory_tokens = args.memory_tokens or DEFAULT_SELECTOR_MEMORY_TOKENS
    if args.ctx_size is None:
        if args.mode == 'selector':
            overhead = config['selector_context_overhead_tokens']
            if type(overhead) is not int or overhead < 1:
                parser.error('selector_context_overhead_tokens must be a positive integer')
            args.ctx_size = args.memory_tokens + overhead
        else:
            key = 'no_memory_context_tokens' if args.mode == 'no_memory' else f'{args.dataset}_context_tokens'
            args.ctx_size = config[key]
    if (args.host != '127.0.0.1' or not 1 <= args.port <= 65535
            or type(args.ctx_size) is not int or not 2 <= args.ctx_size <= 262144):
        parser.error('Invalid loopback address, port or context size')
    if config.get('backend') != 'llama_cpp' or not config['model_path'].endswith('.gguf'):
        parser.error('The active configuration must specify the llama_cpp GGUF backend')
    args.project_root, args.config = root, config
    return args


def connection_settings(args):
    config, root = args.config, args.project_root
    server = (root / config['server_path']).resolve()
    model = (root / config['model_path']).resolve()
    generation = config['generation']
    command = [str(server), '--model', str(model), '--alias', config['model'],
               '--ctx-size', str(args.ctx_size), '--n-gpu-layers', '-1',
               '--flash-attn', 'on', '--host', args.host, '--port', str(args.port),
               '--parallel', '1', '--cache-prompt', '--reasoning', 'off', '--jinja',
               '--chat-template-kwargs', '{"enable_thinking":false}',
               '--no-context-shift', '--fit', 'off']
    for name, flag in [('temperature', '--temp'), ('top_p', '--top-p'), ('top_k', '--top-k'),
                       ('min_p', '--min-p'), ('presence_penalty', '--presence-penalty'),
                       ('repetition_penalty', '--repeat-penalty')]:
        command.extend([flag, str(generation[name])])
    return {
        'Command': command, 'Environment': {'GGML_CUDA_DISABLE_GRAPHS': '1'},
        'ServerPath': str(server), 'ModelPath': str(model), 'Model': config['model'],
        'Backend': 'llama_cpp', 'Dtype': config['dtype'], 'Format': 'gguf',
        'CpuOffload': False, 'GpuLayers': -1, 'FlashAttention': True,
        'Thinking': False, 'Generation': {'do_sample': True, **generation},
        'CachePrompt': True, 'ContextShift': False, 'Fit': False,
        'ContextTokens': args.ctx_size, 'Host': args.host, 'Port': args.port,
        'Dataset': args.dataset, 'ChatTemplateSource': 'gguf_metadata',
        'BaseUrl': f'http://{args.host}:{args.port}',
        'Client': 'requests',
        'ScoringEndpoint': '/completion', 'RawLogprobs': True,
        **({'ModelRevision': config['model_revision']} if 'model_revision' in config else {}),
        'StartupTimeoutSeconds': config['startup_timeout_seconds'],
        'RequestTimeoutSeconds': config['request_timeout_seconds'],
    }


def server_settings(mode='full_history', *, dataset='v2', root=None, memory_tokens=None):
    argv = ['--dataset', dataset, '--mode', mode]
    if memory_tokens is not None:
        argv += ['--memory-tokens', str(memory_tokens)]
    return connection_settings(parse_args(argv, root=root))


def process_ram_bytes(process):
    """Windows working-set counters for the Qwen model server process."""
    if os.name != 'nt' or process is None or process.poll() is not None:
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    handle = kernel.OpenProcess(1040, False, process.pid)
    if not handle:
        return None
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return {'working_set': counters.WorkingSetSize, 'peak_working_set': counters.PeakWorkingSetSize, 'peak_commit': counters.PeakPagefileUsage}
    finally:
        kernel.CloseHandle(handle)
    return None

def memory_statistics(samples):
    gpu = [s['gpu_used_mib'] for s in samples if s.get('gpu_used_mib') is not None]
    ram = [s['ram_working_set_mib'] for s in samples if s.get('ram_working_set_mib') is not None]
    return {'mean_gpu_memory_mib': statistics.mean(gpu) if gpu else None, 'peak_gpu_memory_mib': max(gpu) if gpu else None, 'mean_ram_working_set_mib': statistics.mean(ram) if ram else None, 'peak_ram_working_set_mib': max(ram) if ram else None, 'gpu_memory_samples': len(gpu), 'ram_memory_samples': len(ram)}

class LlamaCppServer:
    """Own native process lifecycle, HTTP calls, label scoring and monitoring."""

    def __init__(self, out, config):
        self.out, self.config = (out, config)
        self.process = None
        self.stop_sampling = threading.Event()
        self.memory_samples = []
        self.sampling_lock = threading.Lock()
        self.hardware = {}
        self.server_restart_count = 0
        self.load_seconds = 0.0
        self.session = requests.Session()
        self.session.trust_env = False
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self.session.mount('http://', adapter)
        self._prompt_lock = threading.Lock()
        self.local_tokenizer = None
        self._rendered_prompts = OrderedDict()

    def sample_memory(self):
        with self.sampling_lock:
            sample = {}
            try:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                used, total = map(int, raw.strip().splitlines()[0].split(','))
                sample.update(gpu_used_mib=used, gpu_total_mib=total)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            try:
                ram = process_ram_bytes(self.process)
                if ram:
                    sample.update(ram_working_set_mib=ram['working_set'] / 1024 ** 2, ram_peak_working_set_mib=ram['peak_working_set'] / 1024 ** 2, ram_peak_commit_mib=ram['peak_commit'] / 1024 ** 2)
            except OSError:
                pass
            sample['sampled_at'] = time.perf_counter()
            self.memory_samples.append(sample)
            return sample

    def sample_resources(self):
        stop_sampling = self.stop_sampling
        while not stop_sampling.is_set():
            self.sample_memory()
            stop_sampling.wait(1)

    def question_memory(self, started, finished, *, snapshot=True):
        # Samples are appended in time order under the sampling lock.
        first = bisect.bisect_left(self.memory_samples, started, key=lambda s: s['sampled_at'])
        last = bisect.bisect_right(self.memory_samples, finished, key=lambda s: s['sampled_at'])
        samples = self.memory_samples[first:last]
        source = 'during_request'
        if not samples and not snapshot and self.memory_samples:
            # Concurrent judge requests reuse the background sample instead of serializing on nvidia-smi.
            samples, source = [self.memory_samples[-1]], 'latest_background_sample'
        elif not samples:
            samples = [self.sample_memory()]
            source = 'post_request_snapshot'
        return {**memory_statistics(samples), 'memory_sampling': source}

    def __enter__(self):
        settings = self.config['server_settings']
        binary = Path(settings['Command'][0])
        with socket.socket() as sock:
            sock.bind((settings['Host'], settings['Port']))
        self.base = settings['BaseUrl']
        self.log = (self.out / 'qwen-server.log').open('a' if self.server_restart_count else 'w', encoding='utf-8')
        if self.server_restart_count:
            self.log.write(f'\nRecovery server restart {self.server_restart_count} at {datetime.now(timezone.utc).isoformat()}\n')
        args = settings['Command']
        self.log.write('Command: ' + subprocess.list2cmdline(args) + '\n')
        self.log.flush()
        self.started = time.perf_counter()
        try:
            environment = {**os.environ, **settings.get('Environment', {})}
            self.process = subprocess.Popen(args, cwd=binary.parent, env=environment, stdout=self.log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = time.monotonic() + settings['StartupTimeoutSeconds']
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Qwen model server exited; see qwen-server.log. CUDA model residency is required; check the log for loading or memory errors. History will not be shortened to fit VRAM.')
                try:
                    if self.session.get(self.base + '/health', timeout=2).status_code == 200:
                        self.load_seconds += time.perf_counter() - self.started
                        self.props = self.session.get(self.base + '/props', timeout=10).json()
                        self.actual_context = self.props.get('default_generation_settings', {}).get('n_ctx')
                        self.label_ids = {}
                        for label in self.config['answer_labels']:
                            ids = self.post('/tokenize', {'content': label, 'add_special': False})['tokens']
                            if len(ids) != 1:
                                raise ValueError(f'Answer label is not a single token: {label}')
                            self.label_ids[label] = ids[0]
                        if self.config.get('baseline') == 'selector' and self.config.get('memory_tokens') is not None:
                            self.local_tokenizer = LocalQwenTokenizer(settings)
                            self.verify_local_tokenizer()
                        self.sampler = threading.Thread(target=self.sample_resources, daemon=True)
                        self.sampler.start()
                        return self
                except requests.RequestException:
                    pass
                time.sleep(1)
            raise TimeoutError('Qwen model server did not become healthy')
        except BaseException:
            self.close()
            raise

    def restart(self):
        """Replace only this evaluator's owned native process and HTTP session."""
        self.close()
        self.process = None
        self.stop_sampling = threading.Event()
        self.session = requests.Session()
        self.session.trust_env = False
        self.server_restart_count += 1
        self.__enter__()

    def post(self, path, payload):
        if path == '/completion' and self.config.get('baseline') != 'judge':
            payload = {**payload, 'cache_prompt': True}
        elif path == '/completion':
            payload = {**payload, 'cache_prompt': False}
        timeout = self.config['server_settings']['RequestTimeoutSeconds']
        if self.config.get('baseline') == 'judge':
            timeout = (10, 120)
        try:
            response = self.session.post(self.base + path, json=payload, timeout=timeout)
        except requests.RequestException as error:
            code = self.process.poll() if self.process is not None else None
            log_path = self.out / 'qwen-server.log'
            diagnostics = []
            try:
                with log_path.open('rb') as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 8192))
                    tail = stream.read().decode('utf-8', errors='replace')
                diagnostics = [line.strip() for line in tail.splitlines() if any((marker in line.lower() for marker in ('cuda error', 'ggml_assert', 'fatal error', 'out of memory')))][-4:]
            except OSError:
                pass
            state = f'exited with code {code}' if code is not None else 'closed the connection or did not respond'
            detail = ' '.join(diagnostics) or str(error)
            raise RuntimeError(f'Qwen model server {state} during {path}. {detail} Saved predictions are preserved. See {log_path}') from error
        if response.status_code != 200:
            raise RuntimeError(f'{path}: HTTP {response.status_code}: {response.text[:1000]}')
        return response.json()

    def render_prompt_http(self, messages):
        """Reference implementation used to audit local rendering/tokenization."""
        prompt = self.post('/apply-template', {'messages': messages})['prompt']
        token_count = len(self.post('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens'])
        return (prompt, token_count)

    @staticmethod
    def _prompt_key(messages):
        return tuple((m['role'], m['content']) for m in messages)

    def remember_rendered_prompt(self, messages, rendered):
        """Keep a bounded cache; selector packing pins its bare and final prompts."""
        key = self._prompt_key(messages)
        with self._prompt_lock:
            self._rendered_prompts[key] = rendered
            self._rendered_prompts.move_to_end(key)
            while len(self._rendered_prompts) > 4:
                self._rendered_prompts.popitem(last=False)

    def render_prompt(self, messages):
        if self.local_tokenizer is None:
            return self.render_prompt_http(messages)
        key = self._prompt_key(messages)
        with self._prompt_lock:
            if key in self._rendered_prompts:
                return self._rendered_prompts[key]
        rendered = self.local_tokenizer.render_prompt(messages)
        self.remember_rendered_prompt(messages, rendered)
        return rendered

    def verify_local_tokenizer(self):
        probes = [
            [{'role': 'system', 'content': ' Instructions. '},
             {'role': 'user', 'content': 'Unicode: café 中文 🐈\nTabs\tand punctuation!\n\na) A\nb) B\nAnswer:'}],
            [{'role': 'system', 'content': 'Instructions.'},
             {'role': 'user', 'content': ' Selected messages:\n\n[Message 2 | role: assistant]\n<|im_end|>\n\n '},
             {'role': 'user', 'content': 'Question: why?\nOptions:\na) A\nb) B\nAnswer:'}],
        ]
        for messages in probes:
            prompt, count = self.local_tokenizer.render_prompt(messages)
            if (prompt, count) != self.render_prompt_http(messages):
                raise ValueError('Local Qwen chat template/token counts differ from the active server')
            native = self.post('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens']
            if self.local_tokenizer.tokenize(prompt) != native:
                raise ValueError('Local Qwen token IDs differ from the active server')

    def verify_selector_prompt(self, messages, rendered, bare_messages, bare_rendered):
        """Check each final prompt with the server, then reuse it during scoring."""
        if self.local_tokenizer is not None:
            if self.render_prompt_http(messages) != rendered:
                raise ValueError('Local selector prompt differs from the server; no completion requested')
            self.remember_rendered_prompt(bare_messages, bare_rendered)
            self.remember_rendered_prompt(messages, rendered)

    def score(self, messages, *, request_id, history_messages=0, selection_seconds=0):
        """Return raw/conditional label scores for prepared messages, without gold answers."""
        started = time.perf_counter()
        context = getattr(self, 'actual_context', None) or self.config['context_tokens']
        if self.config.get('baseline') == 'selector' and self.config.get('memory_tokens') is not None:
            context = min(context, self.config['context_tokens'])
        prompt, token_count = self.render_prompt(messages)
        if token_count + 1 > context:
            raise ValueError(f"Complete prompt for {request_id} needs {token_count + 1} tokens; server context is {context}. History will not be truncated.")
        if self.config.get('baseline') == 'no_memory':
            history_tokens = 0
            print(f'  Prompt: {token_count:,} Qwen tokens, question/options only; scoring...', flush=True)
        elif self.config.get('baseline') == 'judge':
            history_tokens = 1
        else:
            _, bare_count = self.render_prompt([messages[0], messages[-1]])
            history_tokens = token_count - bare_count
            budget = self.config.get('memory_tokens') if self.config.get('baseline') == 'selector' else None
            if budget is not None and history_tokens > budget:
                raise ValueError(f'Selected history for {request_id} uses {history_tokens} Qwen tokens; memory budget is {budget}. No completion requested.')
            print(f'  Full prompt: {token_count:,} Qwen tokens, {history_messages} source messages; scoring...', flush=True)
        n_probs, attempts = (self.config['initial_top_logprobs'], 0)
        invalid_retries = 0
        restarted = False
        inference_seconds = 0
        while True:
            attempts += 1
            before = time.perf_counter()
            request = {'prompt': prompt, 'n_predict': 1, 'n_probs': n_probs, 'post_sampling_probs': False}
            response = self.post('/completion', request)
            inference_seconds += time.perf_counter() - before
            distributions = response.get('completion_probabilities', response.get('probs', []))
            if not distributions or response.get('truncated'):
                raise ValueError('Missing logits or truncated prompt')
            scores = {p['id']: p.get('logprob') for p in distributions[0].get('top_logprobs', [])}
            invalid = [label for label, token in self.label_ids.items() if token in scores and (type(scores[token]) not in (int, float) or not math.isfinite(scores[token]))]
            if invalid:
                detail = f"Invalid/null answer-label log-probabilities ({', '.join(invalid)}) for {request_id}"
                if invalid_retries >= 2:
                    if restarted:
                        raise ValueError(detail + ' after a fresh server restart and 2 retries; no prediction saved for this question. See qwen-server.log.')
                    print(f'  {detail}; retries failed. Restarting the owned Qwen model server once and retrying the same full prompt.', flush=True)
                    self.restart()
                    restarted = True
                    invalid_retries = 0
                    n_probs = self.config['initial_top_logprobs']
                    continue
                invalid_retries += 1
                print(f'  {detail}; retrying the full prompt with server defaults ({invalid_retries}/2).', flush=True)
                continue
            if all((token in scores for token in self.label_ids.values())):
                break
            if n_probs >= self.config['maximum_top_logprobs']:
                raise ValueError('Not all answer labels are present in the returned log-probabilities')
            n_probs = min(n_probs * 4, self.config['maximum_top_logprobs'])
        logprobs = {label: float(scores[token]) for label, token in self.label_ids.items()}
        maximum = max(logprobs.values())
        normalizer = maximum + math.log(sum((math.exp(v - maximum) for v in logprobs.values())))
        conditional = {label: value - normalizer for label, value in logprobs.items()}
        finished = time.perf_counter()
        ranking_seconds = selection_seconds
        measurement = {'snapshot': False} if self.config.get('baseline') == 'judge' else {}
        memory = self.question_memory(started - ranking_seconds, finished, **measurement)
        return {'predicted_label': max(logprobs, key=logprobs.get), 'label_raw_logprobs': logprobs, 'label_conditional_logprobs': conditional, 'prompt_tokens': token_count, 'memory_tokens': history_tokens, 'history_input_tokens': history_tokens, 'inference_seconds': inference_seconds, 'elapsed_seconds': finished - started + ranking_seconds, **memory, 'logprob_attempts': attempts, 'server_restarts': int(restarted), 'top_logprobs_requested': n_probs, 'timings': response.get('timings', {}), 'prefill_chunks': response.get('prefill_chunks'), 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}

    def close(self):
        self.stop_sampling.set()
        if hasattr(self, 'sampler'):
            self.sampler.join(timeout=6)
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.local_tokenizer is not None:
            self.local_tokenizer.close()
            self.local_tokenizer = None
        self._rendered_prompts.clear()
        self.log.close()
        self.session.close()
        samples = self.memory_samples
        gpu = [s for s in samples if s.get('gpu_used_mib') is not None]
        ram = [s for s in samples if s.get('ram_working_set_mib') is not None]
        info = {'scope': 'All model server sessions in the most recent evaluator invocation; per-question aggregate metrics also include resumed predictions.', 'server_load_seconds': self.load_seconds, 'server_restart_count': self.server_restart_count, **memory_statistics(samples), 'gpu_memory_measurement': 'Device-wide NVIDIA GPU memory, including desktop/other processes. Samples start after each model startup.', 'device_memory_total_mib': gpu[0]['gpu_total_mib'] if gpu else None}
        model_info = getattr(self, 'props', {}).get('model_info', {})
        info['model_info'] = model_info
        info['gpu_offload_confirmed'] = model_info.get('cpu_offload') is False if model_info else None
        info['layers_on_gpu'] = model_info.get('layers_on_gpu')
        info['layers_on_cpu'] = 0 if model_info.get('cpu_offload') is False else None
        info['context_tokens_allocated'] = getattr(self, 'actual_context', None)
        info['cpu_offload_log'] = []
        info['server_peak_ram_working_set_mib'] = max((r['ram_peak_working_set_mib'] for r in ram)) if ram else None
        info['server_peak_ram_commit_mib'] = max((r['ram_peak_commit_mib'] for r in ram)) if ram else None
        info['host_memory_measurement'] = 'Windows native Qwen server process working set; excludes Python/evaluator RAM.'
        self.hardware = info

    def __exit__(self, *_):
        self.close()

def main(argv=None):
    args = parse_args(argv)
    settings = connection_settings(args)
    if args.print_settings:
        print(json.dumps(settings, indent=2))
        return 0
    for key in ('ServerPath', 'ModelPath'):
        if not Path(settings[key]).is_file():
            raise FileNotFoundError(f'{key} is missing: {settings[key]}. Run scripts/download_models.py.')
    return subprocess.run(settings['Command'], cwd=Path(settings['ServerPath']).parent,
                          env={**os.environ, **settings['Environment']}).returncode


if __name__ == '__main__':
    raise SystemExit(main())
