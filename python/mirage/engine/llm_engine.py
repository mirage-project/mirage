"""Concurrent generation backed by the persistent GPU kernel and pinned rings."""
from __future__ import annotations

from dataclasses import dataclass

import queue
import threading
import time

from .config import DEFAULT_REQUEST_TIMEOUT
from .output import GenerationEvent, OutputProcessor
from .sampling import SamplingParams
from .tokenizer_manager import TokenizerManager


@dataclass(frozen=True)
class PreparedGeneration:
    """Tokenized input and its bound settings, kept together until submission."""
    token_ids: tuple[int, ...]
    config: tuple[int, ...]
    params: SamplingParams


class GenerationSession:
    def __init__(self, engine, rid, prompt_len, params, timeout):
        self.engine = engine
        self.rid = rid
        self.prompt_len = prompt_len
        self.deadline = time.monotonic() + timeout
        self.output = OutputProcessor(engine.tokenizer_manager, params.stop, engine.eos_ids,
                                      params.stop_token_sequences)
        self.last_step = prompt_len - 1
        self.queue = queue.Queue()

    def __iter__(self):
        try:
            while True:
                event = self.queue.get()
                if isinstance(event, Exception):
                    raise event
                yield event
                if event.finish_reason is not None:
                    return
        finally:
            self.close()

    def close(self):
        self.engine.cancel(self.rid)


class LLMEngine:
    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.runtime = model_runner.runtime
        self.tokenizer_manager = TokenizerManager(model_runner.tokenizer,
                                                  model_runner.config.developer_role)
        self.vocab_size = model_runner.vocab_size
        self.eos_ids = model_runner.eos_ids
        self._next_rid = 0
        self._lock = threading.RLock()
        self._sessions = {}
        self._closed = False
        self._kernel_error = None
        self._stop = threading.Event()
        self.runtime.reset()
        self.runtime.start()
        self._kernel_thread = threading.Thread(target=self._run_kernel, daemon=True)
        self._kernel_thread.start()
        self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self._monitor_thread.start()

    def _run_kernel(self):
        try:
            self.model_runner()
            if not self._closed:
                self._kernel_error = RuntimeError("GPU kernel exited unexpectedly")
        except Exception as exc:
            self._kernel_error = exc

    def prepare(self, *, prompt=None, messages=None, params=None, use_template=False):
        if (prompt is None) == (messages is None):
            raise ValueError("provide exactly one of prompt or messages")
        if params is None:
            defaults = getattr(self.model_runner.config, "sampling_defaults", lambda: {})()
            params = SamplingParams(**defaults)
        ids = (self.tokenizer_manager.tokenize_messages(messages) if messages is not None
               else self.tokenizer_manager.tokenize(prompt, use_template))
        if any(t < 0 or t >= self.vocab_size for t in ids):
            raise ValueError("prompt token exceeds model vocabulary")
        packed = params.pack(len(ids), self.model_runner.config.max_seq_length,
                             self.vocab_size, self.eos_ids)
        return PreparedGeneration(tuple(ids), tuple(packed), params)

    def generate(self, request: PreparedGeneration, timeout=None):
        import torch
        timeout = DEFAULT_REQUEST_TIMEOUT if timeout is None else timeout
        with self._lock:
            if self._closed:
                raise RuntimeError("engine is closed")
            if self._kernel_error:
                raise RuntimeError("GPU kernel failed") from self._kernel_error
            if len(self._sessions) >= self.model_runner.config.max_pending_requests:
                raise OverflowError("server request queue is full")
            rid = self._next_rid
            self._next_rid += 1
            session = GenerationSession(self, rid, len(request.token_ids), request.params, timeout)
            self._sessions[rid] = session
            try:
                self.runtime.submit(rid, torch.tensor(request.token_ids, dtype=torch.int64),
                                    generation_config=request.config)
            except Exception:
                del self._sessions[rid]
                raise
            return session

    def submit(self, prompt: str | None = None, use_template=True, timeout=None, poll_interval=None,
               stream=False, sampling_params=None, messages=None):
        """Python compatibility API. HTTP uses prepare/generate structured events."""
        request = self.prepare(prompt=None if messages is not None else prompt,
                               messages=messages, params=sampling_params, use_template=use_template)
        session = self.generate(request, timeout)
        if stream:
            def iterator():
                try:
                    for event in session:
                        yield event.text, event.finish_reason is not None
                finally:
                    session.close()
            return iterator()
        text = ""
        for event in session:
            text += event.text
        return dict(text=text, token_ids=session.output.token_ids,
                    finish_reason=event.finish_reason, usage=event.usage)

    def cancel(self, rid):
        with self._lock:
            session = self._sessions.pop(rid, None)
            if session is not None:
                self.runtime.abandon_request(rid)
                session.queue.put(RuntimeError("request cancelled"))

    def _monitor(self):
        while not self._stop.is_set():
            with self._lock:
                for rid, session in list(self._sessions.items()):
                    try:
                        if self._kernel_error:
                            raise RuntimeError("GPU kernel failed") from self._kernel_error
                        if time.monotonic() > session.deadline:
                            raise TimeoutError("generation timed out")
                        completion = self.runtime.get_completion(rid)
                        row = completion[0] if completion else self.runtime.find_row_for_rid(rid)
                        if row < 0:
                            continue
                        step = completion[1] if completion else self.runtime.get_current_step_at_row(row)
                        delta = ""
                        if step > session.last_step:
                            tokens = self.runtime.read_tokens_range(row, session.last_step + 1, step).tolist()
                            for token in tokens:
                                delta += session.output.push(token)
                            session.last_step = step
                        reason = None
                        if session.output.stopped:
                            reason = "stop"
                        elif completion:
                            reason = self.runtime.finish_reason(row)
                            delta += session.output.flush()
                        if delta or reason:
                            session.queue.put(GenerationEvent(delta, reason, session.prompt_len,
                                                              len(session.output.token_ids)))
                        if reason:
                            if completion:
                                self.runtime.release_request(rid)
                            else:
                                self.runtime.abandon_request(rid)
                            del self._sessions[rid]
                    except Exception as exc:
                        self.runtime.abandon_request(rid)
                        session.queue.put(exc)
                        self._sessions.pop(rid, None)
            self._stop.wait(0.002)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for rid in list(self._sessions):
                self.cancel(rid)
        self._stop.set()
        self._monitor_thread.join()
        self.runtime.request_shutdown()
        self._kernel_thread.join()
        self.runtime.stop()
