"""LLMEngine — concurrent serving loop backed on persistent kernel + ring buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import queue
import threading
import time

import torch

from .model_runner import ModelRunner
from .output import GenerationEvent, OutputProcessor
from .sampling import SamplingParams
from .tokenizer_manager import TokenizerManager
from ..mpk.online_pinned_runtime import OnlinePinnedRuntime


@dataclass(frozen=True)
class PreparedGeneration:
    """Tokenized API input and its per-request generation settings."""

    token_ids: tuple[int, ...]
    config: tuple[int, ...]
    params: SamplingParams


class _StreamingMonitor:
    """Single background thread that monitors all active streaming sessions."""

    def __init__(
        self,
        runtime: OnlinePinnedRuntime,
        tokenizer_manager: TokenizerManager,
    ) -> None:
        self._runtime = runtime
        self._tokenizer_manager = tokenizer_manager
        self._sessions: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def register(
        self,
        rid: int,
        prompt_len: int,
        timeout: float,
        output: OutputProcessor | None = None,
        max_pending: int | None = None,
    ) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            if max_pending is not None and len(self._sessions) >= max_pending:
                raise OverflowError("server request queue is full")
            self._sessions[rid] = {
                "q": q,
                "output": output,
                "prompt_len": prompt_len,
                "row": -1,
                "last_step": prompt_len - 1,
                "deadline": time.monotonic() + timeout,
            }
        return q

    def unregister(self, rid: int) -> None:
        with self._lock:
            self._sessions.pop(rid, None)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    for rid, s in list(self._sessions.items()):
                        released = False
                        try:
                            completion = self._runtime.get_completion(rid)
                            if completion is not None:
                                row, final_step = completion
                                try:
                                    self._yield_remaining(s, row, final_step)
                                finally:
                                    released = self._runtime.release_request(rid)
                                if not released:
                                    raise RuntimeError(
                                        f"missing completion for rid={rid}")
                                del self._sessions[rid]
                                continue

                            if s["row"] == -1:
                                row = self._runtime.find_row_for_rid(rid)
                                if row >= 0:
                                    s["row"] = row
                                elif time.monotonic() > s["deadline"]:
                                    self._runtime.abandon_request(rid)
                                    s["q"].put(("__timeout__", True))
                                    del self._sessions[rid]
                                continue

                            row = s["row"]
                            current_step = self._runtime.get_current_step_at_row(row)

                            if s["output"] is not None:
                                if self._yield_output(s, row, current_step):
                                    self._runtime.abandon_request(rid)
                                    del self._sessions[rid]
                                    continue
                            elif current_step > s["last_step"]:
                                new_tokens = self._runtime.read_tokens_range(
                                    row, s["last_step"] + 1, current_step)
                                for tid in new_tokens.tolist():
                                    text = self._tokenizer_manager.decode_single(tid)
                                    s["last_step"] += 1
                                    s["q"].put((text, False))

                            if time.monotonic() > s["deadline"]:
                                self._runtime.abandon_request(rid)
                                s["q"].put(("__timeout__", True))
                                del self._sessions[rid]

                        except Exception:
                            try:
                                if not released:
                                    self._runtime.abandon_request(rid)
                            finally:
                                s["q"].put(("__error__", True))
                                self._sessions.pop(rid, None)

            except Exception:
                pass

            self._stop.wait(0.002)

    def _yield_output(
        self,
        s: dict,
        row: int,
        step: int,
        final: bool = False,
    ) -> bool:
        output = s["output"]
        text = ""

        if step > s["last_step"]:
            tokens = self._runtime.read_tokens_range(
                row, s["last_step"] + 1, step)
            for token in tokens.tolist():
                text += output.push(token)
            s["last_step"] = step

        reason = "stop" if output.stopped else None
        if final:
            reason = reason or self._runtime.finish_reason(row)
            text += output.flush()

        if text or reason:
            s["q"].put(GenerationEvent(
                text,
                reason,
                s["prompt_len"],
                len(output.token_ids),
            ))

        return reason is not None

    def _yield_remaining(self, s: dict, row: int, final_step: int) -> None:
        if s["output"] is not None:
            self._yield_output(s, row, final_step, final=True)
            return

        if final_step > s["last_step"]:
            new_tokens = self._runtime.read_tokens_range(
                row, s["last_step"] + 1, final_step)
            new_ids = new_tokens.tolist()
            for j, tid in enumerate(new_ids):
                text = self._tokenizer_manager.decode_single(tid)
                is_final = j == len(new_ids) - 1
                s["q"].put((text, is_final))
                s["last_step"] += 1
        else:
            s["q"].put(("", True))

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join()

        with self._lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()

        cleanup_error: Exception | None = None
        for rid, session in sessions:
            try:
                self._runtime.abandon_request(rid)
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            finally:
                session["q"].put(("__closed__", True))

        if cleanup_error is not None:
            raise RuntimeError(
                "failed to abandon a streaming request") from cleanup_error


class LLMEngine:
    """Generation loop backed by the ``online_pinned`` persistent kernel."""

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.runtime: OnlinePinnedRuntime = model_runner.runtime
        self.tokenizer_manager = TokenizerManager(
            model_runner.tokenizer, model_runner.config.developer_role)
        self.vocab_size = model_runner.vocab_size
        self.eos_ids = model_runner.eos_ids

        self._next_rid = 0
        self._submit_lock = threading.RLock()
        self._kernel_launched = threading.Event()
        self._kernel_thread: threading.Thread | None = None
        self._closed = False

        self._ensure_kernel_running()
        self._monitor = _StreamingMonitor(
            self.runtime, self.tokenizer_manager)

    def prepare(
        self,
        *,
        prompt=None,
        messages=None,
        params=None,
        use_template=False,
    ):
        if (prompt is None) == (messages is None):
            raise ValueError("provide exactly one of prompt or messages")

        if params is None:
            params = SamplingParams(
                **self.model_runner.config.sampling_defaults())

        if messages is not None:
            token_ids = self.tokenizer_manager.tokenize_messages(messages)
        elif use_template:
            token_ids = self.tokenizer_manager.tokenize(prompt)
        else:
            token_ids = self.tokenizer_manager.tokenize_raw(prompt)

        if any(t < 0 or t >= self.vocab_size for t in token_ids):
            raise ValueError("prompt token exceeds model vocabulary")

        config = params.pack(
            len(token_ids),
            self.model_runner.config.max_seq_length,
            self.vocab_size,
            self.eos_ids,
        )
        return PreparedGeneration(
            tuple(token_ids), tuple(config), params)

    def generate(
        self,
        request: PreparedGeneration,
        timeout=120.0,
    ):
        tokens = torch.tensor(request.token_ids, dtype=torch.int64)
        output = OutputProcessor(
            self.tokenizer_manager,
            request.params.stop,
            self.eos_ids,
            request.params.stop_token_sequences,
        )

        with self._submit_lock:
            if self._closed:
                raise RuntimeError("LLMEngine is closed")

            rid = self._next_rid
            self._next_rid += 1

            q = self._monitor.register(
                rid,
                len(tokens),
                timeout,
                output,
                self.model_runner.config.max_pending_requests,
            )

            try:
                self.runtime.submit(
                    rid, tokens, generation_config=request.config)
            except Exception:
                self._monitor.unregister(rid)
                raise

        return self._submit_stream(rid, q)

    def submit(
        self,
        prompt: str,
        use_template: bool = True,
        timeout: float = 120.0,
        poll_interval: float = 1e-4,
        stream: bool = False,
    ):
        """Submit a single prompt for generation."""

        token_ids = self.tokenizer_manager.tokenize(prompt, use_template)
        prompt_len = len(token_ids)

        config = SamplingParams(
            **self.model_runner.config.sampling_defaults()
        ).pack(
            prompt_len,
            self.model_runner.config.max_seq_length,
            self.vocab_size,
            self.eos_ids,
        )

        tokens = torch.tensor(token_ids, dtype=torch.int64)
        stream_queue: queue.Queue | None = None

        with self._submit_lock:
            if self._closed:
                raise RuntimeError("LLMEngine is closed")

            rid = self._next_rid
            self._next_rid += 1

            if stream:
                stream_queue = self._monitor.register(
                    rid, prompt_len, timeout)

            try:
                self.runtime.submit(
                    rid, tokens, generation_config=config)
            except Exception:
                if stream:
                    self._monitor.unregister(rid)
                raise

        if stream:
            assert stream_queue is not None
            return self._submit_stream(rid, stream_queue)

        buffer_row, final_step = self.runtime.wait_for_request(
            rid, timeout, poll_interval)
        try:
            full_tokens = self.runtime.read_tokens_at_row(
                buffer_row, final_step)
            output_ids = full_tokens[prompt_len:].tolist()
            return {
                "text": self.tokenizer_manager.decode(output_ids),
                "token_ids": output_ids,
            }
        finally:
            self.runtime.release_request(rid)

    def _ensure_kernel_running(self) -> None:
        if self._kernel_launched.is_set():
            return

        with self._submit_lock:
            if self._kernel_launched.is_set():
                return

            self.runtime.reset()
            self.runtime.start()
            self._kernel_thread = threading.Thread(
                target=self.model_runner, daemon=True)
            self._kernel_thread.start()
            self._kernel_launched.set()

    def _submit_stream(
        self,
        rid: int,
        q: queue.Queue,
    ):
        def generator():
            while True:
                try:
                    item = q.get(timeout=0.05)
                except queue.Empty:
                    continue

                if isinstance(item, GenerationEvent):
                    yield item
                    if item.finish_reason is not None:
                        break
                    continue

                text, is_final = item

                if text == "__timeout__":
                    raise TimeoutError(
                        f"stream timed out for rid={rid}")
                if text == "__error__":
                    raise RuntimeError(
                        f"stream error for rid={rid}")
                if text == "__closed__":
                    raise RuntimeError(
                        f"engine closed while streaming rid={rid}")

                yield text, is_final
                if is_final:
                    break

        return generator()

    def close(self) -> None:
        with self._submit_lock:
            if self._closed:
                return
            self._closed = True

        monitor_error: Exception | None = None
        try:
            self._monitor.shutdown()
        except Exception as exc:
            monitor_error = exc

        try:
            self.runtime.request_shutdown()
            if self._kernel_thread is not None:
                self._kernel_thread.join()
        finally:
            self.runtime.stop()

        if monitor_error is not None:
            raise monitor_error