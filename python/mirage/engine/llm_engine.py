"""LLMEngine — concurrent serving loop backed on persistent kernel + ring buffer.
"""

from __future__ import annotations

import threading

import torch

from .model_runner import ModelRunner
from .tokenizer_manager import TokenizerManager
from ..mpk.online_pinned_runtime import OnlinePinnedRuntime


class LLMEngine:
    """Generation loop backed by the ``online_pinned`` persistent kernel.

    The kernel runs in a background thread so the engine can accept requests
    concurrently.  Each call to :meth:`submit` allocates a unique, never-
    repeating *request id* (rid), stages tokens in the pinned inbox, writes a
    ring-buffer entry, and then blocks until the GPU reports completion for
    that specific rid.  The GPU manages its own buffer-row pool and a
    waiting/running queue pair, so the CPU does not need to reason about
    slot availability.

    Args:
        model_runner: A fully constructed :class:`ModelRunner` whose MPK is
                      compiled in ``online_pinned`` mode.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.runtime: OnlinePinnedRuntime = model_runner.runtime
        self.tokenizer_manager = TokenizerManager(model_runner.tokenizer)

        # Monotonically incrementing request id (never wraps).
        self._next_rid: int = 0

        # Serialises ring-buffer writes so two callers do not interleave.
        self._submit_lock = threading.RLock()

        # Background kernel bookkeeping.
        self._kernel_launched: threading.Event = threading.Event()
        self._kernel_thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def submit(
        self,
        prompt: str,
        use_template: bool = True,
        timeout: float = 120.0,
        poll_interval: float = 1e-4,
        stream: bool = False,
    ):
        """Submit a single prompt for generation.

        Safe to call concurrently — each invocation gets a unique rid and
        serialises the ring-buffer write under an internal lock.

        Args:
            prompt:        String prompt.
            use_template:  Apply chat template before tokenizing.
            timeout:       Seconds to wait before raising :exc:`TimeoutError`.
            poll_interval: Seconds between completion-ring polls.
            stream:        If True, returns a generator yielding ``(text,
                           is_final)`` tuples. Otherwise returns a dict.

        Returns:
            When stream=False: ``{"text": str, "token_ids": list[int]}``
            When stream=True:  generator yielding ``(text, is_final)``
        """
        token_ids = self.tokenizer_manager.tokenize(prompt, use_template)
        prompt_len = len(token_ids)

        rid = self._next_rid
        self._next_rid += 1

        t = torch.tensor(token_ids, dtype=torch.int64)
        with self._submit_lock:
            self._ensure_kernel_running()
            self.runtime.submit(rid, t)

        if stream:
            return self._submit_stream(rid, prompt_len, timeout, poll_interval)
        else:
            buffer_row, final_step = self.runtime.wait_for_request(
                rid, timeout, poll_interval)
            full_tokens = self.runtime.read_tokens_at_row(buffer_row, final_step)
            output_ids = full_tokens[prompt_len:].tolist()
            self.runtime.release_request(rid)
            return {
                "text": self.tokenizer_manager.decode(output_ids),
                "token_ids": output_ids,
            }

    # ── Internal ──────────────────────────────────────────────────────────

    def _ensure_kernel_running(self) -> None:
        """Launch the persistent kernel once in a background daemon thread."""
        if self._kernel_launched.is_set():
            return
        with self._submit_lock:
            if self._kernel_launched.is_set():
                return
            self.runtime.reset()
            self._kernel_thread = threading.Thread(
                target=self.model_runner, daemon=True)
            self._kernel_thread.start()
            self._kernel_launched.set()

    def _submit_stream(
        self,
        rid: int,
        prompt_len: int,
        timeout: float,
        poll_interval: float,
    ):
        """Generator: yield ``(text, is_final)`` as tokens are decoded.

        Scans ``pinned_rid_at_row`` to discover the buffer row, then polls
        per-step progress so each new token is yielded immediately.
        """
        import queue

        q: queue.Queue = queue.Queue()
        exc_info: list[BaseException] = []

        def _run():
            try:
                # 1. Discover buffer row by scanning pinned_rid_at_row.
                deadline = time.monotonic() + timeout
                row = -1
                while row == -1:
                    row = self.runtime.find_row_for_rid(rid)
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"stream timed out waiting for row assignment, rid={rid}")
                    import time as _time
                    _time.sleep(poll_interval)

                # 2. Poll per-step progress — yield each new token.
                completed = False
                last_yielded_step = prompt_len - 1
                deadline = time.monotonic() + timeout

                while not completed:
                    # Check if request has finished.
                    newly_done = self.runtime.drain_completions()
                    for done_rid, _, final_step in newly_done:
                        if done_rid == rid:
                            completed = True
                            # Yield remaining tokens up to final_step.
                            current_step = final_step
                            if current_step > last_yielded_step:
                                new_tokens = self.runtime.read_tokens_at_row(
                                    row, current_step)
                                new_ids = new_tokens[
                                    last_yielded_step + 1 : current_step + 1
                                ].tolist()
                                for j, tid in enumerate(new_ids):
                                    text = self.tokenizer_manager.decode_single(tid)
                                    is_final = (completed and
                                                j == len(new_ids) - 1)
                                    q.put((text, is_final))
                                    last_yielded_step += 1
                            else:
                                q.put(("", True))
                            break

                    # Poll step progress for in-flight tokens.
                    current_step = self.runtime.get_current_step_at_row(row)
                    if current_step > last_yielded_step:
                        new_tokens = self.runtime.read_tokens_at_row(
                            row, current_step)
                        new_ids = new_tokens[
                            last_yielded_step + 1 : current_step + 1
                        ].tolist()
                        for tid in new_ids:
                            text = self.tokenizer_manager.decode_single(tid)
                            last_yielded_step += 1
                            q.put((text, False))

                    if time.monotonic() > deadline and not completed:
                        raise TimeoutError(
                            f"stream timed out for rid={rid}")

                    import time as _time
                    _time.sleep(poll_interval)

                self.runtime.release_request(rid)

            except BaseException as e:
                exc_info.append(e)
                q.put(("__error__", True))

        import time
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        def generator():
            while True:
                try:
                    text, is_final = q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if text == "__error__":
                    break
                yield (text, is_final)
                if is_final:
                    break
            if exc_info:
                raise exc_info[0]
            thread.join()

        return generator()

    def close(self) -> None:
        """Signal the GPU kernel to shut down at the next idle cycle."""
        self.runtime.shutdown()
