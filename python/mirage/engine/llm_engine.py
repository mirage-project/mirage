"""LLMEngine — serving loop built on OnlinePinnedRuntime.

Usage::

    runner = ModelRunner(RunnerConfig(model="Qwen/Qwen3-0.6B", ...))
    engine = LLMEngine(runner)
    results = engine.generate(["Hello, world!", "What is 2+2?"])
    for r in results:
        print(r["text"])
"""

from __future__ import annotations

import time
import threading
from time import perf_counter
import torch
from tqdm.auto import tqdm

from .model_runner import ModelRunner
from ..mpk.online_pinned_runtime import OnlinePinnedRuntime


class LLMEngine:
    """Generation loop backed by the ``online_pinned`` persistent kernel.

    Args:
        model_runner: A fully constructed :class:`ModelRunner` whose MPK is
                      compiled in ``online_pinned`` mode.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner
        self.runtime: OnlinePinnedRuntime = model_runner.runtime
        self.tokenizer = model_runner.tokenizer
        self._kernel_thread: threading.Thread | None = None
        self._kernel_started: threading.Event = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompts: list[str],
        use_template: bool = True,
        use_tqdm: bool = True,
        timeout: float = 120.0,
        poll_interval: float = 1e-4,
    ) -> list[dict]:
        """Tokenize, submit, run, and collect completions for a list of prompts.

        Flow:
          1. Tokenize all prompts.
          2. Reset ring-buffer state for a fresh session.
          3. Initialize the kernel for ``len(prompts)`` requests.
          4. Load each prompt's tokens into GPU memory and write its ring entry.
          5. Launch the persistent kernel in a background thread.
          6. Poll the completion ring until all responses arrive.
          7. Join the kernel thread and return decoded results.

        Args:
            prompts:       List of string prompts.
            use_template:  Apply the model's chat template before tokenizing.
            use_tqdm:      Display a per-request progress bar.
            timeout:       Seconds to wait before raising :exc:`TimeoutError`.
            poll_interval: Seconds between completion-ring polls.

        Returns:
            Results in the same order as *prompts*, each a dict::

                {"text": str, "token_ids": list[int]}
        """
        n = len(prompts)
        assert n <= self.model_runner.config.max_num_batched_requests, (
            f"Too many prompts ({n}); max_num_batched_requests="
            f"{self.model_runner.config.max_num_batched_requests}"
        )

        # 1. Tokenize ─────────────────────────────────────────────────────────
        all_token_ids: list[list[int]] = [
            self._tokenize(p, use_template) for p in prompts
        ]

        # 2. Reset ring-buffer state for this session ─────────────────────────
        self.runtime.reset()

        # 3. Init kernel for exactly n requests ───────────────────────────────
        self.model_runner.init(n)

        # 4. Load tokens and submit ring entries (request IDs = 0 .. n-1) ─────
        for rid, token_ids in enumerate(all_token_ids):
            t = torch.tensor(token_ids, dtype=torch.int64)
            self.runtime.load_tokens(rid, t)
            self.runtime.submit_request(rid, prompt_len=len(token_ids))
        # # activate if all requests are pre-loaded
        # self.model_runner()
        # completions = self.runtime.wait_all(n,timeout)
        
        # activate if submit requests incrementally
        # 5. Launch kernel in background thread ───────────────────────────────
        kernel_thread = threading.Thread(target=self.model_runner, daemon=True)
        kernel_thread.start() # same to self.model_runner() but run in background thread

        # 6. Poll completion ring ─────────────────────────────────────────────
        pbar = tqdm(total=n, desc="Generating", dynamic_ncols=True) if use_tqdm else None
        completed: list[int] = []
        t_start = perf_counter()
        try:
            while len(completed) < n:
                newly_done = self.runtime.drain_completions() # collect all newly completed request IDs
                if newly_done:
                    completed.extend(newly_done)
                    if pbar is not None:
                        pbar.update(len(newly_done))
                else:
                    time.sleep(poll_interval)
                if perf_counter() - t_start > timeout:
                    raise TimeoutError(
                        f"generate() timed out: {len(completed)}/{n} requests completed"
                    )
        finally:
            if pbar is not None:
                pbar.close()
            kernel_thread.join(timeout=5.0)

        # 7. Decode and return in original prompt order ───────────────────────
        results = []
        for rid, token_ids in enumerate(all_token_ids):
            prompt_len = len(token_ids)
            full_tokens = self.runtime.get_output_tokens(rid)  # prompt + generated
            output_ids = full_tokens[prompt_len:].tolist()     # generated only
            results.append({
                "text": self.tokenizer.decode(output_ids, skip_special_tokens=True),
                "token_ids": output_ids,
            })
        return results

    def start(self, n: int) -> None:
        """Initialize and launch the persistent kernel for *n* total requests.

        Must be called once before :meth:`generate_incremental`.  The kernel
        runs in a background thread and will exit after processing all *n*
        requests submitted via the ring buffer.

        Args:
            n: Total number of requests that will be submitted across all
               subsequent :meth:`generate_incremental` calls.
        """
        assert n <= self.model_runner.config.max_num_batched_requests, (
            f"n={n} exceeds max_num_batched_requests="
            f"{self.model_runner.config.max_num_batched_requests}"
        )
        self.runtime.reset()
        self.model_runner.init(n)

        self._kernel_started.clear()

        def _kernel_loop():
            self._kernel_started.set()  # signal that kernel is now running
            self.model_runner()

        self._kernel_thread = threading.Thread(target=_kernel_loop, daemon=True)
        self._kernel_thread.start()
        self._kernel_started.wait()  # block until kernel is live before returning

    def generate_incremental(
        self,
        arrivals: list[tuple[str, float]],
        use_template: bool = True,
        timeout: float = 120.0,
        poll_interval: float = 1e-4,
    ) -> list[dict]:
        """Submit prompts incrementally and collect completions.

        Requires :meth:`start` to have been called first.  Prompts are fed
        into the ring buffer at the specified delays so the GPU can start on
        early requests while later ones are still arriving.  Each request is
        decoded as soon as it completes.

        Args:
            arrivals:      List of ``(prompt, delay_seconds)`` pairs.  Delays
                           are relative to the start of this call.
            use_template:  Apply the model's chat template before tokenizing.
            timeout:       Seconds to wait before raising :exc:`TimeoutError`.
            poll_interval: Seconds between completion-ring polls.

        Returns:
            Results in the same order as *arrivals*, each a dict::

                {"text": str, "token_ids": list[int]}
        """
        assert self._kernel_thread is not None and self._kernel_thread.is_alive(), (
            "Kernel is not running — call start() before generate_incremental()"
        )

        n = len(arrivals)
        prompts = [p for p, _ in arrivals]
        delays  = [d for _, d in arrivals]

        # 1. Tokenize all prompts up front ────────────────────────────────────
        all_token_ids: list[list[int]] = [
            self._tokenize(p, use_template) for p in prompts
        ]

        # 2. Submit requests at their scheduled arrival times ─────────────────
        t0 = perf_counter()

        def _submit_loop():
            for rid, (token_ids, delay) in enumerate(zip(all_token_ids, delays)):
                wait = t0 + delay - perf_counter()
                if wait > 0:
                    time.sleep(wait)
                t = torch.tensor(token_ids, dtype=torch.int64)
                self.runtime.load_tokens(rid, t)
                self.runtime.submit_request(rid, prompt_len=len(token_ids))

        submit_thread = threading.Thread(target=_submit_loop, daemon=True)
        submit_thread.start()

        # 3. Poll completion ring — decode each request as soon as it completes
        results: list[dict | None] = [None] * n
        num_completed = 0
        try:
            while num_completed < n:
                newly_done = self.runtime.drain_completions()
                for rid in newly_done:
                    prompt_len = len(all_token_ids[rid])
                    full_tokens = self.runtime.get_output_tokens(rid)
                    output_ids = full_tokens[prompt_len:].tolist()
                    results[rid] = {
                        "text": self.tokenizer.decode(output_ids, skip_special_tokens=True),
                        "token_ids": output_ids,
                    }
                    print(f"\nPrompt: {self.tokenizer.decode(all_token_ids[rid],skip_special_tokens=True)!r}")
                    print(f"Completion: {results[rid]['text']!r}")
                num_completed += len(newly_done)
                if not newly_done:
                    time.sleep(poll_interval)
                if perf_counter() - t0 > timeout:
                    raise TimeoutError(
                        f"generate_incremental() timed out: {num_completed}/{n} completed"
                    )
        finally:
            submit_thread.join()

        return results  # type: ignore[return-value]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _tokenize(self, prompt: str, use_template: bool) -> list[int]:
        if use_template:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        return self.tokenizer([text], return_tensors="pt").input_ids[0].tolist()
