"""LLMEngine — serving loop built on OnlinePinnedRuntime.
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
        TODO: OpenAI server add requests, feel free to modify anything in this function
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
        # activate if all requests are pre-loaded
        self.model_runner()
        completions = self.runtime.wait_all(n,timeout)
        
        # activate if submit requests incrementally
        # # 5. Launch kernel in background thread ───────────────────────────────
        # kernel_thread = threading.Thread(target=self.model_runner, daemon=True)
        # kernel_thread.start() # same to self.model_runner() but run in background thread

        # # 6. Poll completion ring ─────────────────────────────────────────────
        # pbar = tqdm(total=n, desc="Generating", dynamic_ncols=True) if use_tqdm else None
        # completed: list[int] = []
        # t_start = perf_counter()
        # try:
        #     while len(completed) < n:
        #         newly_done = self.runtime.drain_completions() # collect all newly completed request IDs
        #         if newly_done:
        #             completed.extend(newly_done)
        #             if pbar is not None:
        #                 pbar.update(len(newly_done))
        #         else:
        #             time.sleep(poll_interval)
        #         if perf_counter() - t_start > timeout:
        #             raise TimeoutError(
        #                 f"generate() timed out: {len(completed)}/{n} requests completed"
        #             )
        # finally:
        #     if pbar is not None:
        #         pbar.close()
        #     kernel_thread.join(timeout=5.0)

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

    def generate_incremental(
        self,
        arrivals: list[tuple[str, float]],
        use_template: bool = True,
        timeout: float = 120.0,
        poll_interval: float = 1e-4,
    ) -> list[dict]:
        """Initialize the kernel and submit prompts incrementally.

        Handles the full lifecycle: init, kernel launch, and completion
        collection.  Prompts are fed into the ring buffer at the specified
        delays so the GPU can start on early requests while later ones are
        still arriving.  Each request is decoded as soon as it completes.

        The submit thread is started before the kernel so that at least one
        request is in the ring buffer when the persistent kernel begins,
        preventing it from seeing an empty batch on its first
        EVENT_END_OF_TASK_GRAPH and terminating immediately.

        TODO: May be helpful to test streaming in the future

        Args:
            arrivals:      List of ``(prompt, delay_microseconds)`` pairs.  Delays
                           are relative to the start of this call.
            use_template:  Apply the model's chat template before tokenizing.
            timeout:       Seconds to wait before raising :exc:`TimeoutError`.
            poll_interval: Seconds between completion-ring polls.

        Returns:
            Results in the same order as *arrivals*, each a dict::

                {"text": str, "token_ids": list[int]}
        """
        n = len(arrivals)
        assert n <= self.model_runner.config.max_num_batched_requests, (
            f"Too many arrivals ({n}); max_num_batched_requests="
            f"{self.model_runner.config.max_num_batched_requests}"
        )

        prompts = [p for p, _ in arrivals]
        delays  = [d for _, d in arrivals]

        # 1. Init for this session ─────────────────────────────────────────────
        self.runtime.reset()
        self.model_runner.init(n)

        # 2. Tokenize all prompts up front ────────────────────────────────────
        all_token_ids: list[list[int]] = [
            self._tokenize(p, use_template) for p in prompts
        ]

        # 3. Start submit thread first; wait until the first request is in the
        #    ring before launching the kernel. We must guarantee there is at least
        #    one request at GPU side
        t0 = perf_counter()
        debug_t = perf_counter()
        first_submitted = threading.Event()
        submit_exc: list[BaseException] = []  # captures submit thread exceptions

        def _submit_loop():
            try:
                for rid, (token_ids, delay) in enumerate(zip(all_token_ids, delays)):
                    wait = t0 + delay / 1_000_000 - perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    t = torch.tensor(token_ids, dtype=torch.int64)
                    self.runtime.load_tokens(rid, t)
                    self.runtime.submit_request(rid, prompt_len=len(token_ids))
                    #print(f"request {rid} has been submitted!",flush=True) # DEBUG PRINT
                    first_submitted.set()  # signal after first submission
            except Exception as e:
                submit_exc.append(e)
                first_submitted.set()  # unblock main thread so it doesn't hang

        submit_thread = threading.Thread(target=_submit_loop, daemon=True)
        submit_thread.start()
        first_submitted.wait()  # block until at least one request is in the ring
        if submit_exc:
            raise submit_exc[0]  # surface the exception immediately

        # 4. Launch kernel — ring buffer now has at least one request ──────────
        self._kernel_thread = threading.Thread(target=self.model_runner, daemon=True)
        self._kernel_thread.start()

        # 5. Poll completion ring — decode each request as soon as it completes
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
                if perf_counter() - debug_t > 5.0:
                    debug_t = perf_counter()
                    print(f"step[0:n]={self.model_runner.mpk.step[:n].tolist()}", flush=True)
                    print(f"req_ready={self.runtime._req_ready.tolist()}", flush=True)
                    print(f"comp_ready={self.runtime._comp_ready.tolist()}", flush=True)

        finally:
            submit_thread.join()
            self.runtime.shutdown()  # signal kernel to exit its spin-wait

        return results  # type: ignore[return-value]

    def close(self):
        self.runtime.shutdown() # signal kernel to exit its spin-wait

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
