"""LLMEngine — serving loop built on mirage.engine primitives.

Usage::

    engine = LLMEngine(manager, model_runner, tokenizer)
    engine.add_request("Hello, world!")
    while not engine.is_finished():
        engine.step()
"""

from __future__ import annotations

import torch
from time import perf_counter
from tqdm.auto import tqdm

from .request import RequestMetadata, SamplingParams
from .manager import RequestMetadataManager


class LLMEngine:
    """Streaming LLM serving loop backed by Mirage engine primitives.

    Args:
        manager:      A fully constructed :class:`RequestMetadataManager`
                      whose meta tensors are shared with *model_runner*.
        model_runner: A zero-argument callable that launches exactly one GPU
                      kernel iteration.  It must read from the meta tensors
                      that *manager* writes during :meth:`~RequestMetadataManager.admit`
                      and write to the ``output_tokens`` / ``qo_indptr_buffer`` /
                      ``step`` tensors that
                      :meth:`~RequestMetadataManager.collect_outputs` reads.
        tokenizer:    HuggingFace tokenizer used to encode prompts and decode
                      generated token ids.
    """

    def __init__(
        self,
        manager: RequestMetadataManager,
        model_runner,
    ) -> None:
        self.manager = manager
        self.model_runner = model_runner
        self.tokenizer = model_runner.tokenizer
        # Maps request_id -> step value recorded just before the last kernel launch.
        self._prev_step: dict[int, int] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        use_template:bool=True
    ) -> RequestMetadata:
        """Tokenize *prompt* (if a string) and enqueue a new request.
        overload python/mirage/mpk/mpk.py:323 load_new_request
        Returns the :class:`RequestMetadata` object so the caller can track
        the request by :attr:`~RequestMetadata.request_id` if needed.
        """
        if not self.model_runner.mpk.is_built:
            raise ValueError("Model is not built yet, so tokenizer is not available")
        
        if use_template:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role":"user","content":prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        prompt_token_ids = self.tokenizer([text],return_tensors="pt").input_ids[0]
        req = RequestMetadata(prompt_token_ids, sampling_params)
        self.manager.enqueue(req)
        return req

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """Run one full serving iteration.

        Iteration order (mirrors :class:`RequestMetadataManager` contract):

        1. ``cuda.synchronize()``   — flush any in-flight GPU work from the
                                      previous kernel launch.
        2. ``manager.step()``       — collect output tokens, mark finished
                                      requests as DONE, admit pending requests
                                      into free GPU slots.
        3. ``model_runner()``       — launch the next GPU kernel iteration.

        Returns:
            ``(outputs, num_tokens)`` where *outputs* is a list of
            ``(request_id, output_token_ids)`` pairs for every request that
            completed this step, and *num_tokens* is a signed token count:
            positive = prefill tokens, negative = decode tokens processed.
        """
        torch.cuda.synchronize() # flush previous kernel

        # Bulk-transfer all read-only GPU tensors to CPU in one shot.
        self.manager.refresh_cpu_cache()

        # Snapshot step values before manager.step() may call complete(), which
        # sets token_row=None and makes the tensor unreadable for those requests.
        pre_active = self.manager.get_active_requests()
        step_after_kernel: dict[int, tuple[int, int]] = {
            req.request_id: (
                int(self.manager._step_cpu[req.token_row]),
                req.prompt_len,
            )
            for req in pre_active
            if req.token_row is not None
        }

        # Collect outputs from the previous GPU iteration, release finished
        # slots, and admit new requests — all in one call.
        completed = self.manager.step()

        # Compute the actual tokens processed this iteration per request.
        # delta = current_step - prev_step (i.e. the chunk the kernel consumed).
        # For requests crossing the prefill→decode boundary within one chunk,
        # split the delta into prefill and decode portions accordingly.
        prefill_tokens = decode_tokens = 0
        for rid, (current, prompt_len) in step_after_kernel.items():
            prev = self._prev_step.get(rid, 0)
            delta = current - prev
            if prev < prompt_len:
                prefill_chunk = min(delta, prompt_len - prev)
                prefill_tokens += prefill_chunk
                decode_tokens += delta - prefill_chunk
            else:
                decode_tokens += delta
        # mark decode_tokens to negative to classify prefill and decode
        num_tokens = prefill_tokens if prefill_tokens > 0 else -decode_tokens 

        # Snapshot step values for every request that will be active in the next
        # kernel launch (includes newly admitted requests whose step starts at 0).
        # _step_cpu is still valid here — no new kernel has launched since the refresh.
        self._prev_step = {
            req.request_id: int(self.manager._step_cpu[req.token_row])
            for req in self.manager.get_active_requests()
            if req.token_row is not None
        }

        # Launch the next iteration.
        self.model_runner()

        outputs = [(req.request_id, req.output_token_ids) for req in completed]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        """True when every submitted request has completed."""
        return self.manager.num_waiting == 0 and self.manager.num_active == 0

    def generate_batch(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: "SamplingParams | list[SamplingParams] | None" = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        """Batch-first generation: admit all → single kernel launch → collect.

        Unlike :meth:`generate`, which drives the kernel step-by-step from
        Python, this method:

        1. Enqueues all prompts.
        2. Admits up to ``max_num_batched_requests`` requests at once, honouring
           prefix-cache hits (step[token_row] written by manager.admit()).
        3. Calls :meth:`~ModelRunner.prepare_batch_state` to lay out prefix
           pages in ``paged_kv_indices_buffer`` (no-op without kernel patch P2).
        4. Calls :meth:`~ModelRunner.init` with the exact admitted count so the
           kernel does not waste cycles on empty slots.
        5. Launches the kernel **once**; it runs until every admitted request
           finishes (offline mode).
        6. Synchronizes and collects all outputs.
        7. Repeats for any remaining waiting requests.

        After the kernel patches P1/P2/P3 land, step 3 will additionally
        populate free-page lists and prefix KV pages for true KV reuse.
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating (batch)", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        request_ids: list[int] = []
        for prompt, sp in zip(prompts, sampling_params):
            req = self.add_request(prompt, sp)
            request_ids.append(req.request_id)

        outputs: dict[int, list[int]] = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            # Admit as many waiting requests as slots and token budget allow.
            while self.manager.can_admit():
                self.manager.admit()

            n_active = self.manager.num_active
            if n_active == 0:
                break

            # Lay out paged_kv buffers for prefix pages (requires P1+P2).
            # Without kernel patches this is a lightweight no-op but still
            # writes the correct paged_kv_indptr offsets.
            self.manager.prepare_batch_state()

            # Re-initialize kernel for exactly n_active slots so empty slots
            # are not processed.
            t = perf_counter()
            self.model_runner.init(n_active)

            # Single kernel launch — runs until all n_active requests complete.
            self.model_runner()
            torch.cuda.synchronize()
            elapsed = perf_counter() - t

            # Bulk-copy read-only GPU tensors to CPU once.
            self.manager.refresh_cpu_cache()

            # Collect outputs; every request should be finished after one run.
            completed = self.manager.collect_outputs()
            for req in completed:
                self.manager.complete(req)
                # TODO (P3): read final_paged_kv_indices / final_paged_kv_num_pages
                # tensors written by the kernel on completion, then call:
                #   self.manager.register_completed(req, page_indices)

            # Decode/prefill throughput estimate (all tokens processed in one shot).
            total_tokens = sum(
                req.prompt_len + req.num_output_tokens for req in completed
            )
            if total_tokens > 0 and elapsed > 0:
                decode_throughput = total_tokens / elapsed

            if use_tqdm:
                pbar.set_postfix({"Decode": f"{int(decode_throughput)}tok/s"})
                for req in completed:
                    outputs[req.request_id] = req.output_token_ids
                    pbar.update(1)
            else:
                for req in completed:
                    outputs[req.request_id] = req.output_token_ids

        if use_tqdm:
            pbar.close()

        return [
            {
                "text": self.tokenizer.decode(outputs[rid], skip_special_tokens=True),
                "token_ids": outputs[rid],
            }
            for rid in request_ids
        ]

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        """Encode, schedule, and generate completions for a list of prompts.

        Args:
            prompts:         List of string prompts or pre-tokenized token-id
                             lists.
            sampling_params: A single :class:`SamplingParams` applied to all
                             prompts, or a per-prompt list.  ``None`` uses the
                             dataclass defaults.
            use_tqdm:        Display a progress bar with prefill/decode
                             throughput statistics.

        Returns:
            Results in the same order as *prompts*, each a dict::

                {"text": str, "token_ids": list[int]}
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # Enqueue all prompts and record the assigned request_ids so we can
        # restore the original prompt order at the end.
        request_ids: list[int] = []
        for prompt, sp in zip(prompts, sampling_params):
            req = self.add_request(prompt, sp)
            request_ids.append(req.request_id)

        outputs: dict[int, list[int]] = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            step_outputs, num_tokens = self.step()
            elapsed = perf_counter() - t

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / elapsed
                else:
                    decode_throughput = max(-num_tokens, 1) / elapsed
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode":  f"{int(decode_throughput)}tok/s",
                })

            for req_id, token_ids in step_outputs:
                outputs[req_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        if use_tqdm:
            pbar.close()

        # Decode and return in the original prompt order.
        return [
            {
                "text": self.tokenizer.decode(outputs[rid], skip_special_tokens=True),
                "token_ids": outputs[rid],
            }
            for rid in request_ids
        ]
