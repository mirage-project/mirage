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
        prompt_token_ids = self.tokenizer.encode([text],return_tensors="pt")
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

        # Collect outputs from the previous GPU iteration, release finished
        # slots, and admit new requests — all in one call.
        completed = self.manager.step()

        # Compute a signed token count for throughput reporting.
        active = self.manager.get_active_requests()
        num_prefill = [r for r in active if r.status.name == "PREFILL"]
        if num_prefill:
            num_tokens = sum(r.prompt_len for r in num_prefill)
        else:
            num_tokens = -len(active)  # negative → decode phase

        # Launch the next iteration.
        self.model_runner()

        outputs = [(req.request_id, req.output_token_ids) for req in completed]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        """True when every submitted request has completed."""
        return self.manager.num_waiting == 0 and self.manager.num_active == 0

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
