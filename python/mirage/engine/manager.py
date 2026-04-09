"""RequestMetadataManager — Python-side request lifecycle manager.

Ownership contract (must never be violated):
  WRITES to meta_tensors:
    tokens[token_row, :]            on admit()
    prompt_lengths[token_row]       on admit()
    step[token_row]                 on admit() when prefix cache hit (sets initial step
                                    to prefix_len; requires kernel patch P2 to take effect)
    paged_kv_indptr_buffer          in prepare_batch_state() (requires P1+P2)
    paged_kv_indices_buffer         in prepare_batch_state() for prefix pages (requires P1+P2)

  READ-ONLY from meta_tensors:
    output_tokens[start:end]        in collect_outputs()   (written by GPU argmax)
    qo_indptr_buffer[slot]          in collect_outputs()   (written by prepare_next_batch)
    step[token_row]                 in collect_outputs()   (written by prepare_next_batch)

  Never touches (without kernel patches P1+P2):
    input_tokens, paged_kv_*, qo_indptr_buffer (write side)
    — all owned end-to-end by prepare_next_batch() on the GPU.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, TYPE_CHECKING

import torch

from .request import RequestMetadata, RequestStatus, SamplingParams

if TYPE_CHECKING:
    from .prefix_cache import PrefixCache


class RequestMetadataManager:
    """Manages per-request state that lives above the kernel boundary.

    One instance is created per PersistentKernel and kept alive across all
    serving iterations. The serving loop calls methods in this order each
    iteration:

        1. collect_outputs()    — after cuda.synchronize()
        2. complete(req)        — for every finished request
        3. while can_admit(): admit()
        4. <kernel launch>

    Args:
        max_num_batched_requests: MPK_MAX_NUM_BATCHED_REQUESTS (number of GPU slots).
        max_num_batched_tokens:   MPK_MAX_NUM_BATCHED_TOKENS.
        max_seq_length:           MPK_MAX_SEQ_LENGTH.
        tokens:          meta_tensors["tokens"]           [total_requests, max_seq_length] int64
        prompt_lengths:  meta_tensors["prompt_lengths"]   [total_requests] int32
        output_tokens:   meta_tensors["output_tokens"]    [max_batched_tokens] int64  (read-only)
        qo_indptr_buffer:meta_tensors["qo_indptr_buffer"] [max_batched_requests+1] int32  (read-only)
        step:            meta_tensors["step"]             [total_requests] int32  (read-only)
    """

    def __init__(
        self,
        max_num_batched_requests: int,
        max_num_batched_tokens: int,
        max_seq_length: int,
        tokens: torch.Tensor,
        prompt_lengths: torch.Tensor,
        output_tokens: torch.Tensor,
        qo_indptr_buffer: torch.Tensor,
        step: torch.Tensor,
        page_size: int = 0,
        prefix_cache: "Optional[PrefixCache]" = None,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        self.max_num_batched_requests = max_num_batched_requests
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_seq_length = max_seq_length
        self.page_size = page_size

        # Meta tensor references
        self._tokens           = tokens
        self._prompt_lengths   = prompt_lengths
        self._output_tokens    = output_tokens       # read-only
        self._qo_indptr_buffer = qo_indptr_buffer    # read-only
        self._step             = step                # read-only

        # Prefix-cache references (None when prefix caching is disabled)
        self._prefix_cache                  = prefix_cache
        self._paged_kv_indptr_buffer        = paged_kv_indptr_buffer
        self._paged_kv_indices_buffer       = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buffer = paged_kv_last_page_len_buffer

        # CPU-side caches refreshed once per iteration via refresh_cpu_cache().
        # All internal reads should use these instead of the GPU tensors to avoid
        # repeated implicit GPU→CPU transfers from .item() calls.
        self._step_cpu:             torch.Tensor = torch.empty_like(step, device="cpu")
        self._output_tokens_cpu:    torch.Tensor = torch.empty_like(output_tokens, device="cpu")
        self._qo_indptr_buffer_cpu: torch.Tensor = torch.empty_like(qo_indptr_buffer, device="cpu")

        total_rows = tokens.shape[0]

        # records which rows in the meta tensors are currently unused
        # Pool of free row indices into tokens[] / prompt_lengths[] / step[].
        # Each admitted request consumes one row; it is returned on complete().
        self._free_token_rows: deque[int] = deque(range(total_rows))

        # active[slot_id] mirrors the GPU's request_ids[] array.
        # slot_id is the index used by prepare_next_batch to index
        # qo_indptr_buffer and paged_kv_indptr_buffer.
        self._active: list[Optional[RequestMetadata]] = (
            [None] * max_num_batched_requests
        )

        # Requests not yet admitted to a GPU slot.
        self._waiting: deque[RequestMetadata] = deque()

    # ── Queue management ─────────────────────────────────────────────────────

    def enqueue(self, req: RequestMetadata) -> None:
        """Add a new request to the waiting queue."""
        assert req.status == RequestStatus.WAITING
        self._waiting.append(req)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_active(self) -> int:
        return sum(1 for r in self._active if r is not None)

    # ── CPU cache ────────────────────────────────────────────────────────────

    def refresh_cpu_cache(self) -> None:
        """Bulk-copy all read-only GPU tensors to CPU in one transfer each.
        Must be called once after cuda.synchronize() and before any method that
        reads _step_cpu, _output_tokens_cpu, or _qo_indptr_buffer_cpu.
        """
        self._step_cpu.copy_(self._step, non_blocking=False)
        self._output_tokens_cpu.copy_(self._output_tokens, non_blocking=False)
        self._qo_indptr_buffer_cpu.copy_(self._qo_indptr_buffer, non_blocking=False)

    # ── Admission ────────────────────────────────────────────────────────────

    def _free_slot(self) -> Optional[int]:
        for i, r in enumerate(self._active):
            if r is None:
                return i
        return None

    def _current_token_budget(self) -> int:
        """Tokens committed in the next batch, accounting for chunked prefill.

        Mirrors the GPU scheduler's sequential allocation in prepare_next_batch:
        each active request takes min(remaining_prompt_tokens, budget_left) for
        prefill, or 1 for decode.
        """
        total = 0
        for r in self._active:
            if r is None:
                continue
            if r.status == RequestStatus.DECODE:
                chunk = 1
            else:
                remaining = r.prompt_len - int(self._step_cpu[r.token_row])
                chunk = min(max(remaining, 0), self.max_num_batched_tokens - total)
            total += chunk
            if total >= self.max_num_batched_tokens:
                break
        return total

    def can_admit(self) -> bool:
        """True when the next waiting request can fit into an available slot.

        With chunked prefill, a new request only needs at least 1 token slot free
        — the kernel will process the prompt across multiple iterations automatically.
        """
        if not self._waiting:
            return False
        if self._free_slot() is None:
            return False
        if not self._free_token_rows:
            return False
        budget_used = self._current_token_budget()
        return budget_used < self.max_num_batched_tokens

    def admit(self) -> Optional[RequestMetadata]:
        """Admit the head of the waiting queue.

        Writes:
            tokens[token_row, :prompt_len]  <- prompt_token_ids
            prompt_lengths[token_row]       <- prompt_len

        These are the only meta_tensor mutations the manager ever performs.
        Returns the admitted RequestMetadata, or None if can_admit() is False.
        """
        if not self.can_admit():
            return None

        req = self._waiting.popleft()
        slot_id   = self._free_slot() # position in the active batch
        token_row = self._free_token_rows.popleft() # request address in token pool

        req.slot_id   = slot_id
        req.token_row = token_row
        req.status    = RequestStatus.PREFILL

        # Write prompt tokens into the shared tensor so prepare_next_batch
        # can find them at tokens[token_row, step:].
        prompt_len = req.prompt_len
        self._tokens[token_row, :prompt_len] = torch.tensor(
            req.prompt_token_ids, dtype=torch.int64, device=self._tokens.device
        ) # add this request to token pool
        self._prompt_lengths[token_row] = prompt_len # update request length

        # Prefix cache lookup: reuse pre-computed KV pages for a shared prefix.
        # Sets step[token_row] = prefix_len so the kernel skips those tokens.
        # Storing prefix_pages for use in prepare_batch_state().
        # NOTE: Requires kernel patch P2 to actually skip prefix computation.
        if self._prefix_cache is not None and self.page_size > 0:
            prefix_len, prefix_pages = self._prefix_cache.lookup(
                list(req.prompt_token_ids)
            )
            if prefix_len > 0:
                req.prefix_len   = prefix_len
                req.prefix_pages = prefix_pages
                self._step[token_row] = prefix_len

        self._active[slot_id] = req # update request to active batch
        return req

    # ── Output collection ────────────────────────────────────────────────────

    def collect_outputs(self) -> list[RequestMetadata]:
        """Read newly generated tokens from the GPU output buffer.

        Call this AFTER cuda.synchronize() and BEFORE the next kernel launch.
        It does NOT write to any kernel-consumed buffer — it only reads
        output_tokens[], qo_indptr_buffer[], and step[] as produced by the
        most recent GPU iteration.

        Returns:
            List of requests that should be passed to complete() because they
            have satisfied a stop condition.
        """
        completed: list[RequestMetadata] = []

        for slot_id, req in enumerate(self._active):
            if req is None:
                continue

            # qo_indptr_buffer[slot_id : slot_id+2] marks the slice of
            # output_tokens[] that belongs to this slot.
            start = int(self._qo_indptr_buffer_cpu[slot_id])
            end   = int(self._qo_indptr_buffer_cpu[slot_id + 1])
            if start == end:
                # This slot produced no tokens this iteration (e.g. capped by
                # MAX_BATCHED_TOKENS during prefill chunking).
                continue

            # step[token_row] has already been advanced by prepare_next_batch
            # to reflect the tokens processed in the iteration that just ended.
            current_step = int(self._step_cpu[req.token_row])
            prompt_len   = req.prompt_len

            # Step before this iteration, inferred from the slot's token count.
            num_tokens = end - start
            old_step   = current_step - num_tokens

            # Transition PREFILL → DECODE once the full prompt is consumed.
            if req.status == RequestStatus.PREFILL and current_step >= prompt_len:
                req.status = RequestStatus.DECODE

            # GPU mirrors prepare_next_batch logic:
            #   output_tokens[qo_indptr + j] is a decode token iff
            #   old_step + j + 1 >= prompt_len  →  j >= prompt_len - old_step - 1
            # During a pure decode iteration old_step >= prompt_len so
            # first_decode_j = 0 and every output token is collected.
            # During a pure prefill iteration (current_step < prompt_len),
            # first_decode_j >= num_tokens so nothing is collected.
            first_decode_j = max(0, prompt_len - old_step - 1)
            if first_decode_j >= num_tokens:
                continue

            done = False
            # check if this request is completed
            for j in range(first_decode_j, num_tokens):
                token_id = int(self._output_tokens_cpu[start + j])
                req.output_token_ids.append(token_id)
                if self._should_stop(req, token_id, old_step + j + 1):
                    done = True
                    break

            if done:
                completed.append(req)

        return completed

    def _should_stop(
        self,
        req: RequestMetadata,
        last_token: int,
        step: int,
    ) -> bool:
        p = req.sampling_params

        if p.eos_token_id != -1 and last_token == p.eos_token_id:
            req.finish_reason = "eos"
            return True

        if last_token in p.stop_token_ids:
            req.finish_reason = "stop_token"
            return True

        if req.num_output_tokens >= p.max_new_tokens:
            req.finish_reason = "length"
            return True

        if step + 1 >= self.max_seq_length:
            req.finish_reason = "max_seq_length"
            return True

        return False

    # ── Prefix-cache batch state ─────────────────────────────────────────────

    def prepare_batch_state(self) -> None:
        """Write paged_kv layout with prefix pages for all currently active requests.

        Must be called after all requests for the next kernel launch have been
        admitted, and before :meth:`ModelRunner.init`.

        For each active slot the prefix pages (if any) are written to
        ``paged_kv_indices_buffer`` and ``paged_kv_indptr_buffer`` is updated
        with cumulative page offsets.

        NOTE: Only the prefix-page portion of this method takes effect.
        Full KV reuse (including free-page management via the kernel's
        page_queue) requires kernel patches P1 and P2.
        Without those patches this method still correctly initialises
        ``step[token_row]`` (done in admit()) but does not produce functional
        KV reuse.
        """
        if self._paged_kv_indices_buffer is None or self.page_size == 0:
            return

        offset = 0
        for slot_id in range(self.max_num_batched_requests):
            if self._paged_kv_indptr_buffer is not None:
                self._paged_kv_indptr_buffer[slot_id] = offset
            req = self._active[slot_id]
            if req is not None and req.prefix_len > 0 and req.prefix_pages:
                num_prefix_pages = (req.prefix_len + self.page_size - 1) // self.page_size
                pages = req.prefix_pages[:num_prefix_pages]
                for j, pg in enumerate(pages):
                    self._paged_kv_indices_buffer[offset + j] = pg
                offset += len(pages)
        if self._paged_kv_indptr_buffer is not None:
            self._paged_kv_indptr_buffer[self.max_num_batched_requests] = offset

    def register_completed(self, req: RequestMetadata, page_indices: list[int]) -> None:
        """Insert a completed request's KV pages into the prefix cache.

        ``page_indices`` must be the final GPU page allocation for this request,
        read from ``final_paged_kv_indices`` / ``final_paged_kv_num_pages`` tensors
        written by the kernel on completion.

        NOTE: Requires kernel patch P3 for the caller to obtain ``page_indices``.
        """
        if self._prefix_cache is not None and page_indices:
            self._prefix_cache.insert(
                list(req.prompt_token_ids), page_indices
            )

    # ── Completion ───────────────────────────────────────────────────────────

    def complete(self, req: RequestMetadata) -> None:
        """Release a finished request's slot and token_row back to the pool.

        The GPU's prepare_next_batch() will have already freed the KV pages for
        this slot (when it detected EOS or max_seq_length). This call only
        updates the Python-side bookkeeping.

        Call this for every request returned by collect_outputs().
        """
        assert req.slot_id is not None, "Request was never admitted"
        assert req.status != RequestStatus.DONE, "Request already completed"

        req.status = RequestStatus.DONE
        self._active[req.slot_id] = None
        self._free_token_rows.append(req.token_row)

        # Clear slot/row references so misuse is caught early
        req.slot_id   = None
        req.token_row = None

    # ── Convenience ──────────────────────────────────────────────────────────

    def get_active_requests(self) -> list[RequestMetadata]:
        return [r for r in self._active if r is not None]

    def get_waiting_requests(self) -> list[RequestMetadata]:
        return list(self._waiting)

    def step(self) -> list[RequestMetadata]:
        """Run one full Python-side iteration of the serving loop.

        Convenience wrapper that:
          1. Calls collect_outputs() to gather new tokens.
          2. Calls complete() on every finished request.
          3. Admits as many waiting requests as the budget allows.

        Returns the list of completed RequestMetadata objects so the caller
        can decode / return responses.

        The caller is still responsible for:
          - calling cuda.synchronize() BEFORE this method
          - launching the kernel AFTER this method
        """
        completed = self.collect_outputs()
        for req in completed:
            self.complete(req) # free metadata for this request
        while self.can_admit():
            self.admit()
        return completed
