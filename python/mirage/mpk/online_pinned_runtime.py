"""
OnlinePinnedRuntime — CPU-side driver for MODE_ONLINE_PINNED persistent kernels.

The kernel uses two lock-free power-of-2 ring buffers backed by pinned
(page-locked) memory so both CPU and GPU can access them without DMA copies.

  Request ring  (CPU→GPU): CPU writes {rid, prompt_len, initial_step} then
                            sets ready=1; GPU drains directly into running batch.
  Completion ring (GPU→CPU): GPU writes {rid, buffer_row, final_step} then
                              sets ready=1; CPU polls, collects, clears to 0.

Each ring slot has its own independent pinned inbox buffer, so concurrent
submits can write prompt tokens without overwriting each other.  The GPU
copies inbox tokens to the assigned buffer row when admitting a request.

When the batch and ring are full, requests queue on the CPU side in a
``collections.deque`` and are flushed into the ring as slots free up.

Usage::

    runtime = OnlinePinnedRuntime(mpk)
    runtime.submit(rid=0, token_ids=[...])
    runtime.submit(rid=1, token_ids=[...])
    buffer_row, final_step = runtime.wait_for_request(rid=0, timeout=60)
    tokens = runtime.read_tokens_at_row(buffer_row, final_step)
"""

import collections
import time
import threading
from typing import Deque, Dict, List, Optional, Tuple

import torch


class OnlinePinnedRuntime:
    """CPU-side helper for the online_pinned persistent kernel mode."""

    def __init__(self, mpk):
        assert mpk.metadata.mode == "online_pinned", (
            f"OnlinePinnedRuntime requires mode='online_pinned', got {mpk.metadata.mode}"
        )
        self._mpk = mpk
        self._cap = mpk.pinned_ring_capacity
        self._mask = self._cap - 1
        self._total_inflight = mpk.total_num_requests

        # Pinned CPU↔GPU ring arrays (allocated in mpk.py, shape=[cap])
        self._req_ready         = mpk.pinned_req_ready        # int32, pinned
        self._req_request_id    = mpk.pinned_req_request_id   # int32, pinned
        self._req_prompt_len    = mpk.pinned_req_prompt_len   # int32, pinned
        self._req_initial_step  = mpk.pinned_req_initial_step # int32, pinned
        self._comp_ready        = mpk.pinned_comp_ready       # int32, pinned
        self._comp_request_id   = mpk.pinned_comp_request_id  # int32, pinned
        self._comp_buffer_row   = mpk.pinned_comp_buffer_row  # int32, pinned
        self._comp_final_step   = mpk.pinned_comp_final_step  # int32, pinned
        self._shutdown          = mpk.pinned_shutdown         # int32[1], pinned
        self._pinned_step       = mpk.pinned_step             # int32[max_batched], pinned
        self._inbox_tokens      = mpk.pinned_inbox_tokens     # int64[cap, max_seq_len], pinned
        self._pinned_rid_at_row = mpk.pinned_rid_at_row       # int32[max_batched], pinned

        # ── Constrained decoding (xgrammar) ───────────────────────────────
        # Optional; activated by init_xgrammar() + set_request_grammar().
        # All xgrammar state lives in StructuredGenerationManager (structured.py);
        # this runtime just drives its per-step tick from the drain loop.
        from .structured import StructuredGenerationManager
        self.structured = StructuredGenerationManager(
            token_bitmask=getattr(mpk, "pinned_token_bitmask", None),
            mask_seq=getattr(mpk, "pinned_mask_seq", None),
            constrained_flag=getattr(mpk, "pinned_constrained_flag", None),
            tokens=mpk.tokens,
            pinned_step=self._pinned_step,
            find_row_for_rid=self.find_row_for_rid,
            prompt_lengths=getattr(mpk, "prompt_lengths", None),
            active_rows=self.active_rows,
        )

        # CPU-private ring cursors.
        self._cpu_req_tail  = 0  # next ring slot to write
        self._cpu_req_ack   = 0  # last slot known to be consumed by GPU
        self._cpu_comp_head = 0

        # CPU-side waiting queue: holds (rid, token_ids) tuples that couldn't
        # be written to the ring because it was full.
        self._waiting: Deque[Tuple[int, torch.Tensor]] = collections.deque()
        self._waiting_lock = threading.Lock()

        # Dedicated stream for HtoD / DtoH copies.
        self._write_stream = torch.cuda.Stream(device=mpk.tokens.device)

        # Serialises ring-slot reservation between submit() and flush_waiting().
        # Without this lock the two callers can claim the same slot (they both
        # read _cpu_req_tail outside any shared critical section), creating a
        # hole in the request ring.  The GPU stops draining at the first
        # ready==0 slot, so the hole permanently blocks every entry behind it.
        self._ring_lock = threading.Lock()

        # Completion tracking: rid → (buffer_row, final_step)
        self._completions: Dict[int, Tuple[int, int]] = {}
        self._lock = threading.RLock()

        # Persistent drain thread — ensures completions are drained and waiting
        # requests are flushed even when no streaming threads are alive.
        self._drain_stop = threading.Event()
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    # ── Public API ────────────────────────────────────────────────────────

    def submit(self, rid: int, token_ids: torch.Tensor, initial_step: int = 0) -> bool:
        """Stage prompt tokens and write a request into the CPU→GPU ring.

        Writes tokens to the slot-specific inbox so concurrent requests never
        overwrite each other.  If the ring is full, the request is enqueued
        to the CPU-side waiting deque instead.

        Parameters
        ----------
        rid          : unique request identifier (never repeats)
        token_ids    : 1-D int64 tensor of token IDs (CPU or CUDA)
        initial_step : starting decode step (0 unless prefix-cache is used)

        Returns
        -------
        True if the request was written to the ring, False if enqueued to
        the CPU waiting deque.
        """
        prompt_len = token_ids.shape[0]

        # Atomically claim the next ring slot.  Must be serialised with
        # flush_waiting() so the two paths never grab the same slot.
        with self._ring_lock:
            slot = self._cpu_req_tail & self._mask
            if self._req_ready[slot].item() != 0:
                # Ring full — enqueue to CPU-side waiting.
                with self._waiting_lock:
                    self._waiting.append((rid, token_ids.clone(), initial_step))
                return False
            self._cpu_req_tail += 1  # reserve slot

        # Copy prompt tokens to this slot's inbox.
        with torch.cuda.stream(self._write_stream):
            self._inbox_tokens[slot, :prompt_len].copy_(token_ids, non_blocking=True)
        self._write_stream.synchronize()

        # Publish ring entry.
        self._req_request_id[slot]   = rid
        self._req_prompt_len[slot]   = prompt_len
        self._req_initial_step[slot] = initial_step
        self._req_ready[slot] = 1
        return True

    def flush_waiting(self) -> int:
        """Move one waiting request from the CPU deque into the ring, if possible.

        Returns the number of requests flushed (0 or 1).
        Called from :meth:`drain_completions` so waiting requests are
        gradually fed into the ring as slots free up.
        """
        # Reserve ring slot first (serialised with submit), then pop from
        # the waiting deque.  This ordering prevents a lost request if the
        # ring is full: we never remove a request from waiting unless we
        # have a guaranteed ring slot.
        with self._ring_lock:
            slot = self._cpu_req_tail & self._mask
            if self._req_ready[slot].item() != 0:
                return 0  # ring still full
            self._cpu_req_tail += 1  # reserve slot

        with self._waiting_lock:
            if not self._waiting:
                # Rare: submit() drained the deque between our check and now.
                # Put the reserved slot back.
                with self._ring_lock:
                    self._cpu_req_tail -= 1
                return 0
            rid, token_ids, initial_step = self._waiting.popleft()

        prompt_len = token_ids.shape[0]
        with torch.cuda.stream(self._write_stream):
            self._inbox_tokens[slot, :prompt_len].copy_(token_ids, non_blocking=True)
        self._write_stream.synchronize()
        self._req_request_id[slot]   = rid
        self._req_prompt_len[slot]   = prompt_len
        self._req_initial_step[slot] = initial_step
        self._req_ready[slot] = 1
        return 1

    def drain_completions(self) -> List[Tuple[int, int, int]]:
        """Non-blocking poll: collect all newly completed requests.

        Returns a list of ``(rid, buffer_row, final_step)`` tuples for
        requests that have finished since the last call.
        """
        finished = []
        while True:
            with self._lock:
                slot = self._cpu_comp_head & self._mask
                if self._comp_ready[slot].item() == 0:
                    break
                rid         = int(self._comp_request_id[slot].item())
                buffer_row  = int(self._comp_buffer_row[slot].item())
                final_step  = int(self._comp_final_step[slot].item())
                self._comp_ready[slot] = 0
                self._cpu_comp_head += 1
                self._completions[rid] = (buffer_row, final_step)
            finished.append((rid, buffer_row, final_step))

        # Auto-release each completed request's grammar state, keyed by the
        # completion ring's buffer_row (authoritative: the GPU has already reset
        # pinned_rid_at_row for this row, so a rid-based lookup would miss it and
        # leave the global constrained flag stuck on). Idempotent. This is why
        # release_grammar() is not needed in the normal path.
        for rid, buffer_row, _ in finished:
            self.structured.release(rid, row=buffer_row)

        # Flush all waiting requests while ring slots are free.
        while self.flush_waiting():
            pass
        return finished

    def _drain_loop(self) -> None:
        """Persistent background loop that drains completions and flushes
        waiting requests, ensuring forward progress even when all per-request
        streaming threads have exited."""
        while not self._drain_stop.is_set():
            try:
                self.drain_completions()
                self.structured.tick()
            except Exception:
                pass
            self._drain_stop.wait(0.0002)

    # ── Constrained decoding (xgrammar) — delegates to structured.py ──────

    def init_xgrammar(self, tokenizer, vocab_size: int) -> None:
        """Build the xgrammar compiler + tokenizer info (see
        :meth:`structured.StructuredGenerationManager.init_xgrammar`).

        ``ModelRunner`` calls this automatically when
        ``enable_constrained_decoding=True``; call it yourself only when driving
        ``OnlinePinnedRuntime`` without a ``ModelRunner``."""
        self.structured.init_xgrammar(tokenizer, vocab_size)

    def set_request_grammar(self, rid: int, *, json_schema=None, ebnf=None,
                            regex=None, structural_tag=None,
                            triggered_tags=None, tags_with_separator=None,
                            dispatch=None, model=None,
                            any_whitespace=True) -> None:
        """Attach a grammar to a request (call once, near submit()).

        Pass a raw grammar source (``json_schema`` / ``ebnf`` / ``regex`` /
        ``structural_tag``) or a tool list from
        :func:`~mirage.mpk.create_tools` via ``triggered_tags=`` /
        ``tags_with_separator=`` / ``dispatch=`` (optionally ``model=`` for the
        model's native format). See
        :meth:`structured.StructuredGenerationManager.set_request_grammar`."""
        self.structured.set_request_grammar(
            rid, json_schema=json_schema, ebnf=ebnf, regex=regex,
            structural_tag=structural_tag, triggered_tags=triggered_tags,
            tags_with_separator=tags_with_separator, dispatch=dispatch,
            model=model, any_whitespace=any_whitespace)

    def set_constrained(self, on: bool) -> None:
        """Flip the global runtime constrained-decoding flag."""
        self.structured.set_constrained(on)

    def release_grammar(self, rid: int) -> None:
        """Drop a request's grammar state. Optional in the normal path — the
        runtime auto-releases on completion (see :meth:`drain_completions`); use
        this only to abort a request's grammar early. Idempotent."""
        self.structured.release(rid)

    def wait_for_request(
        self,
        rid: int,
        timeout: float = 60.0,
        poll_interval: float = 1e-4,
    ) -> Tuple[int, int]:
        """Block until a specific *rid* completes.

        Returns ``(buffer_row, final_step)``.
        Raises ``TimeoutError`` if the request does not complete in time.
        """
        deadline = time.monotonic() + timeout
        while True:
            self.drain_completions()
            with self._lock:
                if rid in self._completions:
                    return self._completions[rid]
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"wait_for_request timed out for rid={rid}"
                )
            time.sleep(poll_interval)

    def read_tokens_at_row(self, buffer_row: int, final_step: int) -> torch.Tensor:
        """Read tokens [0..final_step] from *buffer_row*.

        Parameters
        ----------
        buffer_row : row index into the token buffer
        final_step : number of tokens available (prompt + generated) - 1
        """
        with torch.cuda.stream(self._write_stream):
            result = self._mpk.tokens[buffer_row, : final_step + 1].clone()
        self._write_stream.synchronize()
        return result

    def read_tokens_range(self, buffer_row: int, start: int, end: int) -> torch.Tensor:
        """Read tokens [start..end] (inclusive) from *buffer_row*."""
        with torch.cuda.stream(self._write_stream):
            result = self._mpk.tokens[buffer_row, start : end + 1].clone()
        self._write_stream.synchronize()
        return result

    def release_request(self, rid: int) -> None:
        """Remove a completed request from the completion bookkeeping."""
        with self._lock:
            self._completions.pop(rid, None)

    def get_current_step_at_row(self, buffer_row: int) -> int:
        """Return the latest decode step written by the GPU for *buffer_row*."""
        return int(self._pinned_step[buffer_row].item())

    def find_row_for_rid(self, rid: int) -> int:
        """Scan ``pinned_rid_at_row`` to find which buffer row holds *rid*.

        Returns the row index, or -1 if the GPU hasn't assigned a row yet.
        """
        for r in range(self._total_inflight):
            if int(self._pinned_rid_at_row[r].item()) == rid:
                return r
        return -1

    def active_rows(self) -> List[int]:
        """Buffer rows currently holding a live request (rid != -1). Used by the
        constrained-decoding tick to publish a mask for every active row."""
        return [r for r in range(self._total_inflight)
                if int(self._pinned_rid_at_row[r].item()) >= 0]

    @property
    def waiting_count(self) -> int:
        """Number of requests queued on the CPU side."""
        with self._waiting_lock:
            return len(self._waiting)

    def shutdown(self) -> None:
        """Signal the GPU persistent kernel to terminate."""
        self._shutdown[0] = 1
        self._drain_stop.set()

    def reset(self) -> None:
        """Clear completion bookkeeping and ring state for a new session."""
        self._cpu_req_tail    = 0
        self._cpu_req_ack     = 0
        self._cpu_comp_head   = 0
        self._shutdown[0]     = 0
        with self._lock:
            self._completions.clear()
        with self._waiting_lock:
            self._waiting.clear()
        self._comp_ready.zero_()
        self._req_ready.zero_()
        self._req_request_id.zero_()
        self._pinned_rid_at_row.fill_(-1)
        self.structured.reset()
