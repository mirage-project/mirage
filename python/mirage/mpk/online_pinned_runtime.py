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
import threading
import time
from typing import Deque, Dict, List, Tuple

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

        # CPU-private ring cursors.
        self._cpu_req_tail  = 0  # next ring slot to write
        self._cpu_req_ack   = 0  # last slot known to be consumed by GPU
        self._cpu_comp_head = 0

        # CPU-side waiting queue: holds (rid, token_ids, initial_step) tuples
        # that could not be written to the ring because it was full.
        self._waiting: Deque[Tuple[int, torch.Tensor, int]] = collections.deque()
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
        self._abandoned: set[int] = set()
        self._lock = threading.RLock()

        # The drain thread starts only after reset() has initialized all shared
        # state. Starting it here races reset() and can corrupt ring cursors.
        self._drain_stop = threading.Event()
        self._drain_thread: threading.Thread | None = None
        self._drain_error: Exception | None = None

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
        self._raise_drain_error()

        # Keep the producer lock until ready=1 is published. Otherwise a
        # flusher can reserve a later slot and leave a permanent hole in the
        # ordered request ring.
        with self._ring_lock:
            slot = self._cpu_req_tail & self._mask
            if self._load_i32_acquire(self._req_ready, slot) != 0:
                # Ring full — enqueue to CPU-side waiting.
                with self._waiting_lock:
                    self._waiting.append((rid, token_ids.clone(), initial_step))
                return False
            self._publish_request_locked(slot, rid, token_ids, initial_step)
            self._cpu_req_tail += 1
        return True

    def flush_waiting(self) -> int:
        """Move one waiting request from the CPU deque into the ring, if possible.

        Returns the number of requests flushed (0 or 1).
        Called from :meth:`drain_completions` so waiting requests are
        gradually fed into the ring as slots free up.
        """
        with self._ring_lock:
            slot = self._cpu_req_tail & self._mask
            if self._load_i32_acquire(self._req_ready, slot) != 0:
                return 0  # ring still full
            with self._waiting_lock:
                if not self._waiting:
                    return 0
                request = self._waiting.popleft()

            rid, token_ids, initial_step = request
            try:
                self._publish_request_locked(slot, rid, token_ids, initial_step)
            except Exception:
                with self._waiting_lock:
                    self._waiting.appendleft(request)
                raise
            self._cpu_req_tail += 1
        return 1

    def _publish_request_locked(
        self,
        slot: int,
        rid: int,
        token_ids: torch.Tensor,
        initial_step: int,
    ) -> None:
        """Copy and publish one request while ``_ring_lock`` is held."""
        prompt_len = token_ids.shape[0]
        with torch.cuda.stream(self._write_stream):
            self._inbox_tokens[slot, :prompt_len].copy_(
                token_ids, non_blocking=True)
        self._write_stream.synchronize()
        self._req_request_id[slot] = rid
        self._req_prompt_len[slot] = prompt_len
        self._req_initial_step[slot] = initial_step
        self._store_i32_release(self._req_ready, slot, 1)

    def drain_completions(self) -> List[Tuple[int, int, int]]:
        """Non-blocking poll: collect all newly completed requests.

        Returns a list of ``(rid, buffer_row, final_step)`` tuples for
        requests that have finished since the last call.
        """
        self._raise_drain_error()
        finished = []
        while True:
            with self._lock:
                slot = self._cpu_comp_head & self._mask
                if self._load_i32_acquire(self._comp_ready, slot) == 0:
                    break
                rid         = int(self._comp_request_id[slot].item())
                buffer_row  = int(self._comp_buffer_row[slot].item())
                final_step  = int(self._comp_final_step[slot].item())
                if rid in self._abandoned:
                    self._release_row_locked(rid, buffer_row)
                    self._abandoned.remove(rid)
                else:
                    self._completions[rid] = (buffer_row, final_step)
                # Do not make the slot reusable until completion bookkeeping
                # and any abandoned-row release both succeed.
                self._store_i32_release(self._comp_ready, slot, 0)
                self._cpu_comp_head += 1
            finished.append((rid, buffer_row, final_step))

        # Flush all waiting requests while ring slots are free.
        while self.flush_waiting():
            pass
        return finished

    def _drain_loop(self) -> None:
        """Drain completions and flush queued requests in the background."""
        while not self._drain_stop.is_set():
            try:
                self.drain_completions()
            except Exception as exc:
                with self._lock:
                    self._drain_error = exc
                self._drain_stop.set()
                break
            self._drain_stop.wait(0.0002)

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
                self.abandon_request(rid)
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

    def get_completion(self, rid: int) -> Tuple[int, int] | None:
        """Return the completion for *rid*, if the GPU has published it."""
        with self._lock:
            self._raise_drain_error_locked()
            return self._completions.get(rid)

    def release_request(self, rid: int) -> bool:
        """Acknowledge that the CPU finished reading a completed row."""
        with self._lock:
            completion = self._completions.get(rid)
            if completion is None:
                return False
            buffer_row, _ = completion
            self._release_row_locked(rid, buffer_row)
            del self._completions[rid]
            return True

    def abandon_request(self, rid: int) -> None:
        """Release *rid* when it completes without retaining its output."""
        with self._lock:
            completion = self._completions.get(rid)
            if completion is None:
                self._abandoned.add(rid)
                return
            buffer_row, _ = completion
            self._release_row_locked(rid, buffer_row)
            del self._completions[rid]

    def _release_row_locked(self, rid: int, buffer_row: int) -> None:
        owner = self._load_i32_acquire(self._pinned_rid_at_row, buffer_row)
        if owner != rid:
            raise RuntimeError(
                f"row {buffer_row} belongs to rid={owner}, expected rid={rid}"
            )
        # -1 is a phase-disjoint CPU release acknowledgement. The GPU keeps
        # the owner mapping and row stable until it observes this value.
        self._store_i32_release(self._pinned_step, buffer_row, -1)

    def get_current_step_at_row(self, buffer_row: int) -> int:
        """Return the latest decode step written by the GPU for *buffer_row*."""
        return self._load_i32_acquire(self._pinned_step, buffer_row)

    def find_row_for_rid(self, rid: int) -> int:
        """Scan ``pinned_rid_at_row`` to find which buffer row holds *rid*.

        Returns the row index, or -1 if the GPU hasn't assigned a row yet.
        """
        for r in range(self._total_inflight):
            if self._load_i32_acquire(self._pinned_rid_at_row, r) == rid:
                return r
        return -1

    @property
    def waiting_count(self) -> int:
        """Number of requests queued on the CPU side."""
        with self._waiting_lock:
            return len(self._waiting)

    def request_shutdown(self) -> None:
        """Signal the GPU persistent kernel to terminate when it is idle."""
        self._store_i32_release(self._shutdown, 0, 1)

    def stop(self) -> None:
        """Stop the completion drainer after the GPU kernel has exited."""
        self._drain_stop.set()
        if self._drain_thread is not None:
            self._drain_thread.join()

    def shutdown(self) -> None:
        """Signal the GPU kernel and stop the completion drainer."""
        self.request_shutdown()
        self.stop()

    def start(self) -> None:
        """Start the background completion drainer after shared state is reset."""
        if self._drain_thread is not None and self._drain_thread.is_alive():
            return
        self._raise_drain_error()
        self._drain_stop.clear()
        self._drain_thread = threading.Thread(
            target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def reset(self) -> None:
        """Clear completion bookkeeping and ring state for a new session."""
        if self._drain_thread is not None and self._drain_thread.is_alive():
            raise RuntimeError("cannot reset a running online runtime")

        with self._ring_lock:
            self._cpu_req_tail = 0
            self._cpu_req_ack = 0
            with self._waiting_lock:
                self._waiting.clear()
            self._req_ready.zero_()
            self._req_request_id.zero_()

        with self._lock:
            self._cpu_comp_head = 0
            self._completions.clear()
            self._abandoned.clear()
            self._drain_error = None
            self._comp_ready.zero_()

        self._shutdown.zero_()
        self._pinned_step.zero_()
        self._pinned_rid_at_row.fill_(-1)

    def _load_i32_acquire(self, tensor: torch.Tensor, index: int) -> int:
        return int(self._mpk.persistent_kernel.load_i32_acquire(
            tensor.data_ptr(), index))

    def _store_i32_release(
        self, tensor: torch.Tensor, index: int, value: int,
    ) -> None:
        self._mpk.persistent_kernel.store_i32_release(
            tensor.data_ptr(), index, value)

    def _raise_drain_error(self) -> None:
        with self._lock:
            self._raise_drain_error_locked()

    def _raise_drain_error_locked(self) -> None:
        if self._drain_error is not None:
            raise RuntimeError("completion drainer failed") from self._drain_error
