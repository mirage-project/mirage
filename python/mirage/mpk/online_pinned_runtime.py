"""
OnlinePinnedRuntime — CPU-side driver for MODE_ONLINE_PINNED persistent kernels.

The kernel uses two lock-free power-of-2 ring buffers backed by pinned
(page-locked) memory so both CPU and GPU can access them without DMA copies.

  Request ring  (CPU→GPU): CPU writes metadata then sets ready=1 (release);
                            GPU polls with ld.acquire.sys, clears to 0 when done.
  Completion ring (GPU→CPU): GPU writes fields then sets ready=1 (st.release.sys);
                              CPU polls with acquire, collects, clears to 0.

The GPU-private cursors (gpu_req_head, gpu_comp_tail) live in regular GPU memory
and are never touched by the CPU.  The CPU maintains its own counters
(cpu_req_tail, cpu_comp_head) locally in this class.

Usage::

    runtime = OnlinePinnedRuntime(mpk)
    # load tokens into mpk.tokens[request_id, :prompt_len] before submitting
    runtime.submit_request(request_id=0, prompt_len=32)
    runtime.submit_request(request_id=1, prompt_len=64)
    runtime.wait_all(num_requests=2)
    tokens = runtime.get_output_tokens(request_id=0)
"""

import time
import threading
from typing import Dict, List, Optional

import torch


class OnlinePinnedRuntime:
    """CPU-side helper for the online_pinned persistent kernel mode."""

    def __init__(self, mpk):
        """
        Parameters
        ----------
        mpk : MPK
            Fully initialised MPK object whose mode is 'online_pinned'.
        """
        assert mpk.metadata.mode == "online_pinned", (
            f"OnlinePinnedRuntime requires mode='online_pinned', got {mpk.metadata.mode}"
        )
        self._mpk = mpk
        self._cap = mpk.pinned_ring_capacity
        self._mask = self._cap - 1

        # Pinned CPU↔GPU ring arrays (allocated in mpk.py, shape=[cap])
        self._req_ready        = mpk.pinned_req_ready        # int32, pinned
        self._req_request_id   = mpk.pinned_req_request_id   # int32, pinned
        self._req_prompt_len   = mpk.pinned_req_prompt_len   # int32, pinned
        self._req_initial_step = mpk.pinned_req_initial_step # int32, pinned
        self._comp_ready       = mpk.pinned_comp_ready       # int32, pinned
        self._comp_request_id  = mpk.pinned_comp_request_id  # int32, pinned
        self._comp_final_step  = mpk.pinned_comp_final_step  # int32, pinned
        self._shutdown         = mpk.pinned_shutdown         # int32[1], pinned

        # CPU-private ring cursors (int, not tensors)
        self._cpu_req_tail  = 0  # next slot to write into the request ring
        self._cpu_comp_head = 0  # next slot to read from the completion ring

        # Dedicated stream for HtoD copies in load_tokens. Using the default
        # stream would deadlock: the persistent kernel runs on the default stream
        # and copy_(non_blocking=False) would synchronize it, blocking until the
        # kernel finishes — which never happens while waiting for more requests.
        self._write_stream = torch.cuda.Stream(device=mpk.tokens.device)

        # Completion tracking: request_id → final_step
        self._completions: Dict[int, int] = {}
        self._lock = threading.Lock() # protect self._completions from having concurrent threads access

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_tokens(
        self,
        request_id: int,
        input_ids: torch.Tensor,
    ) -> None:
        """Copy prompt token IDs into the GPU token buffer for this request.

        Must be called before ``submit_request`` so the GPU sees the tokens
        when it processes the ring entry.

        Parameters
        ----------
        request_id : row index into mpk.tokens
        input_ids  : 1-D int64 tensor of prompt token IDs (CPU or CUDA)
        """
        prompt_len = input_ids.shape[0]
        with torch.cuda.stream(self._write_stream):
            self._mpk.tokens[request_id, :prompt_len].copy_(input_ids, non_blocking=True)
        self._write_stream.synchronize()

    def submit_request(
        self,
        request_id: int,
        prompt_len: int,
        initial_step: int = 0,
    ) -> None:
        """Write one request entry into the CPU→GPU ring.

        Call ``load_tokens`` first to ensure prompt tokens are in GPU memory.

        Parameters
        ----------
        request_id    : index into the token/step/prompt_length arrays
        prompt_len    : number of prompt tokens for this request
        initial_step  : starting decode step (0 unless prefix-cache is used)
        """
        slot = self._cpu_req_tail & self._mask
        # Write data fields first, then the ready flag (release semantics are
        # handled by PyTorch's CPU memory model; the GPU side uses
        # ld.acquire.sys to see these writes before the flag).
        self._req_request_id[slot]   = request_id
        self._req_prompt_len[slot]   = prompt_len
        self._req_initial_step[slot] = initial_step
        # Memory barrier: ensure data is visible before setting ready=1.
        # torch has no explicit fence API, but Python assignment on pinned
        # memory is ordered on x86/ARM; use a full memory fence via ctypes if
        # needed on non-x86 platforms.  For now, the store order is sufficient
        # on x86 because stores are not reordered with prior stores.
        self._req_ready[slot] = 1
        self._cpu_req_tail += 1

    def drain_completions(self) -> List[int]:
        """Non-blocking poll: collect all newly completed request IDs.

        Returns a list of request IDs that have finished since the last call.
        """
        finished = []
        while True:
            slot = self._cpu_comp_head & self._mask
            if self._comp_ready[slot].item() == 0:
                break
            request_id  = int(self._comp_request_id[slot].item())
            final_step  = int(self._comp_final_step[slot].item())
            # Clear the slot so the GPU can reuse it.
            self._comp_ready[slot] = 0
            self._cpu_comp_head += 1
            with self._lock:
                self._completions[request_id] = final_step
            finished.append(request_id)
        return finished

    def wait_all(
        self,
        num_requests: int,
        timeout: float = 60.0,
        poll_interval: float = 1e-4,
    ) -> Dict[int, int]:
        """Block until *num_requests* completions have been collected.

        Parameters
        ----------
        num_requests  : total number of requests submitted (and expected back)
        timeout       : seconds before raising TimeoutError
        poll_interval : seconds between polls

        Returns
        -------
        dict mapping request_id → final_step for all completed requests
        """
        deadline = time.monotonic() + timeout
        while True:
            self.drain_completions()
            with self._lock:
                if len(self._completions) >= num_requests:
                    return dict(self._completions)
            if time.monotonic() > deadline:
                with self._lock:
                    done = len(self._completions)
                raise TimeoutError(
                    f"wait_all timed out: {done}/{num_requests} requests completed"
                )
            time.sleep(poll_interval)

    def get_output_tokens(self, request_id: int) -> torch.Tensor:
        """Return the generated token sequence for a completed request.

        Returns a 1-D int64 CPU tensor of length final_step+1 (prompt +
        generated).  Safe to call on the default stream because
        cudaStreamWaitEvent has been removed from the kernel launch path;
        the completion ring entry guarantees tokens are already in DRAM.

        Raises KeyError if the request has not completed yet.
        """
        with self._lock:
            final_step = self._completions[request_id]
        tokens = self._mpk.tokens
        return tokens[request_id, : final_step + 1].clone()

    def shutdown(self) -> None:
        """Signal the GPU persistent kernel to terminate.

        The kernel will exit its spin-wait loop in ``prepare_next_batch``
        after seeing this flag.  Call after all requests have been submitted
        (i.e. after the submit thread finishes) so no in-flight work is lost.
        """
        self._shutdown[0] = 1

    def reset(self) -> None:
        """Clear completion bookkeeping for a new inference session.

        Does *not* reset the ring cursor state — call only between full
        inference sessions (i.e. after the persistent kernel has been
        relaunched via ``mpk.init_per_request()``).
        """
        self._cpu_req_tail  = 0
        self._cpu_comp_head = 0
        self._shutdown[0]   = 0  # clear shutdown flag for next session
        with self._lock:
            self._completions.clear()
