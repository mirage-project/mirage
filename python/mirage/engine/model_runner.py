"""ModelRunner — MPK-backed model runner for mirage.engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from ..mpk.mpk import MPK, MPKMetadata
from ..mpk import OnlinePinnedRuntime
from ..mpk.models.graph_builder import MirageModelConfig


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class RunnerConfig:
    """Configuration for :class:`ModelRunner`.

    All capacity limits are upper bounds; the actual batch size per session is
    determined by the number of requests submitted to the ring buffer.
    """
    model: str
    """HuggingFace model name *or* local model directory."""

    model_path: Optional[str] = None
    """Path to pre-sharded safetensors (required for multi-GPU local loads)."""

    max_num_batched_requests: int = 4
    max_num_batched_tokens: int = 8
    max_seq_length: int = 512
    max_num_pages: int = 16
    page_size: int = 4096

    pinned_ring_capacity: int = 8
    """Power-of-2 capacity for the CPU↔GPU pinned ring buffers."""

    tensor_parallel_size: int = 1
    """Number of GPUs for tensor parallelism (matches ``mpirun -n`` count)."""

    enforce_eager: bool = False
    """Disable CUDA graph capture when ``True`` (useful for debugging)."""

    output_dir: Optional[str] = None
    """Directory for compiled kernel artefacts; ``None`` uses a temp dir."""

    use_cutlass_kernel: bool = True


# ── ModelRunner ───────────────────────────────────────────────────────────────

class ModelRunner:
    """Manages MPK kernel construction and per-session execution.

    Wraps an :class:`~mirage.mpk.MPK` compiled in ``online_pinned`` mode and
    exposes an :class:`~mirage.mpk.OnlinePinnedRuntime` for CPU↔GPU ring-buffer
    communication.  Pass this object to :class:`~mirage.engine.LLMEngine` to
    drive multi-request generation.
    """

    def __init__(
        self,
        config: RunnerConfig,
        rank: Optional[int] = None,
    ) -> None:
        self.config = config

        # ── Distributed init ──────────────────────────────────────────────
        self.rank, self.world_size = self._init_distributed(rank)
        torch.cuda.set_device(self.rank)
        torch.set_default_dtype(torch.bfloat16)

        # ── Meta tensors (shared with kernel) ─────────────────────────────
        self.meta_tensors = self._allocate_meta_tensors(config)

        # ── Build + compile the MPK ───────────────────────────────────────
        mpk_meta = MPKMetadata(
            mode="online_pinned",
            total_num_requests=config.max_num_batched_requests,
            max_seq_length=config.max_seq_length,
            max_num_batched_requests=config.max_num_batched_requests,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_pages=config.max_num_pages,
            page_size=config.page_size,
            pinned_ring_capacity=config.pinned_ring_capacity,
            world_size=self.world_size,
            rank=self.rank,
            weight_from_model=True,
            model_name=config.model,
            model_path=config.model_path,
            model_config=MirageModelConfig(with_lm_head=True),
            use_cutlass_kernel=config.use_cutlass_kernel,
            **self.meta_tensors,
        )
        self.mpk = MPK(mpk_meta)
        self.mpk.build()
        self.runtime = OnlinePinnedRuntime(self.mpk)
        self.tokenizer = self.mpk.tokenizer
        self.mpk.compile(output_dir=config.output_dir)

    # ── Execution ─────────────────────────────────────────────────────────────

    def init(self, n: int) -> None:
        """Initialize the persistent kernel for exactly *n* active requests.

        Call before :meth:`__call__`.  Passing ``n < max_num_batched_requests``
        prevents the kernel from trying to process uninitialized empty slots.
        """
        pk = self.mpk.persistent_kernel
        pk.init_func(
            self.mpk.meta_tensors_ptr,
            self.mpk.profiler_buffer_ptr,
            pk.mpi_rank,
            pk.num_workers,
            pk.num_local_schedulers,
            pk.num_remote_schedulers,
            pk.max_seq_length,
            n,
            pk.eos_token_id,
            pk.allocate_nvshmem_teams,
        )

    def __call__(self) -> None:
        """Launch the MPK persistent kernel.

        Blocks until all requests submitted to the ring buffer have been
        processed.  Intended to run in a background thread so the main thread
        can concurrently submit requests and poll completions via
        :attr:`runtime`.
        """
        self.mpk()

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _init_distributed(
        rank: Optional[int],
    ) -> tuple[int, int]:
        """Helper method for future multi-GPU online serving
        """
        try:
            from mpi4py import MPI  # type: ignore[import-untyped]
            comm = MPI.COMM_WORLD
            world_size = comm.Get_size()
            detected_rank = comm.Get_rank()
        except ImportError:
            world_size = 1
            detected_rank = 0

        effective_rank = rank if rank is not None else detected_rank

        if world_size > 1:
            os.environ.setdefault("RANK", str(effective_rank))
            os.environ.setdefault("WORLD_SIZE", str(world_size))
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")

        return effective_rank, world_size

    @staticmethod
    def _allocate_meta_tensors(config: RunnerConfig) -> dict[str, torch.Tensor]:
        """Allocate the GPU buffers shared between the MPK kernel and the manager."""
        n_req = config.max_num_batched_requests
        n_tok = config.max_num_batched_tokens
        return dict(
            step=torch.zeros(n_req, dtype=torch.int32, device="cuda"),
            tokens=torch.zeros(n_req, config.max_seq_length, dtype=torch.int64, device="cuda"),
            input_tokens=torch.zeros(n_tok, 1, dtype=torch.int64, device="cuda"),
            output_tokens=torch.zeros(n_tok, 1, dtype=torch.int64, device="cuda"),
            num_new_tokens=torch.ones(n_req, dtype=torch.int32, device="cuda"),
            prompt_lengths=torch.zeros(n_req, dtype=torch.int32, device="cuda"),
            qo_indptr_buffer=torch.zeros(n_req + 1, dtype=torch.int32, device="cuda"),
            paged_kv_indptr_buffer=torch.zeros(n_req + 1, dtype=torch.int32, device="cuda"),
            paged_kv_indices_buffer=torch.zeros(config.max_num_pages, dtype=torch.int32, device="cuda"),
            paged_kv_last_page_len_buffer=torch.zeros(n_req, dtype=torch.int32, device="cuda"),
        )
