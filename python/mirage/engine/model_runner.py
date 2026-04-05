"""ModelRunner — MPK-backed model runner for mirage.engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from .manager import RequestMetadataManager
from ..mpk.mpk import MPK, MPKMetadata
from ..mpk.models.graph_builder import MirageModelConfig


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class RunnerConfig:
    """Configuration for :class:`ModelRunner`.

    All capacity limits are upper bounds; the actual batch size per step is
    determined at runtime by :class:`~mirage.engine.RequestMetadataManager`.
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

    tensor_parallel_size: int = 1
    """Number of GPUs for tensor parallelism (matches ``mpirun -n`` count)."""

    enforce_eager: bool = False
    """Disable CUDA graph capture when ``True`` (useful for debugging)."""

    output_dir: Optional[str] = None
    """Directory for compiled kernel artefacts; ``None`` uses a temp dir."""

    use_cutlass_kernel: bool = True


# ── ModelRunner ───────────────────────────────────────────────────────────────

class ModelRunner:
    """Manages MPK kernel construction and per-step execution.

    Each :class:`ModelRunner` instance owns the shared *meta tensors* that
    the kernel and :class:`~mirage.engine.RequestMetadataManager` exchange
    data through.  Call :meth:`make_manager` to get a manager already wired
    to those tensors.

    """

    def __init__(
        self,
        config: RunnerConfig,
        rank: Optional[int] = None,
    ) -> None:
        self.config = config

        # ── Distributed init ──────────────────────────────────────────────
        self.rank, self.world_size = self._init_distributed(rank) # deprecated, may substitute later
        torch.cuda.set_device(self.rank) # set device once during modelrunner init
        torch.set_default_dtype(torch.bfloat16)

        # ── Meta tensors (shared with kernel and manager) ─────────────────
        self.meta_tensors = self._allocate_meta_tensors(config)

        # ── Build + compile the MPK ───────────────────────────────────────
        mpk_meta = MPKMetadata(
            mode="offline",
            total_num_requests=config.max_num_batched_requests,
            max_seq_length=config.max_seq_length,
            max_num_batched_requests=config.max_num_batched_requests,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_pages=config.max_num_pages,
            page_size=config.page_size,
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
        self.tokenizer = self.mpk.tokenizer
        self.mpk.compile(output_dir=config.output_dir)

    # ── Factory ───────────────────────────────────────────────────────────────

    def make_manager(self) -> RequestMetadataManager:
        """Return a :class:`RequestMetadataManager` bound to this runner's
        meta tensors.

        The manager's ``tokens`` / ``prompt_lengths`` writes and its
        ``output_tokens`` / ``qo_indptr_buffer`` / ``step`` reads all go
        directly through the tensors the kernel uses.
        """
        mt = self.meta_tensors
        return RequestMetadataManager(
            max_num_batched_requests=self.config.max_num_batched_requests,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_seq_length=self.config.max_seq_length,
            tokens=mt["tokens"],
            prompt_lengths=mt["prompt_lengths"],
            output_tokens=mt["output_tokens"],
            qo_indptr_buffer=mt["qo_indptr_buffer"],
            step=mt["step"],
        )

    # ── Execution ─────────────────────────────────────────────────────────────

    def __call__(self) -> None:
        """Launch one MPK kernel step.

        Called by :class:`~mirage.engine.LLMEngine` after the manager has
        admitted pending requests.  The kernel reads prompt tokens from
        ``meta_tensors["tokens"]`` / ``meta_tensors["prompt_lengths"]``
        (written by :meth:`~RequestMetadataManager.admit`) and writes
        generated tokens to ``meta_tensors["output_tokens"]``,
        ``meta_tensors["qo_indptr_buffer"]``, and ``meta_tensors["step"]``
        (read back by :meth:`~RequestMetadataManager.collect_outputs`).
        """
        self.mpk()

    def warmup(self) -> None:
        """Optional explicit warmup call.

        The MPK compile step already triggers a warmup pass, so this is a
        no-op by default.  Override or extend in a subclass if you need
        additional warm-up iterations.
        """
        torch.cuda.synchronize()

    def exit(self) -> None:
        """Release GPU resources and tear down the process group."""
        del self.mpk
        torch.cuda.synchronize()
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _init_distributed(
        rank: Optional[int],
    ) -> tuple[int, int]:
        """Detect MPI rank / world-size and, if needed, init NCCL.

        All ranks run the same Python process when launched via ``mpirun``.
        There is no master-worker split.
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
