"""Server, engine, and capacity defaults. No CUDA or model imports required."""
from dataclasses import dataclass
from typing import Optional

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REQUEST_TIMEOUT = 120.0


@dataclass(frozen=True)
class EngineConfig:
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    poll_interval: float = 0.002


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    disconnect_poll_interval: float = 0.05


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

    output_dir: Optional[str] = None
    """Directory for compiled kernel artefacts; ``None`` uses a temp dir."""

    use_cutlass_kernel: bool = True
    max_pending_requests: int = 128
    developer_role: str = "system"

    # Startup defaults for per-request sampling in online_pinned mode.
    do_sample: bool = False
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 20
    sampling_seed: int = 42
    # Retained for callers of the upstream RunnerConfig. Only the standalone
    # SM100 graph sampler uses a fixed candidate budget; HTTP sampling does not.
    sampling_topk_max: int = 32

    def sampling_defaults(self):
        if not self.do_sample:
            return {}
        return dict(temperature=self.temperature, top_p=self.top_p,
                    top_k=self.top_k, seed=self.sampling_seed)

    def __post_init__(self):
        import math
        if min(self.max_num_batched_requests, self.max_num_batched_tokens,
               self.max_seq_length, self.max_num_pages, self.page_size,
               self.max_pending_requests) <= 0:
            raise ValueError("capacity limits must be positive")
        if self.pinned_ring_capacity <= 0 or self.pinned_ring_capacity & (self.pinned_ring_capacity - 1):
            raise ValueError("pinned_ring_capacity must be a power of two")
        if self.max_num_pages < self.max_num_batched_requests * math.ceil(self.max_seq_length / self.page_size):
            raise ValueError("KV page pool must cover the configured maximum concurrent sequences")
        if self.developer_role not in ("native", "system", "reject"):
            raise ValueError("invalid developer role adapter")
        if self.do_sample:
            from .sampling import SamplingOptions
            SamplingOptions(**self.sampling_defaults())
            if self.temperature <= 0:
                raise ValueError("do_sample=True requires temperature > 0")
