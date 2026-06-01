from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class DeepSeekV4C4CompressorConfig:
    """DeepSeek V4 Flash Base CSA C4 compressor constants."""

    head_dim: int = 512
    rope_head_dim: int = 64
    kv_score_dim: int = 2048
    c4_page_size: int = 128
    compress_ratio: int = 4
    coff: int = 2

    @property
    def task_params(self) -> Tuple[int, int, int, int]:
        return (
            self.head_dim,
            self.rope_head_dim,
            self.kv_score_dim,
            self.c4_page_size,
        )


class DeepSeekV4C4Compressor:
    """Thin layer-catalog wrapper for the SM100 C4 compressor task.

    This class keeps model-specific defaults and APE packing out of
    ``PersistentKernel``. Once PR #695's MPKModule/current_pk API lands on mpk,
    this can grow the standard ``forward``/``compile`` split without changing
    the low-level task name or CUDA contract.
    """

    def __init__(
        self,
        config: DeepSeekV4C4CompressorConfig = DeepSeekV4C4CompressorConfig(),
        block_dim: Tuple[int, int, int] = (128, 1, 1),
    ) -> None:
        self.config = config
        self.block_dim = block_dim

    def pack_ape(self, raw_ape: torch.Tensor) -> torch.Tensor:
        """Pack official HF APE [4, 1024] into CUDA layout [8, 512]."""

        cfg = self.config
        expected = (cfg.compress_ratio, cfg.coff * cfg.head_dim)
        if tuple(raw_ape.shape) != expected:
            raise ValueError(f"raw_ape must have shape {expected}, got {tuple(raw_ape.shape)}")
        ape = torch.empty(
            2 * cfg.compress_ratio,
            cfg.head_dim,
            dtype=raw_ape.dtype,
            device=raw_ape.device,
        )
        ape[: cfg.compress_ratio] = raw_ape[:, : cfg.head_dim]
        ape[cfg.compress_ratio :] = raw_ape[:, cfg.head_dim :]
        return ape

    def compile(
        self,
        pk,
        *,
        kv_score,
        token_meta,
        state_cache,
        c4_cache,
        ape,
        norm_weight,
        rope_cos_sin,
        grid_dim: Optional[Tuple[int, int, int]] = None,
        block_dim: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Register this layer into an existing PersistentKernel instance."""

        pk.dsv4_c4_compress_layer(
            kv_score=kv_score,
            token_meta=token_meta,
            state_cache=state_cache,
            c4_cache=c4_cache,
            ape=ape,
            norm_weight=norm_weight,
            rope_cos_sin=rope_cos_sin,
            dsv4_params=self.config.task_params,
            grid_dim=grid_dim,
            block_dim=block_dim or self.block_dim,
        )
