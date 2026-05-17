"""ParallelConfig — TP/EP/DP topology attached to PersistentKernel.

A vLLM-style frozen dataclass that captures the multi-rank topology of a
distributed MPK run. The world is partitioned into TP groups (each rank
holds a slice of every weight) and EP groups (each rank holds a subset
of MoE experts). Derived properties match the math in
``python/mirage/mpk/models/deepseek_v3/builder.py`` so the legacy builders
and the new catalog use the same per-rank arithmetic.

Layers reach this config via ``current_pk().parallel_config`` inside
``with pk.compile_scope():`` — see :mod:`mirage.mpk.context`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelConfig:
    world_size: int = 1
    rank: int = 0
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size != self.tp_size * self.dp_size:
            raise ValueError(
                f"ParallelConfig: world_size ({self.world_size}) must equal "
                f"tp_size ({self.tp_size}) * dp_size ({self.dp_size})"
            )
        if self.tp_size % self.ep_size != 0:
            raise ValueError(
                f"ParallelConfig: ep_size ({self.ep_size}) must divide "
                f"tp_size ({self.tp_size}) so routed_tp_size is integral"
            )
        if not (0 <= self.rank < self.world_size):
            raise ValueError(
                f"ParallelConfig: rank ({self.rank}) must be in [0, "
                f"{self.world_size})"
            )

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def routed_tp_size(self) -> int:
        # TP within an MoE expert group. ``world_size // ep_size`` matches
        # deepseek_v3/builder.py:182.
        return self.world_size // self.ep_size

    @property
    def routed_tp_rank(self) -> int:
        return self.rank % self.routed_tp_size

    @property
    def ep_rank(self) -> int:
        return self.rank // self.routed_tp_size
