"""Central configuration hub for MPK.

Five frozen dataclasses, aggregated by :class:`MPKConfig`, that describe
everything a :class:`~mirage.mpk.PersistentKernel` needs to know about a
run. The :meth:`PersistentKernel.build_from_config` factory consumes an
``MPKConfig`` and returns a fully-wired-up PK with the model
instantiated, weights loaded, and the megakernel nvcc-compiled.

Typical usage::

    from mirage.mpk.configs import MPKConfig, HFConfig, ParallelConfig
    from mirage.mpk import PersistentKernel

    cfg = MPKConfig(
        hf_config       = HFConfig.from_pretrained('/raid/.../Qwen3-8B/'),
        parallel_config = ParallelConfig(world_size=2, rank=rank, tp_size=2),
    )
    mpk = PersistentKernel.build_from_config(cfg)
    text = mpk.run('Give me a short intro to LLMs.')
"""

from .hf import HFConfig
from .kv_cache import KVCacheConfig
from .mpk_config import MPKConfig
from .parallel import ParallelConfig
from .runtime import RuntimeConfig
from .spec_decode import (
    LookaheadConfig,
    PromptLookupConfig,
    SpecDecodeConfig,
)

__all__ = [
    "MPKConfig",
    "HFConfig",
    "ParallelConfig",
    "KVCacheConfig",
    "RuntimeConfig",
    "SpecDecodeConfig",
    "LookaheadConfig",
    "PromptLookupConfig",
]
