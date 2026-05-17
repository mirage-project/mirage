"""MLA (Multi-head Latent Attention) catalog layers — DeepSeek V3.

Re-exports the catalog modules wrapping :class:`PersistentKernel`'s
MLA pk methods. See each module's docstring for the kernel ``.cuh``
it wraps and tensor contracts.

Q-side RoPE (``deepseek_mla_rope_sm100.cuh``): :class:`MLARopeQSingle`,
:class:`MLARopeQFused`, :class:`MLARopeQSplit`, :class:`MLARopeK`.

Paged KV gather (``mla_kv_cache_gather_sm100.cuh`` /
``mla_kv_cache_gather_split_sm100.cuh``): :class:`MLAKVGatherStandard`,
:class:`MLAKVGatherSplit`, :class:`MLAKVGatherUnified`.

Legacy ``MLARopeQ(variant=...)`` / ``MLAKVGather(variant=...)`` factory
functions are also exported for existing model/test/demo call sites.
"""

from .kv_gather import (
    MLAKVGather,  # legacy factory
    MLAKVGatherStandard,
    MLAKVGatherSplit,
    MLAKVGatherUnified,
)
from .rope import (
    MLARopeQ,  # legacy factory
    MLARopeQSingle,
    MLARopeQFused,
    MLARopeQSplit,
    MLARopeK,
)
from .decode import MLADecode, MLAReduce
from .prefill import (
    MLAPrefillAbsorbed,
    MLAPrefillPlain,
    MLAPrefillUnified,
    MLAPrefillTP8,
    MLAPrefillTP8Chunked,
    MLAPrefillTP8ChunkedSplitK,
    MLAPrefillTP8ChunkedReduce,
)
from .mtp_decode import MLAMtpDecodeTP, MLAMtpReduceTP

__all__ = [
    # MLA KV gather (3 variants + legacy factory)
    "MLAKVGather",
    "MLAKVGatherStandard",
    "MLAKVGatherSplit",
    "MLAKVGatherUnified",
    # MLA RoPE (3 Q variants + K + legacy Q factory)
    "MLARopeQ",
    "MLARopeQSingle",
    "MLARopeQFused",
    "MLARopeQSplit",
    "MLARopeK",
    # MLA attention
    "MLADecode",
    "MLAReduce",
    "MLAPrefillAbsorbed",
    "MLAPrefillPlain",
    "MLAPrefillUnified",
    "MLAPrefillTP8",
    "MLAPrefillTP8Chunked",
    "MLAPrefillTP8ChunkedSplitK",
    "MLAPrefillTP8ChunkedReduce",
    "MLAMtpDecodeTP",
    "MLAMtpReduceTP",
]
