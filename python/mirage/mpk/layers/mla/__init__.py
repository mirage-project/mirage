"""MLA (Multi-head Latent Attention) catalog layers — DeepSeek V3.

Re-exports the catalog modules wrapping :class:`PersistentKernel`'s
MLA pk methods. Each module owns one (or a small variant set of)
``mla_*_sm100`` MPK task(s); see each module's docstring for
per-variant task names and tensor contracts.

Variant maps:

* :class:`MLAKVGather` — ``variant in {"standard", "split", "unified"}``
  -> tasks ``mla_kv_gather_sm100`` / ``mla_kv_gather_split_sm100`` /
  ``mla_kv_gather_unified_sm100``.
* :class:`MLARopeQ` — ``variant in {"single", "fused", "split"}`` ->
  tasks ``deepseek_mla_rope_q_sm100`` /
  ``deepseek_mla_rope_q_fused_sm100`` /
  ``deepseek_mla_rope_q_split_sm100``.
* :class:`MLARopeK` — standalone, task ``deepseek_mla_rope_k_sm100``.
* :class:`MLADecode` / :class:`MLAReduce` — tasks
  ``mla_decode_sm100`` / ``mla_reduce_sm100``.
* :class:`MLAPrefill` — 7 variants; see module docstring.
* :class:`MLAMtpDecodeTP` / :class:`MLAMtpReduceTP` —
  ``tp_size in {1,2,4,8}`` -> the per-TP decode/reduce task variants.
"""

from .kv_gather import MLAKVGather
from .rope import MLARopeQ, MLARopeK
from .decode import MLADecode, MLAReduce
from .prefill import MLAPrefill
from .mtp_decode import MLAMtpDecodeTP, MLAMtpReduceTP

__all__ = [
    "MLAKVGather",
    "MLARopeQ",
    "MLARopeK",
    "MLADecode",
    "MLAReduce",
    "MLAPrefill",
    "MLAMtpDecodeTP",
    "MLAMtpReduceTP",
]
