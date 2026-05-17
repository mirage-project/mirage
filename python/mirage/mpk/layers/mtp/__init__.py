"""MTP / speculative-decode catalog layers.

Re-exports the catalog modules wrapping :class:`PersistentKernel`'s
MTP and speculative-decode pk methods. Each module owns one (or a small
variant set of) MPK task(s); see each module's docstring for per-variant
task names and tensor contracts.

Variant maps:

* :class:`MTPTokenScatter` -> ``mtp_token_scatter``.
* :class:`MTPFloatScatter` -> ``mtp_float_scatter``.
* :class:`MTPPrepareVerify` -> ``mtp_prepare_verify``.
* :class:`MTPBuildEmbedInput` -> ``mtp_build_embed_input``.
* :class:`SoftmaxGather` -> ``softmax_gather_sm100``.
* :class:`ProbScatter` -> ``prob_scatter_sm100``.
* :class:`ProbExtract` -> ``prob_extract_sm100``.
* :class:`MTPVerify` — ``mode in {"probabilistic", "strict", "target_greedy"}``
  -> tasks ``mtp_verify_probabilistic`` / ``mtp_verify_strict`` /
  ``target_verify_greedy``.
* :class:`MTPAcceptCommit` -> ``mtp_accept_commit``.
* :class:`FindNgram` — ``scope in {"partial", "global"}``
  -> tasks ``find_ngram_partial`` / ``find_ngram_global``.
"""

from .token_scatter import MTPTokenScatter, MTPFloatScatter
from .prepare_verify import MTPPrepareVerify
from .build_embed import MTPBuildEmbedInput
from .softmax_gather import SoftmaxGather
from .prob_ops import ProbScatter, ProbExtract
from .verify import MTPVerify, MTPAcceptCommit
from .find_ngram import FindNgram

__all__ = [
    "MTPTokenScatter",
    "MTPFloatScatter",
    "MTPPrepareVerify",
    "MTPBuildEmbedInput",
    "SoftmaxGather",
    "ProbScatter",
    "ProbExtract",
    "MTPVerify",
    "MTPAcceptCommit",
    "FindNgram",
]
