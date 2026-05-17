"""MTP / speculative-decode catalog layers.

One module per MPK task (or per task variant). Each module owns exactly
one task name; see each module's docstring for the matching ``.cuh``
under ``include/mirage/persistent_kernel/tasks/{blackwell,speculative_decoding}/``.

Task map:

* :class:`MTPTokenScatter`       -> ``mtp_token_scatter``
* :class:`MTPFloatScatter`       -> ``mtp_float_scatter``
* :class:`MTPPrepareVerify`      -> ``mtp_prepare_verify``
* :class:`MTPBuildEmbedInput`    -> ``mtp_build_embed_input``
* :class:`SoftmaxGather`         -> ``softmax_gather_sm100``
* :class:`ProbScatter`           -> ``prob_scatter_sm100``
* :class:`ProbExtract`           -> ``prob_extract_sm100``
* :class:`MTPVerifyProbabilistic` -> ``mtp_verify_probabilistic``
* :class:`MTPVerifyStrict`       -> ``mtp_verify_strict``
* :class:`MTPVerifyTargetGreedy` -> ``target_verify_greedy``
* :class:`MTPAcceptCommit`       -> ``mtp_accept_commit``
* :class:`FindNgramPartial`      -> ``find_ngram_partial``
* :class:`FindNgramGlobal`       -> ``find_ngram_global``
"""

from .token_scatter import MTPTokenScatter, MTPFloatScatter
from .prepare_verify import MTPPrepareVerify
from .build_embed import MTPBuildEmbedInput
from .softmax_gather import SoftmaxGather
from .prob_ops import ProbScatter, ProbExtract
from .verify import (
    MTPVerifyProbabilistic,
    MTPVerifyStrict,
    MTPVerifyTargetGreedy,
    MTPAcceptCommit,
)
from .find_ngram import FindNgramPartial, FindNgramGlobal

__all__ = [
    "MTPTokenScatter",
    "MTPFloatScatter",
    "MTPPrepareVerify",
    "MTPBuildEmbedInput",
    "SoftmaxGather",
    "ProbScatter",
    "ProbExtract",
    "MTPVerifyProbabilistic",
    "MTPVerifyStrict",
    "MTPVerifyTargetGreedy",
    "MTPAcceptCommit",
    "FindNgramPartial",
    "FindNgramGlobal",
]
