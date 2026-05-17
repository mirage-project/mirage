"""The MPK layer catalog.

Each module in this package is a ``torch.nn.Module`` wrapping one MPK
kernel task. Both ``forward()`` (PyTorch reference) and ``compile()``
(MPK task registration) are implemented; the model author composes
them like normal PyTorch.

``MPKModule`` (the base class) lives in :mod:`._base`. Concrete layers
live in topic subpackages (``linear/``, ``norm/``, ``attention/``, etc.)
and are re-exported here for ``from mirage.mpk import layers`` users.
"""

from ._base import MPKModule

# Standalone utility modules
from .allreduce import AllReduce
from .tensor_init import TensorInit
from .quantize_fp8 import QuantizeFP8
from .transpose_scale import TransposeScale
from .assemble_q_decode import AssembleQDecode
from .sampling import SamplingSM100

# Elementwise
from .elementwise.identity import Identity
from .elementwise import add

# Embedding
from .embedding.embed import Embed

# Normalization
from .norm.rmsnorm import RMSNorm
from .norm.rmsnorm_linear import RMSNormLinear

# Linear / GEMM
from .linear.linear import Linear
from .linear.linear_with_residual import LinearWithResidual
from .linear.linear_fp8 import LinearFP8, LinearFP8BMM, LinearSplitKFP8SwapAB
from .linear.splitk_linear import SplitKLinear
from .linear.fp8_gemm_dense import FP8GEMMDense
from .linear.fp8_group_gemm import FP8GroupGEMM

# Activation
from .activation.silu_mul import SiluMul, SiluMulLinearWithResidual

# Argmax (single-shot + split-reduce; nvshmem-global for TP)
from .argmax.argmax import Argmax
from .argmax.argmax_partial import ArgmaxPartial
from .argmax.argmax_reduce import ArgmaxReduce
from .argmax.nvshmem_global_argmax import NVShmemGlobalArgmax

# Positional / rotary
from .rotary import RotaryEmbedding

# Attention (plain decode + paged prefill/decode + multi-token extend; MLA / split-KV in follow-ups)
from .attention.attention import Attention
from .attention.paged_attention import PagedAttention
from .attention.single_batch_extend_attention import SingleBatchExtendAttention

# MLA (Multi-head Latent Attention — DeepSeek V3)
from .mla import (
    MLAKVGather,
    MLARopeQ,
    MLARopeK,
    MLADecode,
    MLAReduce,
    MLAPrefill,
    MLAMtpDecodeTP,
    MLAMtpReduceTP,
)

# MoE (Mixture-of-Experts — qwen3 + DeepSeek V3)
from .moe import (
    MoETopkRouting,
    MoEW13,
    MoEW2,
    MoESiluMul,
    MoeMulSumAdd,
    MoEPermute,
    MoEUnpermute,
)

# MTP / speculative-decode (DeepSeek V3 MTP path + prompt-lookup spec-decode)
from .mtp import (
    MTPTokenScatter,
    MTPFloatScatter,
    MTPPrepareVerify,
    MTPBuildEmbedInput,
    SoftmaxGather,
    ProbScatter,
    ProbExtract,
    MTPVerify,
    MTPAcceptCommit,
    FindNgram,
)

__all__ = [
    "MPKModule",
    # standalone utility modules
    "AllReduce",
    "TensorInit",
    "QuantizeFP8",
    "TransposeScale",
    "AssembleQDecode",
    "SamplingSM100",
    # elementwise
    "Identity",
    "add",
    # embedding
    "Embed",
    # norm
    "RMSNorm",
    "RMSNormLinear",
    # linear
    "Linear",
    "LinearWithResidual",
    "LinearFP8",
    "LinearFP8BMM",
    "LinearSplitKFP8SwapAB",
    "SplitKLinear",
    "FP8GEMMDense",
    "FP8GroupGEMM",
    # activation
    "SiluMul",
    "SiluMulLinearWithResidual",
    # argmax
    "Argmax",
    "ArgmaxPartial",
    "ArgmaxReduce",
    "NVShmemGlobalArgmax",
    # rotary
    "RotaryEmbedding",
    # attention
    "Attention",
    "PagedAttention",
    "SingleBatchExtendAttention",
    # MLA
    "MLAKVGather",
    "MLARopeQ",
    "MLARopeK",
    "MLADecode",
    "MLAReduce",
    "MLAPrefill",
    "MLAMtpDecodeTP",
    "MLAMtpReduceTP",
    # MoE
    "MoETopkRouting",
    "MoEW13",
    "MoEW2",
    "MoESiluMul",
    "MoeMulSumAdd",
    "MoEPermute",
    "MoEUnpermute",
    # MTP / speculative-decode
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
