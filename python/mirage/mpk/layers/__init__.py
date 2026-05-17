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
from .quantize_fp8 import QuantizeFP8, QuantizeFP8UE8M0, QuantizeFP8F32Scale
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
from .linear.parallel_linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    RowParallelLinearWithResidual,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from .linear.linear_fp8 import (
    LinearFP8,
    LinearFP8WithResidual,
    LinearFP8SwapAB,
    LinearFP8SwapABWithResidual,
    LinearFP8BMM,
    LinearSplitKFP8SwapAB,
)
from .linear.splitk_linear import SplitKLinear
from .linear.fp8_gemm_dense import FP8GEMMDenseSmallM, FP8GEMMDenseMediumM
from .linear.fp8_group_gemm import (
    FP8GroupGEMMSmallM,
    FP8GroupGEMMLargeM,
    FP8GroupGEMMAuto,
)

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
    # Paged KV gather (3 variants + legacy factory)
    MLAKVGather,
    MLAKVGatherStandard,
    MLAKVGatherSplit,
    MLAKVGatherUnified,
    # Q-side RoPE (3 variants + legacy factory) + K-side RoPE
    MLARopeQ,
    MLARopeQSingle,
    MLARopeQFused,
    MLARopeQSplit,
    MLARopeK,
    # Attention
    MLADecode,
    MLAReduce,
    MLAPrefillAbsorbed,
    MLAPrefillPlain,
    MLAPrefillUnified,
    MLAPrefillTP8,
    MLAPrefillTP8Chunked,
    MLAPrefillTP8ChunkedSplitK,
    MLAPrefillTP8ChunkedReduce,
    MLAMtpDecodeTP,
    MLAMtpReduceTP,
)

# MoE (Mixture-of-Experts — qwen3 + DeepSeek V3)
from .moe import (
    MoETopkSoftmaxRouting,
    MoETopkSigmoidRouting,
    MoETopkRouting,
    MoEW13BF16,
    MoEW13FP8,
    MoEW13,
    MoEW2BF16,
    MoEW2FP8,
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
    MTPVerifyProbabilistic,
    MTPVerifyStrict,
    MTPVerifyTargetGreedy,
    MTPAcceptCommit,
    FindNgramPartial,
    FindNgramGlobal,
)

__all__ = [
    "MPKModule",
    # standalone utility modules
    "AllReduce",
    "TensorInit",
    "QuantizeFP8",
    "QuantizeFP8UE8M0",
    "QuantizeFP8F32Scale",
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
    "ColumnParallelLinear",
    "RowParallelLinear",
    "RowParallelLinearWithResidual",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "LinearFP8",
    "LinearFP8WithResidual",
    "LinearFP8SwapAB",
    "LinearFP8SwapABWithResidual",
    "LinearFP8BMM",
    "LinearSplitKFP8SwapAB",
    "SplitKLinear",
    "FP8GEMMDenseSmallM",
    "FP8GEMMDenseMediumM",
    "FP8GroupGEMMSmallM",
    "FP8GroupGEMMLargeM",
    "FP8GroupGEMMAuto",
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
    "MLAKVGatherStandard",
    "MLAKVGatherSplit",
    "MLAKVGatherUnified",
    "MLARopeQ",
    "MLARopeQSingle",
    "MLARopeQFused",
    "MLARopeQSplit",
    "MLARopeK",
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
    # MoE
    "MoETopkSoftmaxRouting",
    "MoETopkSigmoidRouting",
    "MoETopkRouting",
    "MoEW13BF16",
    "MoEW13FP8",
    "MoEW13",
    "MoEW2BF16",
    "MoEW2FP8",
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
    "MTPVerifyProbabilistic",
    "MTPVerifyStrict",
    "MTPVerifyTargetGreedy",
    "MTPAcceptCommit",
    "FindNgramPartial",
    "FindNgramGlobal",
]
