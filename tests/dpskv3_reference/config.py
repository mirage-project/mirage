"""DeepSeek V3 architecture constants.

These values are taken from the published DeepSeek V3 config.json at
https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
and cross-validated against
`vllm/model_executor/models/deepseek_v2.py` (which loads the same
config object).

If you point this reference at a smaller test variant of the model,
override these values via the `Config.from_hf()` factory which reads
the actual config.json — don't edit the defaults here.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json


# DeepSeek V3 production constants — DO NOT EDIT.
# These are the published values for the 671B model.
DEEPSEEK_V3_HIDDEN_SIZE = 7168
DEEPSEEK_V3_INTERMEDIATE_SIZE = 18432       # dense MLP intermediate
DEEPSEEK_V3_MOE_INTERMEDIATE_SIZE = 2048    # MoE expert intermediate
DEEPSEEK_V3_NUM_HIDDEN_LAYERS = 61
DEEPSEEK_V3_NUM_ATTENTION_HEADS = 128
DEEPSEEK_V3_VOCAB_SIZE = 129280
DEEPSEEK_V3_RMS_NORM_EPS = 1e-6

# MLA-specific
DEEPSEEK_V3_KV_LORA_RANK = 512
DEEPSEEK_V3_Q_LORA_RANK = 1536
DEEPSEEK_V3_QK_NOPE_HEAD_DIM = 128
DEEPSEEK_V3_QK_ROPE_HEAD_DIM = 64
DEEPSEEK_V3_V_HEAD_DIM = 128
DEEPSEEK_V3_QK_HEAD_DIM = (
    DEEPSEEK_V3_QK_NOPE_HEAD_DIM + DEEPSEEK_V3_QK_ROPE_HEAD_DIM
)  # = 192

# MoE-specific
DEEPSEEK_V3_NUM_ROUTED_EXPERTS = 256
DEEPSEEK_V3_NUM_SHARED_EXPERTS = 1
DEEPSEEK_V3_NUM_EXPERTS_PER_TOK = 8         # topk
DEEPSEEK_V3_N_GROUP = 8                     # number of expert groups
DEEPSEEK_V3_TOPK_GROUP = 4                  # number of groups to pick from
DEEPSEEK_V3_FIRST_K_DENSE_REPLACE = 3       # layers 0..2 are dense, 3..60 are MoE
DEEPSEEK_V3_ROUTED_SCALING_FACTOR = 2.5     # multiplied into routed weights post-topk
DEEPSEEK_V3_NORM_TOPK_PROB = True           # renormalize topk probs to sum to 1
DEEPSEEK_V3_SCORING_FUNC = "sigmoid"        # gate output activation
DEEPSEEK_V3_TOPK_METHOD = "noaux_tc"        # uses gate.e_score_correction_bias
DEEPSEEK_V3_HIDDEN_ACT = "silu"             # gate/up activation in MLP

# RoPE / YaRN
DEEPSEEK_V3_MAX_POSITION_EMBEDDINGS = 163840
DEEPSEEK_V3_ROPE_THETA = 10000.0
DEEPSEEK_V3_ROPE_SCALING = {
    "factor": 40.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "type": "yarn",
    "original_max_position_embeddings": 4096,
}

# MTP
DEEPSEEK_V3_NUM_NEXTN_PREDICT_LAYERS = 1    # MTP "depth" — 1 MTP layer in DeepSeek V3


@dataclass
class Config:
    """Subset of HF config that the reference actually reads.

    Construct from a checkpoint's `config.json` via `Config.from_hf()`,
    or override fields manually for tests.
    """

    hidden_size: int = DEEPSEEK_V3_HIDDEN_SIZE
    intermediate_size: int = DEEPSEEK_V3_INTERMEDIATE_SIZE
    moe_intermediate_size: int = DEEPSEEK_V3_MOE_INTERMEDIATE_SIZE
    num_hidden_layers: int = DEEPSEEK_V3_NUM_HIDDEN_LAYERS
    num_attention_heads: int = DEEPSEEK_V3_NUM_ATTENTION_HEADS
    vocab_size: int = DEEPSEEK_V3_VOCAB_SIZE
    rms_norm_eps: float = DEEPSEEK_V3_RMS_NORM_EPS

    kv_lora_rank: int = DEEPSEEK_V3_KV_LORA_RANK
    q_lora_rank: int = DEEPSEEK_V3_Q_LORA_RANK
    qk_nope_head_dim: int = DEEPSEEK_V3_QK_NOPE_HEAD_DIM
    qk_rope_head_dim: int = DEEPSEEK_V3_QK_ROPE_HEAD_DIM
    v_head_dim: int = DEEPSEEK_V3_V_HEAD_DIM

    n_routed_experts: int = DEEPSEEK_V3_NUM_ROUTED_EXPERTS
    n_shared_experts: int = DEEPSEEK_V3_NUM_SHARED_EXPERTS
    num_experts_per_tok: int = DEEPSEEK_V3_NUM_EXPERTS_PER_TOK
    n_group: int = DEEPSEEK_V3_N_GROUP
    topk_group: int = DEEPSEEK_V3_TOPK_GROUP
    first_k_dense_replace: int = DEEPSEEK_V3_FIRST_K_DENSE_REPLACE
    routed_scaling_factor: float = DEEPSEEK_V3_ROUTED_SCALING_FACTOR
    norm_topk_prob: bool = DEEPSEEK_V3_NORM_TOPK_PROB
    scoring_func: str = DEEPSEEK_V3_SCORING_FUNC
    topk_method: str = DEEPSEEK_V3_TOPK_METHOD
    hidden_act: str = DEEPSEEK_V3_HIDDEN_ACT

    max_position_embeddings: int = DEEPSEEK_V3_MAX_POSITION_EMBEDDINGS
    rope_theta: float = DEEPSEEK_V3_ROPE_THETA
    rope_scaling: dict = None  # populated post-init

    num_nextn_predict_layers: int = DEEPSEEK_V3_NUM_NEXTN_PREDICT_LAYERS

    def __post_init__(self):
        if self.rope_scaling is None:
            self.rope_scaling = dict(DEEPSEEK_V3_ROPE_SCALING)

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @classmethod
    def from_hf(cls, model_path: str | Path) -> "Config":
        """Load from a HuggingFace `config.json`."""
        with open(Path(model_path) / "config.json") as f:
            hf_config = json.load(f)
        # Map HF keys to ours; tolerate missing fields by using defaults.
        return cls(
            hidden_size=hf_config.get("hidden_size", DEEPSEEK_V3_HIDDEN_SIZE),
            intermediate_size=hf_config.get(
                "intermediate_size", DEEPSEEK_V3_INTERMEDIATE_SIZE
            ),
            moe_intermediate_size=hf_config.get(
                "moe_intermediate_size", DEEPSEEK_V3_MOE_INTERMEDIATE_SIZE
            ),
            num_hidden_layers=hf_config.get(
                "num_hidden_layers", DEEPSEEK_V3_NUM_HIDDEN_LAYERS
            ),
            num_attention_heads=hf_config.get(
                "num_attention_heads", DEEPSEEK_V3_NUM_ATTENTION_HEADS
            ),
            vocab_size=hf_config.get("vocab_size", DEEPSEEK_V3_VOCAB_SIZE),
            rms_norm_eps=hf_config.get(
                "rms_norm_eps", DEEPSEEK_V3_RMS_NORM_EPS
            ),
            kv_lora_rank=hf_config.get(
                "kv_lora_rank", DEEPSEEK_V3_KV_LORA_RANK
            ),
            q_lora_rank=hf_config.get(
                "q_lora_rank", DEEPSEEK_V3_Q_LORA_RANK
            ),
            qk_nope_head_dim=hf_config.get(
                "qk_nope_head_dim", DEEPSEEK_V3_QK_NOPE_HEAD_DIM
            ),
            qk_rope_head_dim=hf_config.get(
                "qk_rope_head_dim", DEEPSEEK_V3_QK_ROPE_HEAD_DIM
            ),
            v_head_dim=hf_config.get(
                "v_head_dim", DEEPSEEK_V3_V_HEAD_DIM
            ),
            n_routed_experts=hf_config.get(
                "n_routed_experts", DEEPSEEK_V3_NUM_ROUTED_EXPERTS
            ),
            n_shared_experts=hf_config.get(
                "n_shared_experts", DEEPSEEK_V3_NUM_SHARED_EXPERTS
            ),
            num_experts_per_tok=hf_config.get(
                "num_experts_per_tok", DEEPSEEK_V3_NUM_EXPERTS_PER_TOK
            ),
            n_group=hf_config.get("n_group", DEEPSEEK_V3_N_GROUP),
            topk_group=hf_config.get("topk_group", DEEPSEEK_V3_TOPK_GROUP),
            first_k_dense_replace=hf_config.get(
                "first_k_dense_replace", DEEPSEEK_V3_FIRST_K_DENSE_REPLACE
            ),
            routed_scaling_factor=hf_config.get(
                "routed_scaling_factor", DEEPSEEK_V3_ROUTED_SCALING_FACTOR
            ),
            norm_topk_prob=hf_config.get(
                "norm_topk_prob", DEEPSEEK_V3_NORM_TOPK_PROB
            ),
            scoring_func=hf_config.get(
                "scoring_func", DEEPSEEK_V3_SCORING_FUNC
            ),
            topk_method=hf_config.get(
                "topk_method", DEEPSEEK_V3_TOPK_METHOD
            ),
            hidden_act=hf_config.get("hidden_act", DEEPSEEK_V3_HIDDEN_ACT),
            max_position_embeddings=hf_config.get(
                "max_position_embeddings",
                DEEPSEEK_V3_MAX_POSITION_EMBEDDINGS,
            ),
            rope_theta=hf_config.get(
                "rope_theta", DEEPSEEK_V3_ROPE_THETA
            ),
            rope_scaling=hf_config.get(
                "rope_scaling", dict(DEEPSEEK_V3_ROPE_SCALING)
            ),
            num_nextn_predict_layers=hf_config.get(
                "num_nextn_predict_layers",
                DEEPSEEK_V3_NUM_NEXTN_PREDICT_LAYERS,
            ),
        )
