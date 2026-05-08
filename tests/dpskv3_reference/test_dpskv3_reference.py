"""Smoke tests for the DeepSeek V3 PyTorch reference.

These tests build the model with a TINY config (so it fits in a few
hundred MB), run forward with random init weights and a synthetic
prompt, and check shape + no-NaN. They do NOT verify alignment with
vLLM — that requires loading the actual 671B checkpoint and is the
job of `runner.run_reference()` invoked from a separate harness.

Run via: `pytest tests/dpskv3_reference/test_dpskv3_reference.py -s`
"""

from __future__ import annotations
import torch
import pytest

from .config import Config
from .modeling import DeepseekV3Model


# Tiny config that exercises the architecture without melting GPUs.
TINY = dict(
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    vocab_size=512,
    rms_norm_eps=1e-6,
    kv_lora_rank=64,
    q_lora_rank=128,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    n_group=2,
    topk_group=1,
    first_k_dense_replace=2,           # layers 0,1 dense; 2,3 MoE
    routed_scaling_factor=2.5,
    norm_topk_prob=True,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    hidden_act="silu",
    max_position_embeddings=512,
    rope_theta=10000.0,
    rope_scaling={
        "factor": 4.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "type": "yarn",
        "original_max_position_embeddings": 128,
    },
    num_nextn_predict_layers=1,
)


@pytest.fixture
def cfg() -> Config:
    return Config(**TINY)


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(cfg: Config, *, enable_mtp: bool, device: str) -> DeepseekV3Model:
    model = DeepseekV3Model(
        cfg, layer_indices=list(range(cfg.num_hidden_layers)),
        enable_mtp=enable_mtp,
    )
    model = model.to(device=device, dtype=torch.float32).eval()
    # Initialise weights with small values so the random forward doesn't
    # blow up (default PyTorch init is fine for this).
    return model


def test_forward_no_mtp_shapes(cfg: Config, device: str) -> None:
    """Case #3: no MTP. Verify shapes + no-NaN. Use record_hidden=True
    so we can check per-layer intermediates."""
    model = _build_model(cfg, enable_mtp=False, device=device)
    T = 8
    input_ids = torch.randint(0, cfg.vocab_size, (T,), device=device)
    positions = torch.arange(T, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, positions=positions, record_hidden=True)
    assert "embed" in out
    assert out["embed"].shape == (T, cfg.hidden_size)
    for li in range(cfg.num_hidden_layers):
        assert f"layer_{li}_output" in out
        assert out[f"layer_{li}_output"].shape == (T, cfg.hidden_size)
    assert out["final_norm"].shape == (T, cfg.hidden_size)
    assert out["logits"].shape == (T, cfg.vocab_size)
    assert out["argmax"].shape == (T,)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"NaN/Inf in {k}"


def test_forward_with_mtp_shapes(cfg: Config, device: str) -> None:
    """Cases #1 / #2: MTP forward attached, record_hidden=True."""
    model = _build_model(cfg, enable_mtp=True, device=device)
    T = 8
    input_ids = torch.randint(0, cfg.vocab_size, (T,), device=device)
    positions = torch.arange(T, device=device)
    prev_mtp = torch.cat([input_ids[1:], input_ids[-1:]])
    with torch.no_grad():
        out = model(
            input_ids=input_ids, positions=positions,
            prev_mtp_input_ids=prev_mtp, record_hidden=True,
        )
    assert out["mtp_output"].shape == (T, cfg.hidden_size)
    assert out["mtp_logits"].shape == (T, cfg.vocab_size)
    assert out["mtp_argmax"].shape == (T,)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"NaN/Inf in {k}"


def test_decode_step_shapes(cfg: Config, device: str) -> None:
    """Case decode-step (T=1). record_hidden=False (the production path)."""
    model = _build_model(cfg, enable_mtp=True, device=device)
    input_ids = torch.tensor([5], device=device, dtype=torch.long)
    positions = torch.tensor([10], device=device)
    prev_mtp = torch.tensor([6], device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, positions=positions,
                    prev_mtp_input_ids=prev_mtp, record_hidden=False)
    # record_hidden=False: only argmax + mtp_argmax populated.
    assert out["argmax"].shape == (1,)
    assert out["mtp_argmax"].shape == (1,)
    assert "embed" not in out
    assert "logits" not in out


def test_partial_layers(cfg: Config, device: str) -> None:
    """Build with only layers [0, 3]; record_hidden=True for verification."""
    model = DeepseekV3Model(
        cfg, layer_indices=[0, 3], enable_mtp=False,
    ).to(device=device, dtype=torch.float32).eval()
    T = 4
    input_ids = torch.randint(0, cfg.vocab_size, (T,), device=device)
    positions = torch.arange(T, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, positions=positions, record_hidden=True)
    assert "layer_0_output" in out
    assert "layer_3_output" in out
    assert "layer_1_output" not in out
    assert "layer_2_output" not in out
    assert out["argmax"].shape == (T,)


def test_yarn_rope_finite() -> None:
    """RoPE cache must be finite for the configured max position."""
    from .modeling import DeepseekYarnRotaryEmbedding
    cfg = Config(**TINY)
    rope = DeepseekYarnRotaryEmbedding(cfg)
    assert torch.isfinite(rope.cos_sin_cache).all()
    assert torch.isfinite(torch.tensor(rope.attn_mscale))


def test_softmax_scale_uses_mscale_squared() -> None:
    """Critical: scale must include mscale^2, not mscale.

    See `vllm/model_executor/models/deepseek_v2.py:889,966`.
    """
    from .modeling import DeepseekYarnRotaryEmbedding, DeepseekV2MLAAttention
    from .parallel import ParallelConfig
    cfg = Config(**TINY)
    rope = DeepseekYarnRotaryEmbedding(cfg)
    attn = DeepseekV2MLAAttention(cfg, rope, ParallelConfig())
    expected = (1.0 / (cfg.qk_head_dim ** 0.5)) * (rope.attn_mscale ** 2)
    assert abs(attn.softmax_scale - expected) < 1e-9


def test_eh_proj_concat_form() -> None:
    """MTP eh_proj must be `Linear(concat([enorm,hnorm]))`, not two
    parallel matmuls. See vllm/model_executor/models/deepseek_mtp.py:110.
    """
    from .modeling import DeepseekV3MTPLayer, DeepseekYarnRotaryEmbedding
    from .parallel import ParallelConfig
    cfg = Config(**TINY)
    rope = DeepseekYarnRotaryEmbedding(cfg)
    mtp = DeepseekV3MTPLayer(cfg, rope, ParallelConfig())
    assert mtp.eh_proj.in_features == 2 * cfg.hidden_size
    assert mtp.eh_proj.out_features == cfg.hidden_size


def test_parallel_config_topology() -> None:
    """ParallelConfig topology calculations for TP=4 EP=2."""
    from .parallel import ParallelConfig
    # TP=4 EP=2: 4 ranks, 2 EP groups, routed_tp_size=2
    p0 = ParallelConfig(tp_size=4, ep_size=2, rank=0)
    p1 = ParallelConfig(tp_size=4, ep_size=2, rank=1)
    p2 = ParallelConfig(tp_size=4, ep_size=2, rank=2)
    p3 = ParallelConfig(tp_size=4, ep_size=2, rank=3)
    assert p0.routed_tp_size == 2
    assert p0.ep_rank == 0 and p0.routed_tp_rank == 0
    assert p1.ep_rank == 0 and p1.routed_tp_rank == 1
    assert p2.ep_rank == 1 and p2.routed_tp_rank == 0
    assert p3.ep_rank == 1 and p3.routed_tp_rank == 1
    # 256 experts / ep_size=2 = 128 each
    assert p0.num_local_routed_experts(256) == 128
    assert p0.first_local_routed_expert(256) == 0
    assert p2.first_local_routed_expert(256) == 128


def test_column_parallel_shape() -> None:
    """ColumnParallelLinear weight shape and forward output shape."""
    from .parallel import ColumnParallelLinear, ParallelConfig
    pcfg = ParallelConfig(tp_size=4, ep_size=2, rank=0)
    layer = ColumnParallelLinear(in_features=128, out_features=256, pcfg=pcfg)
    # Weight is [out_per_partition=64, in_features=128]
    assert tuple(layer.weight.shape) == (64, 128)
    x = torch.randn(8, 128)
    y = layer(x)
    assert tuple(y.shape) == (8, 64)


def test_row_parallel_shape() -> None:
    """RowParallelLinear weight shape (no all-reduce in tp_size=1)."""
    from .parallel import RowParallelLinear, ParallelConfig
    # TP=1: no actual sharding
    pcfg = ParallelConfig()
    layer = RowParallelLinear(in_features=128, out_features=256, pcfg=pcfg)
    assert tuple(layer.weight.shape) == (256, 128)
    x = torch.randn(8, 128)
    y = layer(x)
    assert tuple(y.shape) == (8, 256)
