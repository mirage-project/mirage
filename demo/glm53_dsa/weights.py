"""Load one GLM-5.3 (or 5.2) DSA attention layer for shard `rank` of `tp`, in the layouts that the static megakernel tasks read.

The checkpoint stores the projections in e4m3, with one fp32 scale for each 128 x 128 block. The norms and the indexer
weights_proj are bf16. The loader prepares these tensors, as vLLM does (the package vllm/models/deepseek_v32):
  - q_a_proj and kv_a_proj_with_mqa become one replicated FP8 GEMM.
  - The loader dequantizes the indexer wk to bf16 and joins it with weights_proj into one bf16 GEMM.
  - The loader dequantizes kv_b_proj to bf16 and splits it into W_UK and W_UV (bmm in bf16).
  - TP splits q_b_proj, kv_b_proj and o_proj by heads. All other tensors are replicated.
The tensors (NH = 64 / tp heads on this GPU; FP8 weights in the 128 x 128 piece layout of mirage.mpk.fp8_fp32scale_weights;
fp32 scales):

  w_qkva [2688, 6144] uint8, w_qkva_sf [21, 48]   q_a (rows 0..2047), kv_a (2048..2623: kv_c 512, k_pe 64), 64 zero rows
  q_a_norm [2048], kv_a_norm [512] fp32           the RMSNorm weights
  w_qb [NH 256, 2048] uint8, w_qb_sf               q_b rows of the heads on this GPU (for each head: 192 nope, then 64 rope)
  w_uk [NH, 192, 512] bf16, w_uv [NH, 512, 256] bf16
  w_o [6144, NH 256] uint8, w_o_sf                 o_proj columns of the heads on this GPU
  full layers (the indexer):
  w_iqb [4096, 2048] uint8, w_iqb_sf [32, 16]     indexer.wq_b (32 heads x 128)
  w_kw [256, 6144] bf16                            rows 0..127 dequantized wk, 128..159 weights_proj, then zero rows
  k_norm_w, k_norm_b [128] fp32                    the indexer LayerNorm
"""
import json
import os

import torch

from mirage.mpk.fp8_fp32scale_weights import pack_w2

EXPECTED_CONFIG = {"hidden_size": 6144, "num_attention_heads": 64, "q_lora_rank": 2048, "kv_lora_rank": 512,
                   "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256, "index_n_heads": 32,
                   "index_head_dim": 128, "index_topk": 2048, "rope_interleave": True, "indexer_rope_interleave": True}


def download_layers(checkpoint: str, layers, cache_dir: str = None, revision: str = None) -> str:
    """Download a local snapshot with the attention tensors of `layers`. If `checkpoint` is a local directory, return it
    unchanged. First resolve the revision to one commit, so that the index and the shards come from the same commit."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    sha = HfApi().model_info(checkpoint, revision=revision).sha
    print(f"{checkpoint}: revision {revision or 'main'} = commit {sha}")
    index = hf_hub_download(checkpoint, "model.safetensors.index.json", revision=sha, cache_dir=cache_dir)
    with open(index) as f:
        where = json.load(f)["weight_map"]
    prefixes = tuple(f"model.layers.{l}.{m}" for l in layers for m in ("self_attn.", "input_layernorm."))
    shards = sorted({f for name, f in where.items() if name.startswith(prefixes)})
    return snapshot_download(checkpoint, allow_patterns=["config.json", "model.safetensors.index.json"] + shards,
                             revision=sha, cache_dir=cache_dir)


def check_config(snapshot: str) -> dict:
    """Return the config of the snapshot. Raise ValueError if a shape or an option differs from what the kernels compute
    (EXPECTED_CONFIG, the default RoPE type, rope_theta and rms_norm_eps)."""
    with open(os.path.join(snapshot, "config.json")) as f:
        cfg = json.load(f)
    bad = {k: (cfg.get(k), v) for k, v in EXPECTED_CONFIG.items() if cfg.get(k) != v}
    if "rope_theta" not in cfg.get("rope_parameters", {}) or "rms_norm_eps" not in cfg:
        bad["rope_theta / rms_norm_eps"] = "missing"
    if cfg.get("rope_parameters", {}).get("rope_type", "default") != "default":
        bad["rope_type"] = cfg["rope_parameters"]["rope_type"]
    if bad:
        raise ValueError(f"{snapshot}: config differs from what the kernels compute (got, expected): {bad}")
    return cfg


def is_full_layer(cfg: dict, layer: int) -> bool:
    """Return True if the layer runs its own indexer ("full"). A "shared" layer uses the indices of the last full layer.
    The HF config field indexer_types gives the layer type. vLLM gets the same pattern from index_topk_freq and
    index_topk_pattern."""
    return cfg["indexer_types"][layer] == "full"


def checkpoint_reader(snapshot: str):
    """Return read(name) -> the tensor `name` of the snapshot. read.has(name) is True if the snapshot has the tensor."""
    from safetensors import safe_open
    with open(os.path.join(snapshot, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]

    def read(name):
        with safe_open(os.path.join(snapshot, where[name]), framework="pt") as f:
            return f.get_tensor(name)
    read.has = lambda name: name in where
    return read


def dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """e4m3 [N, K] with fp32 block scales [ceil(N / 128), K / 128] -> fp32."""
    N, K = w.shape
    scale = s.float().repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
    return w.view(torch.float8_e4m3fn).float() * scale


def packed(w: torch.Tensor, s: torch.Tensor):
    """e4m3 [N, K] (N, K multiples of 128) -> (pieces viewed [N, K] uint8, scales [N / 128, K / 128] fp32)."""
    q, sf = pack_w2(w.view(torch.uint8).contiguous(), s.float().contiguous())
    return q.view(w.shape[0], w.shape[1]), sf


def layer_tensors(read, layer: int, tp: int = 8, rank: int = 0, device="cpu", full: bool = None) -> dict:
    """Return the tensors of the attention of `layer` for shard `rank` of `tp`, on `device`, in the layouts of the module
    docstring. full: the layer runs its indexer (None: the checkpoint has indexer.wq_b)."""
    p = f"model.layers.{layer}.self_attn."
    full = read.has(p + "indexer.wq_b.weight") if full is None else full
    get = lambda name: read(p + name).to(device)
    NH = 64 // tp
    w = {"input_norm": read(f"model.layers.{layer}.input_layernorm.weight").float().to(device),
         "q_a_norm": get("q_a_layernorm.weight").float(), "kv_a_norm": get("kv_a_layernorm.weight").float()}
    qa, qa_s = get("q_a_proj.weight"), get("q_a_proj.weight_scale_inv")
    kva, kva_s = get("kv_a_proj_with_mqa.weight"), get("kv_a_proj_with_mqa.weight_scale_inv")
    assert qa.shape == (2048, 6144) and kva.shape == (576, 6144) and kva_s.shape[0] == 5
    zero = torch.zeros(64, 6144, dtype=torch.uint8, device=device)
    w["w_qkva"], w["w_qkva_sf"] = packed(torch.cat([qa.view(torch.uint8), kva.view(torch.uint8), zero]),
                                         torch.cat([qa_s, kva_s]))
    qb, qb_s = get("q_b_proj.weight"), get("q_b_proj.weight_scale_inv")
    rows = slice(rank * NH * 256, (rank + 1) * NH * 256)
    w["w_qb"], w["w_qb_sf"] = packed(qb[rows].contiguous(), qb_s[rank * NH * 2:(rank + 1) * NH * 2].contiguous())
    kvb = dequant(get("kv_b_proj.weight"), get("kv_b_proj.weight_scale_inv")).to(torch.bfloat16).view(64, 448, 512)
    kvb = kvb[rank * NH:(rank + 1) * NH]
    w["w_uk"] = kvb[:, :192].contiguous()                       # [NH, 192, 512]
    w["w_uv"] = kvb[:, 192:].transpose(1, 2).contiguous()       # [NH, 512, 256]
    o, o_s = get("o_proj.weight"), get("o_proj.weight_scale_inv")
    cols = slice(rank * NH * 256, (rank + 1) * NH * 256)
    w["w_o"], w["w_o_sf"] = packed(o[:, cols].contiguous(), o_s[:, rank * NH * 2:(rank + 1) * NH * 2].contiguous())
    if full:
        w["w_iqb"], w["w_iqb_sf"] = packed(get("indexer.wq_b.weight"), get("indexer.wq_b.weight_scale_inv"))
        wk = dequant(get("indexer.wk.weight"), get("indexer.wk.weight_scale_inv")).to(torch.bfloat16)
        wp = get("indexer.weights_proj.weight").to(torch.bfloat16)
        assert wk.shape == (128, 6144) and wp.shape == (32, 6144)
        w["w_kw"] = torch.cat([wk, wp, torch.zeros(96, 6144, dtype=torch.bfloat16, device=device)]).contiguous()
        w["k_norm_w"], w["k_norm_b"] = get("indexer.k_norm.weight").float(), get("indexer.k_norm.bias").float()
    return w
