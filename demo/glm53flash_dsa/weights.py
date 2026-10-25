"""Load one GLM-5.3-Flash DSA attention layer for shard `rank` of `tp`, in the layouts that the static megakernel tasks read.

Flash's MLA has the shapes of GLM-5.3's (demo/glm53_dsa) but hidden 4096, q_lora 1536, qk_nope 256 and no rope (NoPE: the
cache rows and the query rows are the 512 latent values). Its indexer is the pooled one (SGLang's IndexerKPool): wk and a
compress gate project x to a key and a gate per token, a LayerNorm normalizes the key, and every 4 tokens one pooled key
(softmax(gate + ape) over the 4) goes to the index cache; wq_b (bf16) projects q_c to the 32 indexer heads.

The checkpoint stores the MLA projections in e4m3, with one fp32 scale for each 128 x 128 block; the indexer is bf16. The
tensors (NH = 64 / tp heads on this GPU):

  load_layer
  w_qkva [2048, 4096] e4m3, w_qkva_s [16, 32]    q_a (rows 0..1535), kv_a (1536..2047)
  w_qb [NH 256, 1536] e4m3, w_qb_s                q_b rows of the heads on this GPU
  w_o [4096, NH 256] e4m3, w_o_s                  o_proj columns of the heads on this GPU
  w_uk [NH, 256, 512] bf16, w_uv [NH, 512, 256]   kv_b_proj of the heads on this GPU, dequantized and split
  q_a_norm [1536], kv_a_norm [512] fp32           the RMSNorm weights
  load_indexer (replicated)
  w_kgw [384, 4096] bf16                          rows 0..127 wk, 128..255 the compress gate, 256..287 weights_proj, zeros
  w_iqb [4096, 1536] bf16                         wq_b (32 heads x 128)
  k_norm_w, k_norm_b [128], ape [4, 128] fp32     the key LayerNorm, the pool position bias
packed() turns an e4m3 weight and its block scales into scaled_gemm_tile's piece layout (mirage.mpk.fp8_fp32scale_weights).
"""
import json
import os

import torch
from safetensors import safe_open

from mirage.mpk.fp8_fp32scale_weights import pack_w2

H, QL, KVL, NOPE, DV, HEADS = 4096, 1536, 512, 256, 256, 64
IH, ID, KGW = 32, 128, 384   # indexer heads x dims; [wk | gate | weights_proj] rows (288, padded to 384)


def _reader(model_dir: str, prefix: str, dev):
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]

    def get(name):
        with safe_open(os.path.join(model_dir, where[prefix + name]), framework="pt") as f:
            return f.get_tensor(prefix + name).to(dev)
    return get


def load_layer(model_dir: str, layer: int, tp: int, rank: int, dev) -> dict:
    """The MLA tensors of `layer` for shard `rank` of `tp` on dev: the FP8 projections as e4m3 [N, K] with their fp32 128 x 128
    block scales (the reference reads them; packed() gives the kernel's layout), W_UK / W_UV, the norms."""
    get = _reader(model_dir, f"model.language_model.layers.{layer}.self_attn.", dev)
    NH = HEADS // tp
    heads, sb = slice(rank * NH * DV, (rank + 1) * NH * DV), slice(rank * NH * 2, (rank + 1) * NH * 2)
    qa, kva = get("q_a_proj.weight"), get("kv_a_proj_with_mqa.weight")
    assert qa.shape == (QL, H) and kva.shape == (KVL, H), "not the GLM-5.3-Flash MLA (q_lora 1536, kv_lora 512, no rope)"
    kvb = get("kv_b_proj.weight").to(torch.bfloat16).view(HEADS, NOPE + DV, KVL)[rank * NH:(rank + 1) * NH]
    qkva = torch.cat([qa.view(torch.uint8), kva.view(torch.uint8)]).view(torch.float8_e4m3fn)
    return {"w_qkva": qkva, "w_qkva_s": torch.cat([get("q_a_proj.weight_scale_inv"),
                                                   get("kv_a_proj_with_mqa.weight_scale_inv")]).float(),
            "w_qb": get("q_b_proj.weight")[heads].contiguous(), "w_qb_s": get("q_b_proj.weight_scale_inv")[sb].float().contiguous(),
            "w_o": get("o_proj.weight")[:, heads].contiguous(),
            "w_o_s": get("o_proj.weight_scale_inv")[:, sb].float().contiguous(),
            "w_uk": kvb[:, :NOPE].contiguous(),                    # [NH, 256, 512]
            "w_uv": kvb[:, NOPE:].transpose(1, 2).contiguous(),    # [NH, 512, 256]
            "q_a_norm": get("q_a_layernorm.weight").float(), "kv_a_norm": get("kv_a_layernorm.weight").float()}


def load_indexer(model_dir: str, layer: int, dev) -> dict:
    """The indexer tensors of `layer` (replicated over the TP ranks; bf16 in the checkpoint)."""
    get = _reader(model_dir, f"model.language_model.layers.{layer}.self_attn.indexer.", dev)
    wk, gate, wp = get("wk.weight"), get("index_kpool_compress_gate"), get("weights_proj.weight")
    assert wk.shape == (ID, H) and gate.shape == (ID, H) and wp.shape == (IH, H)
    zero = torch.zeros(KGW - 2 * ID - IH, H, dtype=torch.bfloat16, device=dev)
    return {"w_kgw": torch.cat([wk, gate, wp, zero]).to(torch.bfloat16).contiguous(),
            "w_iqb": get("wq_b.weight").to(torch.bfloat16).contiguous(),               # [4096, 1536]
            "k_norm_w": get("k_norm.weight").float(), "k_norm_b": get("k_norm.bias").float(),
            "ape": get("index_kpool_compress_ape").float().contiguous()}               # [4, 128]


def packed(w: torch.Tensor, s: torch.Tensor):
    """e4m3 [N, K] -> (128 x 128 pieces viewed [N, K] uint8, scales [N / 128, K / 128] fp32): scaled_gemm_tile's weight."""
    q, sf = pack_w2(w.view(torch.uint8).contiguous(), s.float().contiguous())
    return q.view(w.shape[0], w.shape[1]), sf
