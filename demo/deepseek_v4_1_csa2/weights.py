"""DeepSeek-V4.1-Flash: one attention layer's tensors for demo.py, read from the checkpoint.

The checkpoint (a local snapshot of deepseek-ai/DeepSeek-V4.1-Flash: inference/config.json, the safetensors index and shards)
keeps the official names (layers.<l>.attn.*). layer_tensors reads one layer's attention weights for GPU `rank` of attention
TP `tp` and returns them under the graph's tensor names (demo.build_graph):
  the same on every GPU   wq_a, wkv, q_norm, kv_norm, the compressor, the indexer (SGLang's DeepseekV41Indexer uses
                          ReplicatedLinear: every GPU scores with all 32 heads, so the top-k needs no reduction)
  split by heads          wq_b's rows of heads [H r, H (r + 1)) (H = 64 / tp), attn_sink's, wo_a's groups [G r, G (r + 1))
                          (G = 8 / tp), wo_b's K columns [1024 G r, 1024 G (r + 1)) (SGLang: ColumnParallel wq_b and wo_a,
                          RowParallel wo_b, so each GPU's y is a partial sum; the all-reduce that adds them is not in the layer)
e4m3 weights keep their bytes. Their 32 x 32 block scales (e8m0) become gemm_fp8_tile's scale chunks (w8_scale_chunks).
"""
import json
import math
import os

import torch
from safetensors import safe_open
from mirage.mpk.static_megakernel import w8_scale_chunks

HEADS, D, RD, GROUPS, OL = 64, 512, 64, 8, 1024   # n_heads, head_dim, rope_head_dim, o_groups, o_lora_rank


def download_layer(checkpoint: str, layer: int, cache_dir: str = None, revision: str = None) -> str:
    """Return a local snapshot path with the layer's attention tensors (as demo/glm53_moe/weights.py does).

    A local directory is returned as it is. A Hugging Face repo id (deepseek-ai/DeepSeek-V4.1-Flash): resolve the revision
    (default: main) to one commit, then download inference/config.json, the index and the shards that hold layer `layer`'s
    attention tensors only (the whole repo is about 510 GB)."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    sha = HfApi().model_info(checkpoint, revision=revision).sha
    print(f"{checkpoint}: revision {revision or 'main'} = commit {sha}")
    index = hf_hub_download(checkpoint, "model.safetensors.index.json", revision=sha, cache_dir=cache_dir)
    with open(index) as f:
        where = json.load(f)["weight_map"]
    shards = sorted({f for name, f in where.items() if name.startswith(f"layers.{layer}.attn.")})
    if not shards:
        raise ValueError(f"{checkpoint}: no attention tensors of layer {layer}")
    return snapshot_download(checkpoint, allow_patterns=["inference/config.json", "model.safetensors.index.json"] + shards,
                             revision=sha, cache_dir=cache_dir)


def checkpoint_reader(snapshot: str):
    """Return read(name) -> CPU tensor. It opens each shard once (the index's weight_map)."""
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    files = {}

    def read(name: str) -> torch.Tensor:
        f = index[name]
        if f not in files:
            files[f] = safe_open(os.path.join(snapshot, f), "pt")
        return files[f].get_tensor(name)
    return read


def model_config(snapshot: str) -> dict:
    """inference/config.json: the official code's ModelArgs (compress_ratios, the source layers, the RoPE parameters)."""
    return json.load(open(os.path.join(snapshot, "inference", "config.json")))


def variant_of(cfg: dict, layer: int) -> str:
    """The layer's attention variant: swa, reuse, full_r2, full_r1 or reindex."""
    r = cfg["compress_ratios"][layer]
    if r == 0:
        return "swa"
    if layer in cfg["kv_source_layers"]:
        return f"full_r{r}"
    return "reindex" if layer in cfg["index_source_layers"] else "reuse"


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device="cuda") -> torch.Tensor:
    """model.py's precompute_freqs_cis (YaRN when original_seq_len > 0), as [seqlen, dim / 2, 2] fp32 (cos, sin). Computed on
    `device`: the official code builds it on the GPU, and a CPU table differs in the last bits of cos and sin (up to 1.2e-4
    at position 32767 of the SWA table)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen, device=device), freqs)
    return torch.view_as_real(torch.polar(torch.ones_like(freqs), freqs)).contiguous()


def freqs_of(cfg: dict, layer: int, rows: int, device="cuda") -> torch.Tensor:
    """The layer's RoPE table, as model.py's Attention builds it. Compressed layers use YaRN at compress_rope_theta.
    SWA layers use plain RoPE at rope_theta."""
    if cfg["compress_ratios"][layer]:
        return precompute_freqs_cis(RD, rows, cfg["original_seq_len"], cfg["compress_rope_theta"], cfg["rope_factor"],
                                    cfg["beta_fast"], cfg["beta_slow"], device)
    return precompute_freqs_cis(RD, rows, 0, cfg["rope_theta"], cfg["rope_factor"], cfg["beta_fast"], cfg["beta_slow"], device)


def layer_tensors(read, cfg: dict, layer: int, tp: int = 1, rank: int = 0, rows: int = 32768, device="cuda") -> dict:
    """Return layer `layer`'s attention tensors for GPU `rank` of attention TP `tp`, with the RoPE table for positions
    0 .. rows - 1. tp is 1, 2 or 4: sparse_attn needs at least 16 heads per GPU."""
    assert tp in (1, 2, 4) and 0 <= rank < tp
    p, var = f"layers.{layer}.attn.", variant_of(cfg, layer)
    H, G = HEADS // tp, GROUPS // tp
    q0, a0 = H * D * rank, G * OL * rank

    def fp8(name, rows=slice(None), cols=slice(None)):   # e4m3 bytes and the scale chunks of a weight slice
        w, s = read(p + name + ".weight"), read(p + name + ".scale")
        div = lambda sl: slice(None if sl.start is None else sl.start // 32, None if sl.stop is None else sl.stop // 32)
        return w[rows, cols].contiguous().view(torch.uint8), w8_scale_chunks(s[div(rows), div(cols)].contiguous())

    out = {}
    out["wq_a_weight"], out["wq_a_scales"] = fp8("wq_a")
    out["wq_b_weight"], out["wq_b_scales"] = fp8("wq_b", rows=slice(q0, q0 + H * D))
    out["wkv_weight"], out["wkv_scales"] = fp8("wkv")
    out["wo_a_weight"], out["wo_a_scales"] = fp8("wo_a", rows=slice(a0, a0 + G * OL))
    out["wo_b_weight"], out["wo_b_scales"] = fp8("wo_b", cols=slice(a0, a0 + G * OL))
    out["q_norm_weight"], out["kv_norm_weight"] = read(p + "q_norm.weight"), read(p + "kv_norm.weight")
    out["attn_sink"] = read(p + "attn_sink")[H * rank:H * (rank + 1)].contiguous()
    out["freqs_cis"] = freqs_of(cfg, layer, rows, device)
    if var in ("full_r2", "full_r1", "reindex"):   # an index source: the indexer's own projections
        out["iwq_b_weight"], out["iwq_b_scales"] = fp8("indexer.wq_b")
        wp = read(p + "indexer.weights_proj.weight")   # [32, dim] bf16, padded to one 128-row gemm_tile block
        out["wproj"] = torch.cat([wp, torch.zeros(128 - wp.shape[0], wp.shape[1], dtype=wp.dtype)])
    if var.startswith("full"):                     # a KV source: the compressor and the index keys
        r = cfg["compress_ratios"][layer]
        cw = [read(p + "compressor.wkv.weight")] + ([read(p + "compressor.wgate.weight")] if r == 2 else [])
        out["cmp_weight"] = torch.cat(cw)
        out["cmp_norm_weight"] = read(p + "compressor.norm.weight")
        out["wk"], out["k_norm_weight"] = read(p + "indexer.wk.weight"), read(p + "indexer.k_norm.weight")
    return {k: v.to(device) for k, v in out.items()}
