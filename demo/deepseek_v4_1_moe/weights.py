"""DeepSeek-V4.1-Flash: one MoE layer's tensors for demo.py, read from the checkpoint.

The checkpoint (a local snapshot of deepseek-ai/DeepSeek-V4.1-Flash: inference/config.json, the safetensors index and shards)
keeps the official names (layers.<l>.ffn.*). layer_tensors returns one layer's tensors for GPU `rank` of MoE TP `tp` under
the graph's names (demo.build_graph):
  gate_weight   layers.<l>.ffn.gate.weight, bf16 [384, 5120] (the router; the same on every GPU)
  gate_bias     layers.<l>.ffn.gate.bias, fp32 [384] (it picks the experts; text tokens use bias, not bias_vl)
  sgu_weight    the shared expert's w1 and w3 rows [I r, I (r + 1)) (I = 2304 / tp), stacked: e4m3 bytes [2 I, 5120]
  sgu_scales    their 32 x 32 block scales as gemm_fp8_tile's scale chunks (w8_scale_chunks)
  sd_weight     w2's columns [I r, I (r + 1)), then zero columns up to KO = shared_down_k(I): e4m3 bytes [5120, KO]
  sd_scales     their scale chunks (the padding's scales 2^0)
Every slice boundary is a multiple of 32, so the checkpoint's block scales are copied unchanged.

routed_tensors returns the routed experts (layers.<l>.ffn.experts.<e>.w1 | w2 | w3, MXFP4: e2m1 pairs with an e8m0 scale per 32)
for the given ranks, as tasks/swiglu_moe_experts.cuh reads them: 128 x 128 pieces (8 KB, [128 rows][64 B], the checkpoint's
bytes) with a 512-B scale chunk each (the scale of row r and K columns 32 j .. + 31 at byte 16 (r % 32) + 4 (r // 32) + j):
  w13         [384, 2 I, 2560]: per expert I / 64 items of the rows [I r + 64 m, + 64) of w1 then the same rows of w3, each in 40 K
              pieces
  w13_scales  [384, 2 I, 160]: their chunks in the same order
  w2          [384, 5120, IRP / 2]: per expert 40 output tiles of 128 rows, each in IRP / 128 K pieces of w2's columns [I r,
              I (r + 1)), then zero columns up to IRP = I padded to whole 128 (TP4: 576 -> 640)
  w2_scales   [384, 5120, IRP / 32] (the padding's scales 2^0)
"""
import json
import os

import torch
from safetensors import safe_open
from mirage.mpk.static_megakernel import w8_scale_chunks

INTER = 2304   # moe_inter_dim


def download_layer(checkpoint: str, layer: int, cache_dir: str = None, revision: str = None) -> str:
    """Return a local snapshot path with the layer's MoE tensors.

    A local directory is returned as it is. A Hugging Face repo id (deepseek-ai/DeepSeek-V4.1-Flash): resolve the revision
    (default: main) to one commit, then download inference/config.json, the index and the shards that hold layer `layer`'s
    MoE tensors only (the whole repo is about 510 GB)."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    sha = HfApi().model_info(checkpoint, revision=revision).sha
    print(f"{checkpoint}: revision {revision or 'main'} = commit {sha}")
    index = hf_hub_download(checkpoint, "model.safetensors.index.json", revision=sha, cache_dir=cache_dir)
    with open(index) as f:
        where = json.load(f)["weight_map"]
    shards = sorted({f for name, f in where.items() if name.startswith(f"layers.{layer}.ffn.")})
    if not shards:
        raise ValueError(f"{checkpoint}: no MoE tensors of layer {layer}")
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
    """inference/config.json: the official code's ModelArgs (experts, top-k, score function, route scale)."""
    return json.load(open(os.path.join(snapshot, "inference", "config.json")))


def piece_chunks(s: torch.Tensor) -> torch.Tensor:
    """Per-row e8m0 scales of pieces [..., 128 rows, 4] (bytes) -> their 512-B scale chunks [..., 512]."""
    return s.reshape(*s.shape[:-2], 4, 32, 4).transpose(-3, -2).reshape(*s.shape[:-2], 512)


def routed_tensors(read, layer: int, tp: int, ranks, device="cuda", experts: int = 384, dim: int = 5120) -> list:
    """The routed experts of layer `layer` for each rank in `ranks` of MoE TP `tp` (the module comment): a dict per rank, on
    `device` (or on device[i] for rank i: a list)."""
    I = INTER // tp
    IRP, MT, KT = -(-I // 128) * 128, I // 64, dim // 128
    devs = list(device) if isinstance(device, (list, tuple)) else [device] * len(ranks)
    out = [{"w13": torch.empty(experts, 2 * I, dim // 2, dtype=torch.uint8, device=d),
            "w13_scales": torch.empty(experts, 2 * I, dim // 32, dtype=torch.uint8, device=d),
            "w2": torch.zeros(experts, dim, IRP // 2, dtype=torch.uint8, device=d),
            "w2_scales": torch.full((experts, dim, IRP // 32), 127, dtype=torch.uint8, device=d)} for d in devs]
    for e in range(experts):
        x = lambda n: read(f"layers.{layer}.ffn.experts.{e}.{n}").contiguous().view(torch.uint8)
        c1, c3, c2 = x("w1.weight"), x("w3.weight"), x("w2.weight")    # [INTER, dim / 2], [INTER, dim / 2], [dim, INTER / 2]
        d1, d3, d2 = x("w1.scale"), x("w3.scale"), x("w2.scale")       # [INTER, dim / 32], [INTER, dim / 32], [dim, INTER / 32]
        for o, r, dv in zip(out, ranks, devs):
            w1, w3, w2, s1, s3, s2 = (v.to(dv) for v in (c1, c3, c2, d1, d3, d2))
            rows = slice(I * r, I * (r + 1))
            item = lambda a, b, n: torch.cat([a[rows].view(MT, 64, n), b[rows].view(MT, 64, n)], 1)   # [MT, 128 rows, n]
            o["w13"][e].view(MT, KT, 128, 64).copy_(item(w1, w3, dim // 2).view(MT, 128, KT, 64).transpose(1, 2))
            o["w13_scales"][e].view(MT, KT, 512).copy_(
                piece_chunks(item(s1, s3, dim // 32).view(MT, 128, KT, 4).transpose(1, 2)))
            w2p = o["w2"][e].clone()
            w2p[:, :I // 2] = w2[:, I * r // 2:I * (r + 1) // 2]
            s2p = o["w2_scales"][e].clone()
            s2p[:, :I // 32] = s2[:, I * r // 32:I * (r + 1) // 32]
            o["w2"][e].view(KT, IRP // 128, 128, 64).copy_(w2p.view(KT, 128, IRP // 128, 64).transpose(1, 2))
            o["w2_scales"][e].view(KT, IRP // 128, 512).copy_(piece_chunks(s2p.view(KT, 128, IRP // 128, 4).transpose(1, 2)))
    return out


def unpack_pieces(w: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
    """Packed pieces back to values: bytes [..., 128, 64] (e2m1 pairs, low nibble first) and their 512-B chunks [..., 512] ->
    fp64 [..., 128, 128] (a check of the packing: the demo's references)."""
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64, device=w.device)
    v = torch.stack([lut[(w & 15).long()], lut[(w >> 4).long()]], -1).flatten(-2)
    s = chunks.view(*chunks.shape[:-1], 32, 4, 4).transpose(-3, -2).reshape(*chunks.shape[:-1], 128, 4).double() - 127
    return v * torch.exp2(s).repeat_interleave(32, -1)


def shared_down_k(inter: int) -> int:
    """Shared down's K on a GPU: its I columns padded to whole 256-column stages, at least 2 (gemm_fp8_tile; TP4: 576 -> 768)."""
    return max(512, -(-inter // 256) * 256)


def layer_tensors(read, layer: int, tp: int = 4, rank: int = 0, device="cuda") -> dict:
    """Layer `layer`'s MoE tensors for GPU `rank` of MoE TP `tp` on `device`, by graph tensor name (the module comment). tp is
    1, 2 or 4 (I a multiple of 64: whole 128-row blocks of gate | up)."""
    assert tp in (1, 2, 4) and 0 <= rank < tp
    p, I = f"layers.{layer}.ffn.", INTER // tp
    KO, r = shared_down_k(I), slice(I * rank, I * (rank + 1))
    r32 = slice(r.start // 32, r.stop // 32)
    w = {"gate_weight": read(p + "gate.weight"), "gate_bias": read(p + "gate.bias")}
    assert w["gate_weight"].dtype == torch.bfloat16 and w["gate_bias"].dtype == torch.float32
    e = lambda name: read(p + "shared_experts." + name)
    b = lambda t: t.contiguous().view(torch.uint8)   # e4m3 / e8m0 as bytes
    w["sgu_weight"] = torch.cat([b(e("w1.weight")[r]), b(e("w3.weight")[r])])
    w["sgu_scales"] = w8_scale_chunks(torch.cat([b(e("w1.scale")[r32]), b(e("w3.scale")[r32])]))
    w2, s2 = b(e("w2.weight")[:, r]), b(e("w2.scale")[:, r32])
    w["sd_weight"] = torch.cat([w2, torch.zeros(w2.shape[0], KO - I, dtype=torch.uint8)], 1)
    w["sd_scales"] = w8_scale_chunks(torch.cat([s2, torch.full((s2.shape[0], (KO - I) // 32), 127, dtype=torch.uint8)], 1))
    return {k: v.contiguous().to(device) for k, v in w.items()}
