"""GLM-5.3-Flash KDA attention: checkpoint reading, TP sharding and a PyTorch reference.

The reference follows vLLM's glm5next implementation step by step, including where it rounds to bf16 (GEMM outputs, the conv
output, core_attn_out), so the static layer can be compared value by value:
  KDA       merged projection q | k | v | b | f_a | g_a (bf16) -> short conv + SiLU (fp32 weights, bf16 out) -> q, k L2 norm,
            q D^-1/2 -> gate lb sigmoid(e^A (f_b f_a + dt_bias)), beta sigmoid(b) -> delta rule on S [V][K] fp32 ->
            o (bf16) -> RMSNorm(o) w sigmoid(g_b g_a) -> o_proj                                    (kda.py, fused_recurrent)
(the mHC pre / post around it: dsa_fused_demo.hc_pre_ref / hc_post_ref, SGLang's formulas)
"""
import json
import os
import re
from typing import Dict

import torch

D = 128   # KDA head dim (linear_head_dim)


class Checkpoint:
    """Single tensors from a sharded safetensors checkpoint (model.safetensors.index.json); block-FP8 weights (with a
    weight_scale_inv) come back dequantized to bf16."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.weight_map: Dict[str, str] = json.load(f)["weight_map"]

    def _raw(self, key: str) -> torch.Tensor:
        from safetensors import safe_open
        with safe_open(os.path.join(self.dir, self.weight_map[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def get(self, key: str) -> torch.Tensor:
        t = self._raw(key)
        if t.dtype == torch.float8_e4m3fn:
            skey = key[: -len("weight")] + "weight_scale_inv"
            if skey not in self.weight_map:
                raise KeyError(f"{key} is FP8 without {skey}")
            t = dequant_fp8_block(t, self._raw(skey))
        return t

    def layer_key(self, layer: int, tail: str) -> str:
        """The one key of decoder layer `layer` (not the MTP or vision layers) ending in `tail`."""
        pat = re.compile(rf"(^|\.)layers\.{layer}\.")
        hits = [k for k in self.weight_map
                if pat.search(k) and k.endswith(tail) and "visual" not in k and "mtp" not in k]
        if len(hits) != 1:
            raise KeyError(f"layer {layer}: {len(hits)} keys end in {tail!r}: {hits[:4]}")
        return hits[0]

    def layer(self, layer: int, tail: str) -> torch.Tensor:
        return self.get(self.layer_key(layer, tail))

    def embed_rows(self, ids) -> torch.Tensor:
        """Rows `ids` of the text embedding (bf16 [len(ids), H]), read one by one."""
        from safetensors import safe_open
        keys = [k for k in self.weight_map if k.endswith("embed_tokens.weight") and "visual" not in k and "mtp" not in k]
        key = min(keys, key=len)
        with safe_open(os.path.join(self.dir, self.weight_map[key]), framework="pt", device="cpu") as f:
            sl = f.get_slice(key)
            return torch.cat([sl[i:i + 1] for i in ids], 0)


def dequant_fp8_block(w: torch.Tensor, scale_inv: torch.Tensor, block: int = 128) -> torch.Tensor:
    """e4m3 [out, in] with fp32 scales per 128 x 128 block -> bf16 (as vLLM's _dequant_fp8_block)."""
    out_dim, in_dim = w.shape
    s = scale_inv.float().repeat_interleave(block, 0).repeat_interleave(block, 1)[:out_dim, :in_dim]
    return (w.float() * s).to(torch.bfloat16)


def load_kda_layer(ck: Checkpoint, layer: int) -> dict:
    """The full (all heads) weights of KDA layer `layer`, on the CPU."""
    def g(*tails):   # the first of the candidate names the checkpoint has
        for i, tail in enumerate(tails):
            try:
                return ck.layer(layer, tail)
            except KeyError:
                if i == len(tails) - 1:
                    raise
    hc = lambda part: g(f"hc_attn_{part}", f"attn_hc.{part}")
    w = {
        "q": g("self_attn.q_proj.weight"), "k": g("self_attn.k_proj.weight"), "v": g("self_attn.v_proj.weight"),
        "b": g("self_attn.b_proj.weight"), "f_a": g("f_a_proj.weight"), "g_a": g("g_a_proj.weight"),
        "f_b": g("f_b_proj.weight"), "g_b": g("g_b_proj.weight"),
        "q_conv": g("q_conv1d.weight"), "k_conv": g("k_conv1d.weight"), "v_conv": g("v_conv1d.weight"),
        "A_log": g("A_log"), "dt_bias": g("dt_bias"), "o_norm": g("o_norm.weight"), "o_proj": g("self_attn.o_proj.weight"),
        "hc_fn": hc("fn"), "hc_base": hc("base"), "hc_scale": hc("scale"),
        "gamma": g("input_layernorm.weight"),
    }
    for c in ("q_conv", "k_conv", "v_conv"):
        w[c] = w[c].reshape(w[c].shape[0], -1).float()          # [8192, 1, 4] -> [8192, 4]
    w["A_log"] = w["A_log"].reshape(-1).float()
    w["dt_bias"] = w["dt_bias"].reshape(-1).float()
    w["o_norm"] = w["o_norm"].float()
    for kk in ("hc_fn", "hc_base", "hc_scale"):
        w[kk] = w[kk].float()
    w["num_heads"] = w["A_log"].numel()
    return w


def shard_kda(w: dict, rank: int, tp: int, device) -> dict:
    """GPU `rank`'s tensors of the KDA sublayer at TP `tp` (heads split, f_a / g_a / mHC replicated)."""
    NH = w["num_heads"]
    HL = NH // tp
    cs, hs = slice(rank * HL * D, (rank + 1) * HL * D), slice(rank * HL, (rank + 1) * HL)
    rows = [w["q"][cs], w["k"][cs], w["v"][cs], w["b"][hs], w["f_a"], w["g_a"]]
    n = sum(r.shape[0] for r in rows)
    pad = (-n) % 128
    w_in = torch.cat(rows + [torch.zeros(pad, rows[0].shape[1], dtype=torch.bfloat16)], 0)
    hc_small = torch.zeros(32, dtype=torch.float32)
    hc_small[:24], hc_small[24:27] = w["hc_base"], w["hc_scale"]
    return {
        "w_in": w_in.contiguous().to(device),
        "conv_w": torch.cat([w["q_conv"][cs], w["k_conv"][cs], w["v_conv"][cs]], 0).contiguous().to(device),
        "f_b": w["f_b"][cs].contiguous().to(device), "g_b": w["g_b"][cs].contiguous().to(device),
        "small": torch.cat([w["dt_bias"][cs], w["A_log"][hs], w["o_norm"]]).contiguous().to(device),
        "w_o": w["o_proj"][:, cs].contiguous().to(device),
        "hc_fn": w["hc_fn"].contiguous().to(device), "hc_small": hc_small.to(device),
        "gamma": w["gamma"].contiguous().to(device),
    }


# ---------------- the reference (all heads, one device) ----------------
def bf(x):
    return x.to(torch.bfloat16)


def mm_bf16(a, w):
    """a [T, K] bf16 @ w [N, K]^T -> bf16 (fp32 accumulation, one rounding: a bf16 GEMM)."""
    return bf(a.float() @ w.float().t())


def tp_o_proj(y, w_o, tp: int):
    """o_proj as TP `tp` computes it (vLLM's RowParallelLinear + all-reduce, and the static layer): each GPU's partial
    product rounded to bf16, the partials added in fp32 in GPU order, rounded once more."""
    HLD = y.shape[1] // tp
    acc = torch.zeros(y.shape[0], w_o.shape[0], dtype=torch.float32, device=y.device)
    for r in range(tp):
        acc += mm_bf16(y[:, r * HLD:(r + 1) * HLD], w_o[:, r * HLD:(r + 1) * HLD]).float()
    return bf(acc)


def ref_kda(x, w, conv_state, rec, lower_bound=-5.0, norm_eps=1e-5, tp: int = 1):
    """One decode step of T requests: token t's state slot and conv state row are t ([T, 3 NH D, 3], oldest first).
    Returns (o_proj output, o after the gated norm, new conv state, new recurrent state); o_proj summed as TP `tp`
    does."""
    dev = x.device
    NH = w["num_heads"]
    T = x.shape[0]
    assert conv_state.shape[2] == 3
    q, k, v = mm_bf16(x, w["q"].to(dev)), mm_bf16(x, w["k"].to(dev)), mm_bf16(x, w["v"].to(dev))
    b, fa, ga = mm_bf16(x, w["b"].to(dev)), mm_bf16(x, w["f_a"].to(dev)), mm_bf16(x, w["g_a"].to(dev))
    qkv = torch.cat([q, k, v], 1)                                            # [T, 3 NH D] bf16
    conv_w = torch.cat([w["q_conv"], w["k_conv"], w["v_conv"]], 0).to(dev)   # [3 NH D, 4]
    co = torch.empty(T, qkv.shape[1], dtype=torch.float32, device=dev)
    new_conv = conv_state.clone()
    for t in range(T):   # causal_conv1d_update
        xs = torch.cat([conv_state[t].float(), qkv[t:t + 1].float().t()], 1)   # [C, 4]
        co[t] = (xs * conv_w).sum(-1)
        new_conv[t] = torch.cat([conv_state[t, :, 1:3], qkv[t:t + 1].t()], 1)
    co = bf(co * torch.sigmoid(co)).float()
    cq, ck, cv = co.split(NH * D, 1)
    g1 = mm_bf16(fa, w["f_b"].to(dev)).float().view(T, NH, D)
    gate = lower_bound * torch.sigmoid(torch.exp(w["A_log"].to(dev)).view(1, NH, 1) *
                                       (g1 + w["dt_bias"].to(dev).view(1, NH, D)))
    og = mm_bf16(ga, w["g_b"].to(dev)).float().view(T, NH, D)
    beta = torch.sigmoid(b.float())                                          # [T, NH]
    qn = cq.view(T, NH, D)
    qn = qn / torch.sqrt(qn.square().sum(-1, keepdim=True) + 1e-6) * (D ** -0.5)
    kn = ck.view(T, NH, D)
    kn = kn / torch.sqrt(kn.square().sum(-1, keepdim=True) + 1e-6)
    cv = cv.view(T, NH, D)
    o = torch.empty(T, NH, D, dtype=torch.float32, device=dev)
    new_rec = rec.clone()
    for t in range(T):   # fused_recurrent_kda: one state slot per token
        S = rec[t].float()                                                   # [NH, V, K]
        S = S * torch.exp(gate[t]).unsqueeze(1)
        kv = (S * kn[t].unsqueeze(1)).sum(-1)
        delta = (cv[t] - kv) * beta[t].unsqueeze(-1)
        S = S + delta.unsqueeze(-1) * kn[t].unsqueeze(1)
        o[t] = bf((S * qn[t].unsqueeze(1)).sum(-1)).float()
        new_rec[t] = S
    y = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + norm_eps) * w["o_norm"].to(dev) * torch.sigmoid(og)
    y = bf(y).reshape(T, NH * D)
    return tp_o_proj(y, w["o_proj"].to(dev), tp), y, new_conv, new_rec

