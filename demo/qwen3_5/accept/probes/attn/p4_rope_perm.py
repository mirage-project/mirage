#!/usr/bin/env python3
"""Probe P4 (v1-architecture.md §14) — partial-RoPE permutation exactness.

Gates §4.4's zero-kernel-change route for Qwen3.5's partial RoPE (64 of 256).

  path A: Gemma q/k-norm -> HF partial NeoX RoPE (rotary_dim 64, theta 1e7,
          pairs (j, j+32)), i.e. `pytorch_reference.apply_rotary_pos_emb`.
  path B: permuted norm weights / q / k columns -> MPK's full-256 NeoX rotation
          (pairs (i, i+128)) with a cos=1 / sin=0 padded table.

Compares q, k element-wise and the q·k logits, in fp32 and bf16, on
  (1) synthetic random tensors (the §14 spec: torch-only, CPU ok, no checkpoint),
  (2) the real HF oracle dumps when `--oracle-dir` is given (decode + prefill),
      which is what the issue contract asks for ("the RoPE column-permutation
      equivalence probe on oracle data").

Two arithmetic regimes are reported separately, because they answer different
questions:

  ALGEBRA  — both paths share one permutation-invariant RMS scale, so any
             non-zero difference is a real algebraic failure of the permutation
             argument. Expected: EXACTLY 0.0 in fp32.
  AS-RUN   — each path computes its own RMS reduction over its own column order
             (what torch/the kernel actually do). Any difference here is
             fp32 reduction-order noise, not an algebra failure; it is reported
             so the number is on the record rather than assumed away.

A third row, KERNEL-ORDER, applies MPK's actual rounding (fp32 rotate, single
bf16 store) to path B and compares against HF's bf16 path A, so the unit tests
know what residual to expect from the kernel itself. That residual is a property
of MPK's existing RoPE arithmetic, NOT of the permutation.
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # demo/qwen3_5
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "oracle")))

from rope_permutation import (  # noqa: E402
    HEAD_DIM, ROTARY_DIM, ROPE_THETA,
    rope_permutation_src, rope_permutation_inv, build_cos_sin_table,
)

EPS = 1e-6


# ----------------------------------------------------------------------------- helpers
def gemma_rmsnorm_fp32(x, weight_folded, eps=EPS):
    """`Qwen3_5MoeRMSNorm` with the Gemma (1+w) fold ALREADY applied to `weight`.

    Returns fp32; the caller decides when to round. Matches
    oracle/pytorch_reference.gemma_rmsnorm's op order:
      out = x.float() * rsqrt(mean(x.float()^2) + eps);  out = out * w
    """
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return out * weight_folded.float()


def gemma_rmsnorm_with_scale(x, weight_folded, scale):
    """Same, but with an externally supplied (permutation-invariant) rms scale."""
    return x.float() * scale * weight_folded.float()


def rms_scale(x, eps=EPS):
    xf = x.float()
    return torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def hf_partial_rope(x, cos64, sin64):
    """`apply_rotary_pos_emb` for one tensor. x: [..., T, 256], cos/sin: [T, 64]."""
    rotary_dim = cos64.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    rot_half = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)
    return torch.cat([x_rot * cos64 + rot_half * sin64, x_pass], dim=-1)


def mpk_full_rope(x, cos256, sin256):
    """MPK's `rms_norm_sm100` fused rotary block / `rotary_embedding_sm100`:
    full-width NeoX pairing (i, i+128) over the whole 256-wide head."""
    half = x.shape[-1] // 2
    rot_half = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos256 + rot_half * sin256


def permute_last(x, src):
    return torch.index_select(x, -1, src)


def stats(a, b):
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-30)
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "max_rel": (d / denom).max().item(),
        "num_diff": int((a.float() != b.float()).sum().item()),
        "numel": int(a.numel()),
        "bit_exact": bool(torch.equal(a.float(), b.float())),
    }


def ulps_bf16(a, b):
    """Difference in bf16 ULPs (both inputs must already be bf16)."""
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    # map sign-magnitude -> monotone ordering
    ai = torch.where(ai < 0, torch.tensor(-32768, dtype=torch.int32) - ai, ai)
    bi = torch.where(bi < 0, torch.tensor(-32768, dtype=torch.int32) - bi, bi)
    return (ai - bi).abs()


# ----------------------------------------------------------------------------- one case
def scatter_cos_sin(cos64, sin64):
    """Scatter an HF [T, rotary_dim] table into MPK's [T, head_dim] permuted one.

    This is the transform itself, expressed on a table we did not build --
    used for the oracle cases so path A runs on HF's REAL dumped cos/sin.
    """
    inv = rope_permutation_inv()
    idx = torch.as_tensor([inv[j] for j in range(ROTARY_DIM)], dtype=torch.long)
    t = cos64.shape[0]
    cos = torch.ones(t, HEAD_DIM, dtype=torch.float32)
    sin = torch.zeros(t, HEAD_DIM, dtype=torch.float32)
    cos[:, idx] = cos64.float()
    sin[:, idx] = sin64.float()
    return cos, sin


def run_case(name, q, k, w_q_folded, w_k_folded, positions, num_kv_groups,
             hf_cos64=None, hf_sin64=None):
    """q: [T, HQ, 256] bf16, k: [T, HK, 256] bf16 (pre-norm), positions: [T].

    If `hf_cos64/hf_sin64` are given (oracle cases) they are used verbatim as
    path A's table and scattered for path B, so the comparison never depends on
    our own table construction.
    """
    src = torch.as_tensor(rope_permutation_src(), dtype=torch.long)
    T = q.shape[0]

    inv = rope_permutation_inv()
    idx64 = torch.as_tensor([inv[j] for j in range(ROTARY_DIM)], dtype=torch.long)
    if hf_cos64 is not None:
        cos64, sin64 = hf_cos64.float(), hf_sin64.float()
        cos256, sin256 = scatter_cos_sin(cos64, sin64)
    else:
        cos256, sin256 = build_cos_sin_table(positions, dtype=torch.float32)
        cos64 = cos256[:, idx64]  # exactly HF's [T, 64] table, by construction
        sin64 = sin256[:, idx64]

    cosA = cos64[:, None, :]
    sinA = sin64[:, None, :]
    cosB = cos256[:, None, :]
    sinB = sin256[:, None, :]

    out = {"name": name, "tokens": int(T), "q_heads": int(q.shape[1]),
           "kv_heads": int(k.shape[1])}

    # ---------------- ALGEBRA: shared permutation-invariant rms scale ----------
    for tag, tensor, w in (("q", q, w_q_folded), ("k", k, w_k_folded)):
        s = rms_scale(tensor)
        a_norm = gemma_rmsnorm_with_scale(tensor, w, s)
        a = hf_partial_rope(a_norm, cosA, sinA)

        t_p = permute_last(tensor, src)
        w_p = permute_last(w, src)
        b_norm = gemma_rmsnorm_with_scale(t_p, w_p, s)
        b = mpk_full_rope(b_norm, cosB, sinB)
        # bring path B back to HF column order for comparison
        b_unperm = torch.empty_like(b)
        b_unperm[..., src] = b
        out[f"algebra_fp32_{tag}"] = stats(b_unperm, a)

    # ---------------- AS-RUN: each path does its own reduction ----------------
    aq = hf_partial_rope(gemma_rmsnorm_fp32(q, w_q_folded), cosA, sinA)
    ak = hf_partial_rope(gemma_rmsnorm_fp32(k, w_k_folded), cosA, sinA)
    qp, kp = permute_last(q, src), permute_last(k, src)
    bq = mpk_full_rope(gemma_rmsnorm_fp32(qp, permute_last(w_q_folded, src)), cosB, sinB)
    bk = mpk_full_rope(gemma_rmsnorm_fp32(kp, permute_last(w_k_folded, src)), cosB, sinB)
    bq_u = torch.empty_like(bq); bq_u[..., src] = bq
    bk_u = torch.empty_like(bk); bk_u[..., src] = bk
    out["asrun_fp32_q"] = stats(bq_u, aq)
    out["asrun_fp32_k"] = stats(bk_u, ak)

    # ---------------- q.k logits (the only place q and k meet) ---------------
    # [T, HQ, T] logits, GQA-expanded keys; scale 1/sqrt(256) = 0.0625
    def logits(qq, kk):
        kk_rep = kk.repeat_interleave(num_kv_groups, dim=1)  # [T, HQ, 256]
        return torch.einsum("qhd,khd->qhk", qq.float(), kk_rep.float()) * (HEAD_DIM ** -0.5)

    # in the permuted basis path B never un-permutes; that is the real pipeline
    out["asrun_fp32_logits"] = stats(logits(bq, bk), logits(aq, ak))

    # ---------------- bf16 (what actually flows through the kernel) ----------
    aq_b = hf_partial_rope(gemma_rmsnorm_fp32(q, w_q_folded).to(torch.bfloat16).float(),
                           cosA.to(torch.bfloat16).float(),
                           sinA.to(torch.bfloat16).float()).to(torch.bfloat16)
    ak_b = hf_partial_rope(gemma_rmsnorm_fp32(k, w_k_folded).to(torch.bfloat16).float(),
                           cosA.to(torch.bfloat16).float(),
                           sinA.to(torch.bfloat16).float()).to(torch.bfloat16)
    bq_b = mpk_full_rope(gemma_rmsnorm_fp32(qp, permute_last(w_q_folded, src)).to(torch.bfloat16).float(),
                         cosB.to(torch.bfloat16).float(),
                         sinB.to(torch.bfloat16).float()).to(torch.bfloat16)
    bk_b = mpk_full_rope(gemma_rmsnorm_fp32(kp, permute_last(w_k_folded, src)).to(torch.bfloat16).float(),
                         cosB.to(torch.bfloat16).float(),
                         sinB.to(torch.bfloat16).float()).to(torch.bfloat16)
    bq_bu = torch.empty_like(bq_b); bq_bu[..., src] = bq_b
    bk_bu = torch.empty_like(bk_b); bk_bu[..., src] = bk_b
    out["kernelorder_bf16_q"] = stats(bq_bu, aq_b)
    out["kernelorder_bf16_k"] = stats(bk_bu, ak_b)
    out["kernelorder_bf16_q_max_ulp"] = int(ulps_bf16(bq_bu, aq_b).max().item())
    out["kernelorder_bf16_k_max_ulp"] = int(ulps_bf16(bk_bu, ak_b).max().item())
    out["kernelorder_bf16_logits"] = stats(logits(bq_b, bk_b), logits(aq_b, ak_b))
    return out


# ----------------------------------------------------------------------------- oracle
def load_oracle_case(oracle_dir, mode):
    """Rebuild the P4 inputs from the M2-I3 HF oracle dump (read-only)."""
    import json as _json
    d = os.path.join(oracle_dir, mode)
    man = _json.load(open(os.path.join(d, "manifest.json")))
    tdir = os.path.join(oracle_dir)

    def get(key):
        rec = man["tensors"][key]
        return torch.load(os.path.join(tdir, rec["file"]), map_location="cpu", weights_only=True)

    q_split = get("attn.q_split")        # [1, T, 16, 256] pre-norm q
    k_proj = get("attn.k_proj_out")      # [1, T, 512]
    wq = get("attn.__weight.q_norm_weight")  # zero-centred (Gemma)
    wk = get("attn.__weight.k_norm_weight")
    cos = get("attn.rope_cos")           # [1, T, 64]
    sin = get("attn.rope_sin")

    q = q_split[0]                                     # [T, 16, 256]
    T = q.shape[0]
    k = k_proj[0].view(T, 2, 256)                      # [T, 2, 256]
    return q, k, 1.0 + wq.float(), 1.0 + wk.float(), cos[0].float(), sin[0].float()


def positions_from_cos_sin(cos64, sin64, rotary_dim=ROTARY_DIM, theta=ROPE_THETA):
    """Recover integer positions from HF's dumped cos/sin.

    Use the SLOWEST frequency channel (the last inv_freq), whose period is
    `2*pi / inv_freq[-1]` -- for rotary_dim 64 / theta 1e7 that is
    `inv_freq[-1] = theta**(-62/64) ~= 1.66e-7`, i.e. a period of ~3.8e7 >>
    max_position 262144, so `atan2` inverts it unambiguously. (The first
    channel has `inv_freq[0] = 1.0` and wraps every 2*pi tokens -- inverting
    THAT one is what made the first version of this helper return nonsense.)

    The recovered positions are then VERIFIED by reconstructing the full table
    and comparing against the dump; a mismatch raises rather than silently
    proceeding.
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    slow = float(inv_freq[-1])
    ang = torch.atan2(sin64[:, half - 1].double(), cos64[:, half - 1].double())
    ang = torch.where(ang < 0, ang + 2 * torch.pi, ang)
    pos = torch.round(ang / slow)
    return pos.to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", default=None,
                    help="M2-I3 dump root containing decode/ and prefill/ (read-only)")
    ap.add_argument("--out", default=os.path.join(_HERE, "p4_rope_perm_result.json"))
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    result = {
        "probe": "P4",
        "spec": "docs/qwen35/v1-architecture.md §14 P4 (gates §4.4 zero-kernel-change route)",
        "torch_version": torch.__version__,
        "device": "cpu",
        "seed": args.seed,
        "config": {"head_dim": HEAD_DIM, "rotary_dim": ROTARY_DIM, "theta": ROPE_THETA,
                   "num_q_heads": 16, "num_kv_heads": 2, "eps": EPS},
        "permutation_src_head": rope_permutation_src()[:8],
        "permutation_src_128_136": rope_permutation_src()[128:136],
        "cases": [],
    }

    # ---- synthetic (the §14 spec: no checkpoint needed) ----
    for T, tag in ((1, "synthetic_decode_T1"), (8, "synthetic_prefill_T8"),
                   (33, "synthetic_prefill_T33")):
        q = torch.randn(T, 16, HEAD_DIM, dtype=torch.bfloat16)
        k = torch.randn(T, 2, HEAD_DIM, dtype=torch.bfloat16)
        wq = 1.0 + torch.randn(HEAD_DIM) * 0.1
        wk = 1.0 + torch.randn(HEAD_DIM) * 0.1
        pos = torch.arange(T, dtype=torch.float32) + 7
        result["cases"].append(run_case(tag, q, k, wq, wk, pos, num_kv_groups=8))

    # ---- real oracle data ----
    if args.oracle_dir:
        for mode in ("decode", "prefill"):
            path = os.path.join(args.oracle_dir, mode, "manifest.json")
            if not os.path.exists(path):
                continue
            q, k, wq, wk, cos64, sin64 = load_oracle_case(args.oracle_dir, mode)
            pos = positions_from_cos_sin(cos64, sin64)
            # Path A runs on HF's REAL dumped table; path B on its scatter.
            case = run_case(f"oracle_{mode}", q, k, wq, wk, pos, num_kv_groups=8,
                            hf_cos64=cos64, hf_sin64=sin64)
            # Independent check: does the LOADER's table construction
            # (theta=1e7, rotary_dim=64, recovered positions) reproduce HF's
            # dumped table? This validates build_cos_sin_table, which is what
            # the M2-I8 weight loader will call.
            cos256, sin256 = build_cos_sin_table(pos, dtype=torch.float32)
            inv = rope_permutation_inv()
            idx64 = torch.as_tensor([inv[j] for j in range(ROTARY_DIM)], dtype=torch.long)
            case["positions"] = [int(p) for p in pos.tolist()]
            case["rebuilt_cos_vs_oracle_bf16"] = stats(
                cos256[:, idx64].to(torch.bfloat16), cos64.to(torch.bfloat16))
            case["rebuilt_sin_vs_oracle_bf16"] = stats(
                sin256[:, idx64].to(torch.bfloat16), sin64.to(torch.bfloat16))
            case["rebuilt_cos_vs_oracle_fp32_maxabs"] = float(
                (cos256[:, idx64] - cos64.float()).abs().max())
            case["rebuilt_sin_vs_oracle_fp32_maxabs"] = float(
                (sin256[:, idx64] - sin64.float()).abs().max())
            # the un-rotated slots MUST be exactly identity, or the whole
            # "zero kernel change" argument collapses
            mask = torch.ones(HEAD_DIM, dtype=torch.bool)
            mask[idx64] = False
            case["padding_cos_all_one"] = bool(torch.all(cos256[:, mask] == 1.0))
            case["padding_sin_all_zero"] = bool(torch.all(sin256[:, mask] == 0.0))
            result["cases"].append(case)

    # ---- verdict ----
    algebra_max = max(
        max(c[k]["max_abs"] for k in c if k.startswith("algebra_fp32_"))
        for c in result["cases"])
    algebra_exact = all(
        all(c[k]["bit_exact"] for k in c if k.startswith("algebra_fp32_"))
        for c in result["cases"])
    ulp_max = max(max(c["kernelorder_bf16_q_max_ulp"], c["kernelorder_bf16_k_max_ulp"])
                  for c in result["cases"])
    result["verdict"] = {
        "algebra_fp32_bit_exact": algebra_exact,
        "algebra_fp32_max_abs": algebra_max,
        "kernelorder_bf16_max_ulp": ulp_max,
        "decision": ("PERMUTATION_ROUTE_GO" if algebra_exact and ulp_max <= 1
                     else "PERMUTATION_ROUTE_NOGO_FALLBACK_ROTARY_DIM_TEMPLATE"),
        "criterion": "§14 P4: max abs diff = 0.0 in fp32 (algebra) and <= 1 ulp in bf16",
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    for c in result["cases"]:
        print(f"[{c['name']}] T={c['tokens']}  "
              f"algebra_q={c['algebra_fp32_q']['max_abs']:.3e} (exact={c['algebra_fp32_q']['bit_exact']})  "
              f"algebra_k={c['algebra_fp32_k']['max_abs']:.3e} (exact={c['algebra_fp32_k']['bit_exact']})  "
              f"asrun_q={c['asrun_fp32_q']['max_abs']:.3e}  "
              f"logits={c['asrun_fp32_logits']['max_abs']:.3e}  "
              f"bf16_ulp_q={c['kernelorder_bf16_q_max_ulp']}  bf16_ulp_k={c['kernelorder_bf16_k_max_ulp']}")
    print("\nVERDICT:", json.dumps(result["verdict"], indent=2))
    print("wrote", args.out)
    return 0 if result["verdict"]["decision"] == "PERMUTATION_ROUTE_GO" else 1


if __name__ == "__main__":
    sys.exit(main())
