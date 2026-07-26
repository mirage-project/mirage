#!/usr/bin/env python3
"""M2-I6 -- SM100 attention (Qwen3.5 QKVG-gate + Q-loop) vs the HF oracle.

Ground truth is the M2-I3 per-op dump of the REAL HF FP8 checkpoint
(`demo/qwen3_5/oracle/`, probe P6), layer 3 (the first full-attention layer),
for a 8-token prefill and the following decode step.

Per the standing project rule, HF's EMPIRICAL rounding is the authority: where
the architecture doc and HF disagree in low-order bits, HF wins. Two orderings
are therefore asserted as COUNTERFACTUALS that must MISS, so the choice the
kernel makes is recorded as a measurement rather than an assumption:

  * the output gate with the sigmoid kept in fp32 and folded into a single
    rounding (HF rounds sigmoid to bf16 FIRST -- see `--counterfactuals`);
  * RMSNorm taken over the rotary_dim (64) instead of the full head_dim (256).

Intermediates checked against the oracle
----------------------------------------
  q/k norm + RoPE : the kernel's own `rms_norm_sm100` output, observed DIRECTLY
                    -- for k through the paged-cache write, and for q by
                    replaying the oracle's q heads through the k slot with the
                    q_norm weight (identical code path, NUM_HEAD=1) so the
                    normed+roped q is materialised in the cache and comparable
                    to `attn.q_rope`.
  KV write        : `attn.kv_cache_{k,v}_after_write`, including the §2.2.5
                    ordering invariant (the current token is in the cache and
                    is attended by its own query).
  attention out   : `attn.core_attn_out`   (UNGATED instantiation)
  gated out       : `attn.gate_sigmoid_mul_out` (gated instantiation)

Run:
  python test_attention_qwen35_oracle.py --oracle-dir ~/mpk-qwen35/oracle-work/dumps
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "demo", "qwen3_5"))
sys.path.insert(0, _HERE)

from rope_permutation import (  # noqa: E402
    HEAD_DIM, ROTARY_DIM, rope_permutation_src, build_cos_sin_table,
)

import runtime_kernel_blackwell_attention_qwen35 as K  # noqa: E402

NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
NUM_QO_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
PAGE_SIZE = 64
MAX_SEQ_LEN = 2048
MAX_PAGES = 8
EPS = 1e-6
DEV = "cuda"
BF = torch.bfloat16


# ----------------------------------------------------------------- oracle I/O
class Oracle:
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.man = json.load(open(os.path.join(root, mode, "manifest.json")))

    def __getitem__(self, key):
        rec = self.man["tensors"][key]
        return torch.load(os.path.join(self.root, rec["file"]),
                          map_location="cpu", weights_only=True)


# ----------------------------------------------------------------- reporting
def cmp(name, got, ref, notes=""):
    g, r = got.float().cpu(), ref.float().cpu()
    d = (g - r).abs()
    ndiff = int((g != r).sum())
    rec = {
        "intermediate": name,
        "shape": list(got.shape),
        "numel": int(g.numel()),
        "bit_exact": ndiff == 0,
        "num_diff": ndiff,
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel": float((d / r.abs().clamp_min(1e-30)).max()),
        "notes": notes,
    }
    return rec


def show(rows):
    w = max(len(r["intermediate"]) for r in rows) + 2
    print(f"\n{'intermediate':<{w}}{'exact':<7}{'ndiff':>12}{'max_abs':>13}{'mean_abs':>13}")
    print("-" * (w + 45))
    for r in rows:
        print(f"{r['intermediate']:<{w}}{str(r['bit_exact']):<7}"
              f"{r['num_diff']:>6}/{r['numel']:<6}{r['max_abs']:>13.3e}{r['mean_abs']:>13.3e}")


# ----------------------------------------------------------------- builders
def perm_idx(device):
    return torch.as_tensor(rope_permutation_src(), dtype=torch.long, device=device)


# ------------------------------------------------------- torch reference ladder
# Two rounding orders for RoPE. HF's `apply_rotary_pos_emb` evaluates
#   (x_rot * cos) + (rotate_half(x_rot) * sin)
# on bf16 tensors, so EACH PRODUCT is rounded to bf16 before the sum. MPK's
# kernel keeps both products in fp32 and rounds once. Which one the kernel
# should match is an empirical question about HF, so both are computed and
# reported rather than assumed.
def rope_hf_rounding(x_bf16, cos_bf16, sin_bf16):
    half = x_bf16.shape[-1] // 2
    rh = torch.cat((-x_bf16[..., half:], x_bf16[..., :half]), dim=-1)
    return ((x_bf16.float() * cos_bf16.float()).to(BF).float()
            + (rh.float() * sin_bf16.float()).to(BF).float()).to(BF)


def rope_fp32_fused(x_bf16, cos_bf16, sin_bf16):
    half = x_bf16.shape[-1] // 2
    rh = torch.cat((-x_bf16[..., half:], x_bf16[..., :half]), dim=-1)
    return (x_bf16.float() * cos_bf16.float()
            + rh.float() * sin_bf16.float()).to(BF)


def gemma_norm_bf16(x, w_folded):
    """`Qwen3_5MoeRMSNorm`: fp32 normalise+weight, cast to bf16 LAST."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (out * w_folded.float()).to(BF)


def mpk_norm_bf16(x, w_folded):
    """`rms_norm_sm100`: sum of squares / HEAD_DIM, then val *= rms_rcp * w."""
    xf = x.float()
    rms_rcp = torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) / x.shape[-1] + EPS)
    return (xf * (rms_rcp * w_folded.float())).to(BF)


def build_qkvg(q_perm, gate, k_perm, v, gated):
    """Assemble MPK's kv-group-interleaved packed row.

    gated:   [ g0: (q|gate) x NUM_QO_PER_KV, k, v ][ g1: ... ]   width 9216
    ungated: [ g0:  q       x NUM_QO_PER_KV, k, v ][ g1: ... ]   width 5120

    q_perm/gate: [T, 16, 256]; k_perm/v: [T, 2, 256].
    """
    T = q_perm.shape[0]
    groups = []
    for g in range(NUM_KV_HEADS):
        hs = slice(g * NUM_QO_PER_KV, (g + 1) * NUM_QO_PER_KV)
        if gated:
            # interleave [q|gate] per head -> [T, NUM_QO_PER_KV, 2, 256]
            qg = torch.stack([q_perm[:, hs, :], gate[:, hs, :]], dim=2)
            head_block = qg.reshape(T, NUM_QO_PER_KV * 2 * HEAD_DIM)
        else:
            head_block = q_perm[:, hs, :].reshape(T, NUM_QO_PER_KV * HEAD_DIM)
        groups.append(torch.cat([head_block, k_perm[:, g, :], v[:, g, :]], dim=1))
    return torch.cat(groups, dim=1).contiguous()


def empty_cache():
    k = torch.zeros(MAX_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=BF, device=DEV)
    return k, torch.zeros_like(k)


def meta(num_tokens, seq_len, num_requests=1):
    npages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    t = lambda x: torch.tensor(x, dtype=torch.int32, device=DEV)
    return (t([0, num_tokens]), t([0, npages]), t(list(range(npages))),
            t([seq_len - (npages - 1) * PAGE_SIZE]))


def run_kernel(qkv, kc, vc, out, num_tokens, seq_len, *, max_tokens, gate,
               q_pass, qn, kn, cos, sin, num_qo_per_kv=NUM_QO_PER_KV,
               num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM):
    qo, kvp, kvi, last = meta(num_tokens, seq_len)
    K.attention_qwen35(qkv, kc, vc, out, qo, kvp, kvi, last, qn, kn, cos, sin,
                       1, num_qo_per_kv, num_kv_heads, head_dim, max_tokens,
                       gate, q_pass, True, True)


# ----------------------------------------------------------------- main case
def run_mode(orc, mode, args, rows, failures):
    tag = mode
    q_split = orc["attn.q_split"][0].to(DEV)            # [T, 16, 256] pre-norm
    gate = orc["attn.gate_split"][0].view(-1, NUM_Q_HEADS, HEAD_DIM).to(DEV)
    k_proj = orc["attn.k_proj_out"][0].view(-1, NUM_KV_HEADS, HEAD_DIM).to(DEV)
    v_proj = orc["attn.v_proj_out"][0].view(-1, NUM_KV_HEADS, HEAD_DIM).to(DEV)
    wq = (1.0 + orc["attn.__weight.q_norm_weight"].float()).to(DEV)
    wk = (1.0 + orc["attn.__weight.k_norm_weight"].float()).to(DEV)
    T = q_split.shape[0]

    # positions: prefill is 0..T-1, decode is the single next position
    kc_ref = orc["attn.kv_cache_k_after_write"][0].to(DEV)   # [2, S, 256]
    vc_ref = orc["attn.kv_cache_v_after_write"][0].to(DEV)
    S = kc_ref.shape[1]
    assert S - T >= 0
    seq_len = S

    p = perm_idx(DEV)
    q_perm = q_split.index_select(-1, p)
    k_perm = k_proj.index_select(-1, p)
    wq_perm = wq.index_select(0, p).to(BF)
    wk_perm = wk.index_select(0, p).to(BF)

    cos, sin = build_cos_sin_table(torch.arange(MAX_SEQ_LEN), dtype=BF, device=DEV)
    cos, sin = cos.contiguous(), sin.contiguous()

    # ---------------------------------------------------------------- 1. k path
    # A fresh cache + the kernel's own write is the direct observation of
    # k_norm -> k_rope -> paged write (vllm-graph.md §2.2.5).
    kc_dev, vc_dev = empty_cache()
    if S > T:  # decode: prefill the context exactly as the runtime would
        ctx_k = kc_ref[:, : S - T, :].index_select(-1, p)     # permuted basis
        ctx_v = vc_ref[:, : S - T, :]
        for pos in range(S - T):
            kc_dev[pos // PAGE_SIZE, pos % PAGE_SIZE] = ctx_k[:, pos, :]
            vc_dev[pos // PAGE_SIZE, pos % PAGE_SIZE] = ctx_v[:, pos, :]
    out_ungated = torch.zeros(T, NUM_Q_HEADS * HEAD_DIM, dtype=BF, device=DEV)
    qkv_ungated = build_qkvg(q_perm, gate, k_perm, v_proj, gated=False)
    run_kernel(qkv_ungated, kc_dev, vc_dev, out_ungated, T, seq_len,
               max_tokens=args.max_tokens, gate=0, q_pass=args.q_pass,
               qn=wq_perm, kn=wk_perm, cos=cos, sin=sin)

    got_k = torch.stack([kc_dev[pos // PAGE_SIZE, pos % PAGE_SIZE]
                         for pos in range(seq_len)], dim=1)   # [2, S, 256]
    got_v = torch.stack([vc_dev[pos // PAGE_SIZE, pos % PAGE_SIZE]
                         for pos in range(seq_len)], dim=1)
    # back to the HF column order for comparison
    got_k_hf = torch.empty_like(got_k)
    got_k_hf[..., p] = got_k
    rows.append(cmp(f"[{tag}] k_norm+rope (new tokens, via cache write)",
                    got_k_hf[:, S - T:, :], kc_ref[:, S - T:, :],
                    "kernel rms_norm_sm100 + fused NeoX rope on the permuted basis"))

    # --- reference ladder: which RoPE rounding does the kernel implement? ---
    # k_ref computed in the PERMUTED basis exactly as the kernel sees it.
    k_new = k_perm[S - T:] if False else k_perm      # [T, 2, 256] pre-norm
    kn_ref = mpk_norm_bf16(k_new, wk_perm)
    pos_new = torch.arange(S - T, S, device=DEV)
    c = cos[pos_new].unsqueeze(1)                     # [T, 1, 256]
    s = sin[pos_new].unsqueeze(1)
    k_hf_round = rope_hf_rounding(kn_ref, c, s)
    k_fp32_fused = rope_fp32_fused(kn_ref, c, s)
    got_k_new_perm = got_k[:, S - T:, :].transpose(0, 1)   # [T, 2, 256] permuted
    rows.append(cmp(f"[{tag}] k rope vs REF fp32-fused rounding (MPK order)",
                    got_k_new_perm, k_fp32_fused, "expect bit-exact"))
    rows.append(cmp(f"[{tag}] k rope vs REF HF per-product bf16 rounding",
                    got_k_new_perm, k_hf_round, "counterfactual ordering"))
    # and the reference itself against the oracle, to separate 'my inputs are
    # wrong' from 'the kernel rounds differently'
    kh = torch.empty_like(k_hf_round); kh[..., p] = k_hf_round
    kf = torch.empty_like(k_fp32_fused); kf[..., p] = k_fp32_fused
    rows.append(cmp(f"[{tag}] REF(HF rounding) vs oracle k_rope",
                    kh.transpose(0, 1), kc_ref[:, S - T:, :], "input-construction check"))
    rows.append(cmp(f"[{tag}] REF(fp32-fused) vs oracle k_rope",
                    kf.transpose(0, 1), kc_ref[:, S - T:, :], ""))
    # ATTRIBUTION: is the residual the RoPE rounding or the NORM's fp32
    # association? MPK computes `val *= rms_rcp * w` (weight folded into the
    # scale first); HF computes `(x * rsqrt) * (1 + w)`. Swapping only the norm
    # -- keeping MPK's RoPE -- isolates which one moves the number.
    kn_hfnorm = gemma_norm_bf16(k_new, wk_perm)
    k_hfnorm_mpkrope = rope_fp32_fused(kn_hfnorm, c, s)
    khn = torch.empty_like(k_hfnorm_mpkrope); khn[..., p] = k_hfnorm_mpkrope
    rows.append(cmp(f"[{tag}] ATTRIB REF(HF norm order + MPK rope) vs oracle",
                    khn.transpose(0, 1), kc_ref[:, S - T:, :],
                    "isolates the norm's fp32 association from the rope"))
    kn_ref_hf = gemma_norm_bf16(k_new, wk_perm)
    k_all_hf = rope_hf_rounding(kn_ref_hf, c, s)
    kah = torch.empty_like(k_all_hf); kah[..., p] = k_all_hf
    rows.append(cmp(f"[{tag}] ATTRIB REF(HF norm + HF rope) vs oracle",
                    kah.transpose(0, 1), kc_ref[:, S - T:, :],
                    "full HF ordering -- the floor this basis can reach"))
    # ATTRIBUTION, last candidate: the NORM WEIGHT's dtype. MPK's task takes
    # `k_norm_weight_ptr` as `T const *` (bf16) for every model, so the folded
    # Gemma weight (1 + w) is rounded to bf16 before it is ever used; HF keeps
    # it in fp32. If that is the residual, using the fp32 weight here collapses
    # it -- and the gap is then a property of MPK's weight representation, not
    # of anything this issue introduced.
    kn_w32 = gemma_norm_bf16(k_new, wk.index_select(0, p))   # fp32 weight
    k_w32 = rope_hf_rounding(kn_w32, c, s)
    kw = torch.empty_like(k_w32); kw[..., p] = k_w32
    rows.append(cmp(f"[{tag}] ATTRIB REF(fp32 norm weight + HF rope) vs oracle",
                    kw.transpose(0, 1), kc_ref[:, S - T:, :],
                    "isolates the bf16 norm-weight rounding"))
    rows.append(cmp(f"[{tag}] kv cache k (full seq incl. context)",
                    got_k_hf, kc_ref, "§2.2.5 write-before-read"))
    rows.append(cmp(f"[{tag}] kv cache v (raw, no norm/rope)", got_v, vc_ref, ""))

    # ---------------------------------------------------------------- 2. q path
    # Replay the oracle's q heads through the k slot (same rms_norm_sm100 code,
    # NUM_HEAD=1) with the q_norm weight, on a FRESH prefill of length T so the
    # k positions coincide with the q positions. The cache then holds exactly
    # the normed+roped q.
    if args.q_path:
        q_rope_ref = orc["attn.q_rope"][0].to(DEV)  # [16, T, 256]
        got_q = torch.zeros(NUM_Q_HEADS, T, HEAD_DIM, dtype=BF, device=DEV)
        for pair in range(NUM_Q_HEADS // NUM_KV_HEADS):
            heads = [pair * NUM_KV_HEADS + i for i in range(NUM_KV_HEADS)]
            kslot = q_perm[:, heads, :].contiguous()
            kck, vck = empty_cache()
            o = torch.zeros(T, NUM_Q_HEADS * HEAD_DIM, dtype=BF, device=DEV)
            qkv_probe = build_qkvg(q_perm, gate, kslot, v_proj, gated=False)
            run_kernel(qkv_probe, kck, vck, o, T, T, max_tokens=args.max_tokens,
                       gate=0, q_pass=args.q_pass, qn=wq_perm, kn=wq_perm,
                       cos=cos, sin=sin)
            for i, h in enumerate(heads):
                got_q[h] = torch.stack(
                    [kck[pos // PAGE_SIZE, pos % PAGE_SIZE, i] for pos in range(T)])
        got_q_hf = torch.empty_like(got_q)
        got_q_hf[..., p] = got_q
        # decode's q sits at position S-1, but this probe replays it at 0..T-1;
        # only the prefill case has matching positions, so compare there.
        if S == T:
            rows.append(cmp(f"[{tag}] q_norm+rope (all 16 heads, via k-slot replay)",
                            got_q_hf, q_rope_ref,
                            "same rms_norm_sm100 path, NUM_HEAD=1"))

    # ---------------------------------------------------------------- 3. attn out
    core_ref = orc["attn.core_attn_out"][0].to(DEV)   # [T, 4096]
    rows.append(cmp(f"[{tag}] attention out (UNGATED)", out_ungated, core_ref,
                    "MPK online/flash softmax vs HF softmax(QK^T)@V"))

    # ---- isolate the attention math from norm+rope --------------------------
    # Feed the oracle's OWN post-rope q/k with qk_norm=False and rope=False.
    # The kernel then performs nothing but paged attention, so any residual
    # here is attributable to the softmax/accumulation order alone. No
    # permutation is involved: q.k is basis-independent and these tensors are
    # already in HF's basis.
    q_rope = orc["attn.q_rope"][0].transpose(0, 1).contiguous().to(DEV)   # [T,16,256]
    k_rope = orc["attn.k_rope"][0].transpose(0, 1).contiguous().to(DEV)   # [T,2,256]
    kc_b, vc_b = empty_cache()
    for pos in range(S - T):
        kc_b[pos // PAGE_SIZE, pos % PAGE_SIZE] = kc_ref[:, pos, :]
        vc_b[pos // PAGE_SIZE, pos % PAGE_SIZE] = vc_ref[:, pos, :]
    out_bypass = torch.zeros(T, NUM_Q_HEADS * HEAD_DIM, dtype=BF, device=DEV)
    qkv_bypass = build_qkvg(q_rope, gate, k_rope, v_proj, gated=False)
    qo, kvp, kvi, last = meta(T, seq_len)
    K.attention_qwen35(qkv_bypass, kc_b, vc_b, out_bypass, qo, kvp, kvi, last,
                       wq_perm, wk_perm, cos, sin, 1, NUM_QO_PER_KV,
                       NUM_KV_HEADS, HEAD_DIM, args.max_tokens, 0, args.q_pass,
                       False, False)
    rows.append(cmp(f"[{tag}] attention out from PRE-ROPED q/k (norm+rope bypassed)",
                    out_bypass, core_ref, "isolates the softmax/accumulation order"))

    # and a torch reference of the same attention, HF's eager formulation, so
    # the softmax residual can be attributed rather than merely observed
    kk = kc_ref.repeat_interleave(NUM_QO_PER_KV, dim=0)      # [16, S, 256]
    vv = vc_ref.repeat_interleave(NUM_QO_PER_KV, dim=0)
    qq = q_rope.transpose(0, 1)                              # [16, T, 256]
    logits = torch.matmul(qq.float(), kk.float().transpose(1, 2)) * (HEAD_DIM ** -0.5)
    qpos = torch.arange(S - T, S, device=DEV).unsqueeze(-1)
    kpos = torch.arange(S, device=DEV).unsqueeze(0)
    logits = logits.masked_fill(kpos > qpos, float("-inf"))
    attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(BF)
    ref_attn = torch.matmul(attn.float(), vv.float()).to(BF)  # [16, T, 256]
    ref_attn = ref_attn.transpose(0, 1).reshape(T, NUM_Q_HEADS * HEAD_DIM)
    rows.append(cmp(f"[{tag}] torch eager attention (same inputs) vs oracle",
                    ref_attn, core_ref, "upper bound on what any kernel can match"))
    rows.append(cmp(f"[{tag}] kernel (bypassed) vs torch eager attention",
                    out_bypass, ref_attn, ""))

    # ---------------------------------------------------------------- 4. gated
    kc_g, vc_g = empty_cache()
    if S > T:
        ctx_k = kc_ref[:, : S - T, :].index_select(-1, p)
        ctx_v = vc_ref[:, : S - T, :]
        for pos in range(S - T):
            kc_g[pos // PAGE_SIZE, pos % PAGE_SIZE] = ctx_k[:, pos, :]
            vc_g[pos // PAGE_SIZE, pos % PAGE_SIZE] = ctx_v[:, pos, :]
    out_gated = torch.zeros(T, NUM_Q_HEADS * HEAD_DIM, dtype=BF, device=DEV)
    qkv_gated = build_qkvg(q_perm, gate, k_perm, v_proj, gated=True)
    run_kernel(qkv_gated, kc_g, vc_g, out_gated, T, seq_len,
               max_tokens=args.max_tokens, gate=1, q_pass=args.q_pass,
               qn=wq_perm, kn=wk_perm, cos=cos, sin=sin)
    gated_ref = orc["attn.gate_sigmoid_mul_out"][0].to(DEV)
    rows.append(cmp(f"[{tag}] gated out (out * sigmoid(gate))", out_gated, gated_ref,
                    "QKVG slice + sigma-gate epilogue"))

    # The gate epilogue in ISOLATION: apply HF's own rounding order to the
    # kernel's own ungated output. Bit-exactness here proves the epilogue is
    # exactly `bf16(out * bf16(sigmoid(gate)))` with nothing else mixed in.
    flat_gate = gate.reshape(T, NUM_Q_HEADS * HEAD_DIM)
    epi_ref = (out_ungated.float()
               * torch.sigmoid(flat_gate.float()).to(BF).float()).to(BF)
    rows.append(cmp(f"[{tag}] gate epilogue isolated (kernel ungated x HF sigmoid)",
                    out_gated, epi_ref, "must be bit-exact"))
    if not torch.equal(out_gated, epi_ref):
        failures.append(f"[{tag}] gate epilogue is not bit-exact vs HF's rounding order")

    # COUNTERFACTUAL: fp32 sigmoid folded into one rounding must MISS.
    cf = (out_ungated.float() * torch.sigmoid(flat_gate.float())).to(BF)
    cf_rec = cmp(f"[{tag}] COUNTERFACTUAL gate w/ fp32-folded sigmoid",
                 out_gated, cf, "must NOT be bit-exact")
    rows.append(cf_rec)
    if cf_rec["bit_exact"]:
        failures.append(f"[{tag}] counterfactual (fp32-folded sigmoid) did NOT miss "
                        "-- the test cannot discriminate the two roundings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--q-pass", type=int, default=0)
    ap.add_argument("--no-q-path", dest="q_path", action="store_false")
    ap.add_argument("--out", default=os.path.join(_HERE, "oracle_result.json"))
    args = ap.parse_args()

    rows, failures = [], []
    for mode in ("prefill", "decode"):
        p = os.path.join(args.oracle_dir, mode, "manifest.json")
        if not os.path.exists(p):
            continue
        orc = Oracle(args.oracle_dir, mode)
        T = orc["attn.q_split"].shape[1]
        # decode is one token -> a single pass either way; prefill is 8 tokens,
        # which is 2 passes at Q_PASS_SIZE=4.
        a = argparse.Namespace(**vars(args))
        # The smem arena holds MAX_TOKENS query rows. A request with more
        # tokens than that MUST use the Q-loop -- that is the whole point of
        # §4.3 -- so pick the pass size automatically rather than letting a
        # default silently overflow q_smem.
        if T > args.max_tokens and args.q_pass == 0:
            a.q_pass = args.max_tokens
            print(f"[{mode}] T={T} > MAX_TOKENS={args.max_tokens} "
                  f"-> enabling the Q-loop with pass size {a.q_pass}")
        run_mode(orc, mode, a, rows, failures)

    show(rows)
    result = {"test": "M2-I6 attention vs HF oracle",
              "max_tokens": args.max_tokens, "q_pass_size": args.q_pass,
              "torch": torch.__version__, "rows": rows, "failures": failures}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote", args.out)
    if failures:
        print("\nFAILURES:")
        for f_ in failures:
            print(" -", f_)
        return 1
    print("\nall hard assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
