"""Per-transform unit checks for the Qwen3.5 weight loader, against the oracle.

Each §2.0 / §5.2 transform is checked in isolation on the REAL checkpoint
tensors the M2-I3 oracle dumped, so a wiring bug shows up here rather than as a
diffuse end-to-end error. Where the oracle also dumped the op's OUTPUT, the
transform is checked by reproducing that output; where it did not, the check is
structural (exact equality against the source tensor).

It also RESOLVES one design question this issue found (see
`transforms.permute_fp8_blockscale_rows`): the partial-RoPE column permutation
provably cannot be applied to a block-scaled fp8 weight by permuting the scale
alongside it, so the loader has two candidate representations for the attention
q/k projections. Phase `qk` measures both against the oracle's own
`q_proj_out` / `k_proj_out`, next to the no-permutation floor, and prints the
table the default is pinned from.

Run (on the GPU box, checkpoint + oracle dumps present):
    python .../test_loader_transforms.py \
        --snapshot <hf snapshot> --dumps <oracle dump dir> [--json out.json]
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "python"))

from mirage.mpk.models.qwen3_5 import transforms as T          # noqa: E402
from mirage.mpk.models.qwen3_5.rope import rope_permutation_src  # noqa: E402
from mirage.mpk.models.qwen3_5.weight_loader import Qwen35Config  # noqa: E402

DEV = "cuda"
BF = torch.bfloat16
BLOCK = 128
FP8_MAX = 448.0
EPS_Q = 1e-10
EPS = 1e-6

ROWS = []
FAILURES = []


def record(name, got, ref, limit, note=""):
    g, r = got.float(), ref.float()
    denom = r.norm().item()
    err = (g - r).norm().item() / denom if denom > 0 else 0.0
    mx = (g - r).abs().max().item()
    ok = err <= limit
    ROWS.append({"check": name, "frob_rel": err, "max_abs": mx,
                 "limit": limit, "ok": ok, "note": note})
    print(f"  {'OK  ' if ok else 'FAIL'} {name:44s} frob_rel={err:.3e} "
          f"max_abs={mx:.3e} (limit {limit:.1e}) {note}")
    if not ok:
        FAILURES.append(f"{name}: frob_rel {err:.3e} > {limit:.1e}")
    return err


def record_exact(name, cond, note=""):
    ROWS.append({"check": name, "exact": bool(cond), "ok": bool(cond), "note": note})
    print(f"  {'OK  ' if cond else 'FAIL'} {name:44s} EXACT {note}")
    if not cond:
        FAILURES.append(f"{name}: exact-equality check failed")


def load_dump(dumps, mode, name):
    return torch.load(os.path.join(dumps, mode, "tensors", f"{name}.pt"),
                      map_location=DEV, weights_only=True)


# --------------------------------------------------------------------------
# torch models of the MPK kernels the loader feeds
# --------------------------------------------------------------------------
def mpk_rmsnorm(x, w_folded, eps=EPS):
    """`norm_sm100.cuh`: fp32 accumulate, weight folded into the scale, one
    rounding at the store."""
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (rms * w_folded.float())).to(x.dtype)


def quantize_act(x):
    """`quantize_fp8_f32scale_sm100` == vLLM's per-token-group primitive:
    group 128, absmax/448 with a 1e-10 floor, DIVISION, clamp then cast."""
    shape = x.shape
    k = shape[-1]
    xf = x.float().reshape(-1, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS_Q)
    scale = torch.div(absmax, torch.tensor(FP8_MAX, dtype=torch.float32,
                                           device=x.device))
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return (q.reshape(shape).to(torch.float8_e4m3fn),
            scale.reshape(*shape[:-1], k // BLOCK).float())


def blockscale_gemm(xq, xs, w_fp8, w_scale):
    """`linear_fp8_blockscale_sm100`: unscaled fp8 MMA per 128-K tile, fp32
    promotion, `a_scale * b_scale` folded in at the tile boundary."""
    m, k = xq.shape
    n = w_fp8.shape[0]
    out = torch.zeros(m, n, dtype=torch.float32, device=xq.device)
    for kt in range(k // BLOCK):
        a = xq[:, kt * BLOCK:(kt + 1) * BLOCK].float()
        b = w_fp8[:, kt * BLOCK:(kt + 1) * BLOCK].float()
        part = a @ b.t()
        out += part * xs[:, kt:kt + 1] * w_scale[:, kt].repeat_interleave(BLOCK)[None, :]
    return out.to(BF)


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------
def phase_norms(dumps):
    print("\n[1] Gemma (1+w) fold, and the ONE norm that must NOT be folded")
    for tag, layer in (("gdn", "gdn"), ("attn", "attn")):
        x = load_dump(dumps, "decode", f"{layer}.input_layernorm.input")
        ref = load_dump(dumps, "decode", f"{layer}.input_layernorm.output")
        w = load_dump(dumps, "decode", f"{layer}.input_layernorm.__weight.weight")
        folded = T.gemma_fold(w)
        record_exact(f"{tag} gemma_fold dtype/shape",
                     folded.dtype == BF and folded.shape == w.shape)
        record(f"{tag} input_layernorm via gemma_fold",
               mpk_rmsnorm(x, folded), ref, 6e-3,
               "bf16 norm-weight store is MPK's representation (I6)")
        # COUNTERFACTUAL: no fold must miss by roughly the whole weight
        cf = mpk_rmsnorm(x, w.to(BF))
        cf_err = (cf.float() - ref.float()).norm().item() / ref.float().norm().item()
        ok = cf_err > 0.5
        print(f"  {'OK  ' if ok else 'FAIL'} {tag + ' COUNTERFACTUAL no-fold':44s} "
              f"frob_rel={cf_err:.3e} (must be > 5e-1)")
        ROWS.append({"check": f"{tag} counterfactual no-fold", "frob_rel": cf_err,
                     "ok": ok, "note": "must MISS"})
        if not ok:
            FAILURES.append(f"{tag} no-fold counterfactual did not miss")

    # GDN gated norm: ones-init, fp32, NO +1
    wg = load_dump(dumps, "decode", "gdn.__weight.norm_weight")
    kept = T.no_fold_norm(wg)
    record_exact("gdn norm kept fp32, unfolded",
                 kept.dtype == torch.float32 and torch.equal(kept, wg.float()))
    o = load_dump(dumps, "decode", "gdn.core_attn_out").reshape(-1, 128)
    z = load_dump(dumps, "decode", "gdn.z_proj_out").reshape(-1, 128)
    gated = load_dump(dumps, "decode", "gdn.gated_norm_out").reshape(-1, 128)
    ob = o.to(BF).float()
    xh = (ob * torch.rsqrt(ob.pow(2).mean(-1, keepdim=True) + EPS)).to(BF).float()
    good = ((xh * kept) * torch.nn.functional.silu(z.float())).to(BF)
    bad = ((xh * (1.0 + kept)) * torch.nn.functional.silu(z.float())).to(BF)
    record("gdn gated norm (no fold)", good, gated, 5e-3)
    bad_err = (bad.float() - gated.float()).norm().item() / gated.float().norm().item()
    ok = bad_err > 0.3
    print(f"  {'OK  ' if ok else 'FAIL'} {'gdn COUNTERFACTUAL folded gated norm':44s} "
          f"frob_rel={bad_err:.3e} (must be > 3e-1)")
    ROWS.append({"check": "gdn counterfactual folded gated norm",
                 "frob_rel": bad_err, "ok": ok, "note": "must MISS"})
    if not ok:
        FAILURES.append("folding the GDN gated norm did not miss")


def phase_gdn_pack(dumps):
    print("\n[2] GDN packing: A_log/dt_bias, conv1d view, [b|a] concat")
    a_log = load_dump(dumps, "decode", "gdn.__weight.A_log")
    dt = load_dump(dumps, "decode", "gdn.__weight.dt_bias")
    packed = T.pack_alog_dtbias(a_log, dt)
    record_exact("alog_dtbias [2,H] fp32",
                 packed.shape == (2, a_log.shape[0])
                 and packed.dtype == torch.float32
                 and torch.equal(packed[0], a_log.float())
                 and torch.equal(packed[1], dt.float()))

    cw = load_dump(dumps, "decode", "gdn.__weight.conv1d_weight")
    got = T.conv1d_weight(cw)
    record_exact("conv1d [C,1,4] -> [C,4] bf16",
                 got.shape == (cw.shape[0], cw.shape[-1]) and got.dtype == BF
                 and torch.equal(got.float(), cw.reshape(got.shape).float()))

    b = load_dump(dumps, "decode", "gdn.__weight.in_proj_b")
    a = load_dump(dumps, "decode", "gdn.__weight.in_proj_a")
    ba = T.concat_ba(b, a)
    nv = b.shape[0]
    record_exact("in_proj_ba = [b | a], plain order",
                 torch.equal(ba[:nv].float(), b.float())
                 and torch.equal(ba[nv:].float(), a.float()))
    # reproduce the two projection outputs through the fused weight
    x = load_dump(dumps, "decode", "gdn.input_layernorm.output").reshape(-1, ba.shape[1])
    fused = (x.float() @ ba.float().t()).to(BF)
    b_ref = load_dump(dumps, "decode", "gdn.b_proj_out").reshape(-1, nv)
    a_ref = load_dump(dumps, "decode", "gdn.a_proj_out").reshape(-1, nv)
    record("in_proj_ba fused vs b_proj_out", fused[:, :nv], b_ref, 8e-3)
    record("in_proj_ba fused vs a_proj_out", fused[:, nv:], a_ref, 8e-3)


def phase_qk(dumps, config):
    """The decision phase: how to represent the RoPE-permuted q/k projections."""
    print("\n[3] partial-RoPE permutation vs the fp8 block-scale layout")
    c = config
    x = load_dump(dumps, "decode", "attn.input_layernorm.output").reshape(-1, c.hidden_size)
    q_ref = load_dump(dumps, "decode", "attn.q_proj_out").reshape(x.shape[0], -1)
    k_ref = load_dump(dumps, "decode", "attn.k_proj_out").reshape(x.shape[0], -1)
    v_ref = load_dump(dumps, "decode", "attn.v_proj_out").reshape(x.shape[0], -1)
    wq = load_dump(dumps, "decode", "attn.__weight.q_proj")
    wk = load_dump(dumps, "decode", "attn.__weight.k_proj")
    wv = load_dump(dumps, "decode", "attn.__weight.v_proj")
    sq = load_dump(dumps, "decode", "attn.__weight.q_proj_scale_inv").float()
    sk = load_dump(dumps, "decode", "attn.__weight.k_proj_scale_inv").float()
    sv = load_dump(dumps, "decode", "attn.__weight.v_proj_scale_inv").float()

    q_map = T.q_proj_dest_from_src(c.num_attention_heads, c.head_dim, c.rotary_dim, DEV)
    k_map = T.k_proj_dest_from_src(c.num_key_value_heads, c.head_dim, c.rotary_dim, DEV)
    inv_q = torch.argsort(q_map)
    inv_k = torch.argsort(k_map)

    # THE STRUCTURAL FINDING, asserted rather than asserted-in-prose: the
    # permutation moves rows across 128-row scale-block boundaries, so the
    # shipped [N/128, K/128] scale cannot simply follow the rows.
    crossing = int((torch.div(q_map, BLOCK, rounding_mode="floor")
                    != torch.arange(q_map.numel(), device=DEV) // BLOCK).sum())
    print(f"  q_proj rows that change 128-row scale block: {crossing} "
          f"of {q_map.numel()}")
    record_exact("permutation DOES cross scale blocks (the collision)",
                 crossing > 0, "if this ever becomes 0 the fp8 rescale is dead code")

    xq, xs = quantize_act(x)
    results = {}

    # floor: no permutation at all, exact shipped scales
    results["floor_no_permute"] = (
        record("q floor: fp8, unpermuted, exact scales",
               blockscale_gemm(xq, xs, wq, sq), q_ref, 8e-3, "the achievable floor"),
        record("k floor: fp8, unpermuted, exact scales",
               blockscale_gemm(xq, xs, wk, sk), k_ref, 8e-3, ""))

    # candidate A: permute the fp8 rows, rescale only what changes block
    qp, qsp, q_stats = T.permute_fp8_blockscale_rows(wq, sq, q_map)
    kp, ksp, k_stats = T.permute_fp8_blockscale_rows(wk, sk, k_map)
    print(f"  fp8 permute stats  q: rescaled {q_stats['rescaled_fraction']:.3f} of "
          f"(row,kblock) entries, dequant frob_rel {q_stats['frob_rel_vs_exact_permute']:.3e}")
    print(f"  fp8 permute stats  k: rescaled {k_stats['rescaled_fraction']:.3f} of "
          f"(row,kblock) entries, dequant frob_rel {k_stats['frob_rel_vs_exact_permute']:.3e}")
    results["fp8_permute"] = (
        record("q fp8-permute (unpermuted back)",
               blockscale_gemm(xq, xs, qp, qsp).index_select(1, inv_q), q_ref, 2e-2),
        record("k fp8-permute (unpermuted back)",
               blockscale_gemm(xq, xs, kp, ksp).index_select(1, inv_k), k_ref, 2e-2))

    # candidate B: exact dequant to bf16, permute, plain bf16 GEMM
    qb = T.permute_bf16_rows(wq, sq, q_map)
    kb = T.permute_bf16_rows(wk, sk, k_map)
    results["bf16_permute"] = (
        record("q bf16-dequant-permute (unpermuted back)",
               (x.float() @ qb.float().t()).to(BF).index_select(1, inv_q), q_ref, 5e-2),
        record("k bf16-dequant-permute (unpermuted back)",
               (x.float() @ kb.float().t()).to(BF).index_select(1, inv_k), k_ref, 5e-2))

    print("\n  DECISION TABLE (frob_rel vs the HF oracle's own projection output)")
    print(f"    {'representation':22s} {'q':>10s} {'k':>10s}")
    for name, (eq, ek) in results.items():
        print(f"    {name:22s} {eq:10.3e} {ek:10.3e}")
    winner = min(("fp8_permute", "bf16_permute"),
                 key=lambda n: results[n][0] + results[n][1])
    print(f"    -> closer to the oracle: {winner}")
    ROWS.append({"check": "qk representation decision", "ok": True,
                 "table": {k: {"q": v[0], "k": v[1]} for k, v in results.items()},
                 "winner": winner,
                 "q_stats": q_stats, "k_stats": k_stats})

    # ---- the fused QKVG layout the attention task actually reads ----------
    print("\n[4] fused QKVG layout (kv-group interleave)")
    fused = T.fuse_qkvg(qp, kp, wv, c.num_key_value_heads)
    fused_s = T.fuse_qkvg_scale(qsp, ksp, sv, c.num_key_value_heads)
    record_exact("fused qkvg width == MPK's packed row",
                 fused.shape == (c.qkvg_dim, c.hidden_size)
                 and fused_s.shape == (c.qkvg_dim // BLOCK, c.hidden_size // BLOCK))
    out = blockscale_gemm(xq, xs, fused, fused_s)
    # unpack with the addressing the kernel uses (test_attention_qwen35_testmode
    # reference, persistent_kernel.py:986-991)
    hd, nkv = c.head_dim, c.num_key_value_heads
    qo_per_kv = c.num_attention_heads // nkv
    group_w = qo_per_kv * 2 * hd + 2 * hd
    q_cols, k_cols, v_cols = [], [], []
    for g in range(nkv):
        base = g * group_w
        for h in range(qo_per_kv):
            s = base + h * 2 * hd
            q_cols += list(range(s, s + 2 * hd))        # [q|gate] pair, as shipped
        kb = base + qo_per_kv * 2 * hd
        k_cols += list(range(kb, kb + hd))
        v_cols += list(range(kb + hd, kb + 2 * hd))
    idx = lambda cols: torch.tensor(cols, dtype=torch.long, device=DEV)  # noqa: E731
    record("fused->q|gate slice vs direct q_proj",
           out.index_select(1, idx(q_cols)).index_select(1, inv_q), q_ref, 2e-2,
           "layout check: a wrong interleave shows up as O(1) error")
    record("fused->k slice vs direct k_proj",
           out.index_select(1, idx(k_cols)).index_select(1, inv_k), k_ref, 2e-2)
    record("fused->v slice vs direct v_proj",
           out.index_select(1, idx(v_cols)), v_ref, 2e-2)

    # q/k norm weights: Gemma fold, then the same permutation, then bf16
    for tag, wname in (("q", "q_norm_weight"), ("k", "k_norm_weight")):
        w = load_dump(dumps, "decode", f"attn.__weight.{wname}")
        got = T.permute_head_dim_norm(T.gemma_fold(w), c.head_dim, c.rotary_dim)
        want = (1.0 + w.float()).index_select(
            0, torch.as_tensor(rope_permutation_src(c.head_dim, c.rotary_dim),
                               dtype=torch.long, device=DEV)).to(BF)
        record_exact(f"{tag}_norm fold->permute->bf16 (I6 convention)",
                     torch.equal(got, want))


def phase_moe(dumps, config):
    print("\n[5] MoE stacking: [gate;up] packing and UNEXPANDED scales")
    c = config
    names = sorted({f.split(".")[1] for f in os.listdir(
        os.path.join(dumps, "decode", "tensors")) if f.startswith("moe0.expert_")})
    assert names, "no per-expert oracle dumps found"
    gate_list, up_list, gs, us, downs, ds = [], [], [], [], [], []
    for n in names[:4]:
        gu = load_dump(dumps, "decode", f"moe0.{n}.__weight.gate_up_proj")
        gus = load_dump(dumps, "decode", f"moe0.{n}.__weight.gate_up_proj_scale_inv")
        dn = load_dump(dumps, "decode", f"moe0.{n}.__weight.down_proj")
        dns = load_dump(dumps, "decode", f"moe0.{n}.__weight.down_proj_scale_inv")
        inter = gu.shape[0] // 2
        gate_list.append(gu[:inter])
        up_list.append(gu[inter:])
        gs.append(gus[:inter // BLOCK].float())
        us.append(gus[inter // BLOCK:].float())
        downs.append(dn)
        ds.append(dns.float())
    w13 = T.stack_expert_w13(gate_list, up_list)
    w13s = T.stack_expert_w13_scale(gs, us)
    w2 = T.stack_expert_w2(downs)
    w2s = T.stack_expert_w2_scale(ds)
    inter = c.moe_intermediate_size
    record_exact("w13 [E,2*inter,H] with [gate;up] packing",
                 w13.shape == (len(names[:4]), 2 * inter, c.hidden_size)
                 and torch.equal(w13[0, :inter].float(), gate_list[0].float())
                 and torch.equal(w13[0, inter:].float(), up_list[0].float()))
    record_exact("w13 scale UNEXPANDED [E,2*inter/128,H/128]",
                 w13s.shape == (len(names[:4]), 2 * inter // BLOCK,
                                c.hidden_size // BLOCK)
                 and w13s.dtype == torch.float32)
    record_exact("w2 [E,H,inter] + scale [E,H/128,inter/128]",
                 w2.shape == (len(names[:4]), c.hidden_size, inter)
                 and w2s.shape == (len(names[:4]), c.hidden_size // BLOCK,
                                   inter // BLOCK))
    # the whole point of "unexpanded": the per-row repeat_interleave form the
    # rejected 241/242 predecessor wanted is 128x bigger and a different tensor
    record_exact("scale is NOT the per-row expansion",
                 w13s.shape[1] * BLOCK == w13.shape[1])

    print("\n[6] shared-expert gate_up interleave for silu_mul")
    gu = load_dump(dumps, "decode", "moe0.__weight.shared_expert.gate_proj")
    up = load_dump(dumps, "decode", "moe0.__weight.shared_expert.up_proj")
    split = c.shared_expert_intermediate_size // BLOCK
    fused = T.fuse_gate_up(gu, up, split)
    per = gu.shape[0] // split
    ok = True
    for g in range(split):
        base = g * 2 * per
        ok &= torch.equal(fused[base:base + per].float(),
                          gu[g * per:(g + 1) * per].float())
        ok &= torch.equal(fused[base + per:base + 2 * per].float(),
                          up[g * per:(g + 1) * per].float())
    record_exact("shared gate_up chunks interleave as [gate|up]", ok,
                 f"split={split}, {per} rows per chunk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    config = Qwen35Config.from_json(os.path.join(args.snapshot, "config.json"))

    phase_norms(args.dumps)
    phase_gdn_pack(args.dumps)
    phase_qk(args.dumps, config)
    phase_moe(args.dumps, config)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": ROWS, "failures": FAILURES}, f, indent=1)
        print(f"\nwrote {args.json}")

    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print("\nALL LOADER TRANSFORM CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
