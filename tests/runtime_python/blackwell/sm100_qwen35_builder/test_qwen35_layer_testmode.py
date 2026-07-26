"""Single-layer END-TO-END gates: one real Qwen3.5 layer of each type, built by
the registry builder, run through the REAL megakernel, checked against the HF
oracle at every op boundary (M2-I8 acceptance 4).

This is the rung of the bring-up ladder between "each kernel matches the oracle"
(M2-I4..I7) and "40 layers match the reference" (M2-I9, AC-3): it is the first
time the BUILDER's wiring — buffer shapes, grid dims, task ordering, the weight
loader's transforms and the annotated graph's fork/join legality — is exercised
by the runtime on real checkpoint weights.

Two phases, one process each (`TaskRegister` is process-global):

  `gdn`   layer 0  = GDN mixer + MoE block   vs the oracle's `gdn.*` / `moe0.*`
  `attn`  layer 3  = full attention + MoE    vs the oracle's `attn.*` / `moe3.*`

Both replay the oracle's PREFILL pass (8 tokens from position 0, one request),
which is exactly what MPK test mode produces: `step == 0` so both GDN state
pools start zeroed (the branch `v1-architecture.md` §3.3 specifies) and the KV
cache starts empty, matching a fresh sequence in the oracle.

Run:
    python .../test_qwen35_layer_testmode.py --snapshot <hf> --dumps <oracle> \
        [--json out.json]
"""

import argparse
import json
import os
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "python"))

import mirage                                                     # noqa: E402
from mirage.mpk.models.qwen3_5.builder import Qwen35Builder       # noqa: E402
from mirage.mpk.models.qwen3_5.weight_loader import (             # noqa: E402
    Qwen35Config, Qwen35WeightLoader)
from mirage.mpk.persistent_kernel import PersistentKernel         # noqa: E402

DEV = "cuda"
BF = torch.bfloat16
GDN_LAYER = 0
ATTN_LAYER = 3
PAGE_SIZE = 64
MAX_SEQ = 64

# Per-boundary budgets. These are NOT arbitrary: MPK and HF run different fp8
# kernels over the same block-scaled weights, so every boundary carries the
# accumulated delta of that class (probe P10 measured a single dense GEMM of
# this family at 2-4e-3 frob-rel). The budget grows down the chain because the
# deltas compose; a WIRING error is orders of magnitude larger than any of
# these, which is what makes the test discriminating rather than a rubber stamp.
LIMITS = {
    "pre_norm": 8e-3,
    "proj": 2e-2,          # one fp8 GEMM off the oracle's own input
    "gdn_conv": 3e-2,
    "gdn_recurrent": 4e-2,
    "attn_core": 4e-2,
    "mixer_residual": 2e-2,
    "post_norm": 3e-2,      # downstream of the whole mixer chain; the norm
                            # divides by a small RMS and so amplifies it
    "attribution": 8e-3,    # op replayed on the megakernel's OWN buffers
    "router": 2e-2,
    "shared": 3e-2,
    # SwiGLU is badly conditioned in the relative-error sense: `silu(g) * u` has
    # a much smaller norm than either factor, so an input delta comes out
    # amplified (measured 7x on layer 0: gate_up 1.7e-2 -> silu_mul 1.2e-1).
    # These two rows therefore carry `post_norm` x that conditioning. They are
    # NOT the gate on the ops themselves - the bit-exact "own gate_up" /
    # "own quantized x" attribution rows below are, and the block output is the
    # gate on the composition.
    "shared_midchain": 1.5e-1,
    "block": 5e-2,
}


def load_dump(dumps, name, mode="prefill"):
    return torch.load(os.path.join(dumps, mode, "tensors", f"{name}.pt"),
                      map_location=DEV, weights_only=True)


class Report:
    def __init__(self):
        self.rows, self.failures = [], []

    def cmp(self, name, got, ref, limit, note=""):
        g, r = got.float().reshape(-1), ref.float().reshape(-1)
        assert g.numel() == r.numel(), f"{name}: {g.numel()} vs {r.numel()} elements"
        denom = r.norm().item()
        err = (g - r).norm().item() / denom if denom > 0 else 0.0
        mx = (g - r).abs().max().item()
        ok = err <= limit
        self.rows.append({"boundary": name, "frob_rel": err, "max_abs": mx,
                          "limit": limit, "ok": ok, "note": note})
        print(f"  {'OK  ' if ok else 'FAIL'} {name:32s} frob_rel={err:.3e} "
              f"max_abs={mx:.3e} (limit {limit:.1e}) {note}", flush=True)
        if not ok:
            self.failures.append(f"{name}: frob_rel {err:.3e} > {limit:.1e}")

    def exact(self, name, cond, note=""):
        self.rows.append({"boundary": name, "exact": bool(cond),
                          "ok": bool(cond), "note": note})
        print(f"  {'OK  ' if cond else 'FAIL'} {name:32s} EXACT {note}", flush=True)
        if not cond:
            self.failures.append(f"{name}: exact check failed")


def build_and_run(snapshot, layer_idx, hidden, out_dir):
    """Load that ONE layer's weights, wire it, compile, run one pass."""
    config = Qwen35Config.from_json(os.path.join(snapshot, "config.json"))
    loader = Qwen35WeightLoader(snapshot, config, device=DEV,
                                layer_filter=[layer_idx])
    weights = loader.load()

    tokens = hidden.shape[0]
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True, num_workers=num_workers,
        num_local_schedulers=num_schedulers, mpi_rank=0, world_size=1,
        max_num_batched_tokens=tokens, max_num_batched_requests=1,
        max_num_pages=2, page_size=PAGE_SIZE, max_seq_length=MAX_SEQ,
        meta_tensors={
            "tokens": torch.zeros((1, MAX_SEQ), dtype=torch.int64, device=DEV),
            "prompt_lengths": torch.tensor([tokens], dtype=torch.int32, device=DEV),
            "step": torch.zeros(1, dtype=torch.int32, device=DEV),
        },
    )
    pk = PersistentKernel(**params)
    assert pk.target_cc >= 100, "Qwen3.5 is Blackwell-only"
    b = Qwen35Builder(pk)
    b.build_layer_probe(weights, config, layer_idx, hidden)
    pk.compile(output_dir=out_dir)
    pk()
    torch.cuda.synchronize()
    step = int(pk.meta_tensors["step"][0].item())
    assert step == tokens, (
        f"prepare_next_batch scheduled {step} tokens, expected {tokens} - the "
        f"layer did not see the whole prefill chunk")
    return pk, b, config, weights


def _blockscale_dequant(w_fp8, scale, block=128):
    n, k = w_fp8.shape
    return (w_fp8.float().reshape(n // block, block, k // block, block)
            * scale.float()[:, None, :, None]).reshape(n, k)


def check_moe(rep, b, dumps, tag, i, h2_ref, c):
    """MoE-block boundaries, shared by both phases.

    Every op is checked TWICE where it matters: once against the oracle's own
    output (end-to-end fidelity) and once against a torch reference computed on
    the buffers THE MEGAKERNEL ITSELF produced (op fidelity). The pair is what
    separates "my input drifted upstream" from "this op is mis-wired" — the
    attribution discipline M2-I7's block test established.
    """
    buf = b.buffers
    w = b.weights
    x = buf[f"layer_{i}_post_norm"]
    rep.cmp("post_attn_norm", x,
            load_dump(dumps, f"{tag}.post_attention_layernorm.output").reshape(-1, c.hidden_size),
            LIMITS["post_norm"])

    moe = "moe0" if tag == "gdn" else "moe3"

    # ---- router -------------------------------------------------------
    # `moe_topk_softmax` ZEROES its input as it reads it
    # (persistent_kernel.py:1718), so the logits BUFFER is legitimately all
    # zeros after the run and cannot be compared. Check the router GEMM through
    # a torch replay on the megakernel's own post-norm instead.
    logits_buf = buf[f"layer_{i}_router_logits"]
    rep.exact("router logits buffer zeroed by topk",
              bool((logits_buf == 0).all()),
              "documented topk_softmax side effect, not a bug")
    logits_ref = load_dump(dumps, f"{moe}.router_logits").reshape(-1, c.num_experts)
    w_router = w[f"layer_{i}_router"]
    logits_replay = (x.float() @ w_router.float().t()).to(BF)
    rep.cmp("router GEMM (replay on MPK's x)", logits_replay, logits_ref,
            LIMITS["router"], "bf16 router, never quantized (vllm-graph 2.3.1)")

    routing = buf[f"layer_{i}_routing"]
    ids_ref = load_dump(dumps, f"{moe}.topk_ids").reshape(-1, c.num_experts_per_tok)
    got = torch.full_like(ids_ref, -1)
    nz = routing.nonzero()
    got[nz[:, 1], routing[nz[:, 0], nz[:, 1]].long() - 1] = nz[:, 0]

    # A top-8 selection is a DISCRETE function of a continuous input, so a
    # difference is only a defect if it happens at a margin the measured input
    # perturbation cannot explain. Same adjudication discipline as the AC-3
    # harness's logit ties (M1-I3 §6.5, probe P5).
    probs_ref = torch.softmax(logits_ref.float(), dim=-1)
    srt = torch.sort(probs_ref, dim=-1, descending=True).values
    margin = (srt[:, c.num_experts_per_tok - 1] - srt[:, c.num_experts_per_tok])
    perturb = (logits_replay.float() - logits_ref.float()).abs().max(dim=-1).values
    unexplained, near_tie = [], []
    for t in range(ids_ref.shape[0]):
        d = set(got[t].tolist()) ^ set(ids_ref[t].tolist())
        if not d:
            continue
        m, p = margin[t].item(), perturb[t].item()
        (near_tie if m <= p else unexplained).append(
            {"token": t, "sym_diff": sorted(d), "margin": m, "perturbation": p})
    rep.rows.append({"boundary": "router top-8 selection", "near_ties": near_tie,
                     "unexplained": unexplained, "ok": not unexplained,
                     "tokens": int(ids_ref.shape[0])})
    print(f"  {'OK  ' if not unexplained else 'FAIL'} "
          f"{'router top-8 selection':32s} {len(near_tie)} near-tie swap(s), "
          f"{len(unexplained)} unexplained of {ids_ref.shape[0]} tokens", flush=True)
    for r in near_tie:
        print(f"       token {r['token']}: {r['sym_diff']} "
              f"(prob margin {r['margin']:.3e} <= logit perturbation "
              f"{r['perturbation']:.3e})", flush=True)
    for r in unexplained:
        print(f"       UNEXPLAINED token {r['token']}: {r['sym_diff']} "
              f"(margin {r['margin']:.3e} > perturbation {r['perturbation']:.3e})",
              flush=True)
        rep.failures.append(f"router selection flip at token {r['token']} with "
                            f"margin {r['margin']:.3e} > perturbation "
                            f"{r['perturbation']:.3e}")

    rep.cmp("router weights", buf[f"layer_{i}_topk_w"],
            load_dump(dumps, f"{moe}.topk_renorm_weights").reshape(
                -1, c.num_experts_per_tok), LIMITS["router"],
            "HF rounds these to bf16 (round_weights_to_input_dtype)")

    # ---- shared expert -------------------------------------------------
    si = c.shared_expert_intermediate_size
    gu_ref = torch.cat([
        load_dump(dumps, f"{moe}.shared_gate_proj_out").reshape(-1, si),
        load_dump(dumps, f"{moe}.shared_up_proj_out").reshape(-1, si)], dim=1)
    smid = buf[f"layer_{i}_shared_mid"]
    # un-interleave MPK's [g0|u0|g1|u1|...] back into [gate | up]
    split = si // 128
    per = si // split
    gcols, ucols = [], []
    for g in range(split):
        base = g * 2 * per
        gcols += list(range(base, base + per))
        ucols += list(range(base + per, base + 2 * per))
    idx = torch.tensor(gcols + ucols, dtype=torch.long, device=DEV)
    rep.cmp("shared gate_up (un-interleaved)", smid.index_select(1, idx), gu_ref,
            LIMITS["shared"], "wrong interleave => O(1) here")

    xq_name = f"layer_{i}_shared_gate_up_xq"
    x_deq = None
    if xq_name in buf:
        xs = buf[f"layer_{i}_shared_gate_up_xs"]
        x_deq = (buf[xq_name].float().reshape(-1, c.hidden_size // 128, 128)
                 * xs.float().unsqueeze(-1)).reshape(-1, c.hidden_size)
        ref_gu = x_deq @ _blockscale_dequant(
            w[f"layer_{i}_shared_gate_up"],
            w[f"layer_{i}_shared_gate_up_scale"]).t()
        rep.cmp("shared gate_up (own quantized x)", smid, ref_gu.to(BF),
                LIMITS["attribution"],
                "ATTRIBUTION: op fidelity, input drift removed")

    sact = buf[f"layer_{i}_shared_act"]
    rep.cmp("shared silu_mul", sact,
            load_dump(dumps, f"{moe}.shared_silu_mul_out").reshape(-1, si),
            LIMITS["shared_midchain"], "input drift x SwiGLU conditioning")
    g_own, u_own = smid.index_select(1, idx[:si]), smid.index_select(1, idx[si:])
    rep.cmp("shared silu_mul (own gate_up)", sact,
            (torch.nn.functional.silu(g_own.float()).to(BF).float()
             * u_own.float()).to(BF), LIMITS["attribution"],
            "ATTRIBUTION: op fidelity, input drift removed")

    rep.cmp("shared down_proj", buf[f"layer_{i}_shared_out"],
            load_dump(dumps, f"{moe}.shared_down_proj_out").reshape(-1, c.hidden_size),
            LIMITS["shared_midchain"],
            "fp8 preserved-scale path, NOT bf16-dequant (M2-I7 flag)")
    rep.cmp("shared gate + residual", buf[f"layer_{i}_r_prime"],
            (h2_ref.float()
             + load_dump(dumps, f"{moe}.shared_output_gated").reshape(
                 -1, c.hidden_size).float()).to(BF), LIMITS["shared"])
    rep.cmp("BLOCK OUT (h2 + moe)", buf[f"layer_{i}_moe_out"],
            (h2_ref.float()
             + load_dump(dumps, f"{moe}.combined_output").reshape(
                 -1, c.hidden_size).float()).to(BF), LIMITS["block"],
            "deferred-residual form folded into one add (vllm-graph 1.3)")


def phase_gdn(snapshot, dumps, json_out):
    rep = Report()
    i = GDN_LAYER
    h = load_dump(dumps, "gdn.input_layernorm.input").reshape(-1, 2048).contiguous()
    pk, b, c, _ = build_and_run(snapshot, i, h,
                                os.path.join(_HERE, "test_output_q35_gdn"))
    buf = b.buffers
    print(f"  tokens={h.shape[0]} layer={i} (linear_attention)", flush=True)

    rep.cmp("pre_norm", buf[f"layer_{i}_pre_norm"],
            load_dump(dumps, "gdn.input_layernorm.output").reshape(-1, c.hidden_size),
            LIMITS["pre_norm"])
    rep.cmp("in_proj_qkv", buf[f"layer_{i}_gdn_qkv"],
            load_dump(dumps, "gdn.qkv_proj_out").reshape(-1, c.conv_dim),
            LIMITS["proj"])
    rep.cmp("in_proj_z", buf[f"layer_{i}_gdn_z"],
            load_dump(dumps, "gdn.z_proj_out").reshape(-1, c.gdn_z_dim),
            LIMITS["proj"])
    nv = c.linear_num_value_heads
    ba_ref = torch.cat([load_dump(dumps, "gdn.b_proj_out").reshape(-1, nv),
                        load_dump(dumps, "gdn.a_proj_out").reshape(-1, nv)], dim=1)
    rep.cmp("in_proj_ba (bf16)", buf[f"layer_{i}_gdn_ba"], ba_ref, LIMITS["proj"])

    # HF dumps the conv output channel-major [B, D, T]
    conv_ref = load_dump(dumps, "gdn.conv_out")[0].transpose(0, 1).contiguous()
    rep.cmp("gdn_conv1d (task 234)", buf[f"layer_{i}_gdn_qkv_c"], conv_ref,
            LIMITS["gdn_conv"])
    rep.cmp("gdn_recurrent + gated norm (237)", buf[f"layer_{i}_gdn_out"],
            load_dump(dumps, "gdn.gated_norm_out").reshape(-1, c.gdn_z_dim),
            LIMITS["gdn_recurrent"])
    # the recurrent state the kernel leaves behind must be the oracle's
    st_ref = load_dump(dumps, "gdn.core_state_after")[0].transpose(-1, -2)
    rep.cmp("recurrent state after", b.recurrent_state[b._gdn_slot[i], 0], st_ref,
            LIMITS["gdn_recurrent"], "per-slot fp32 pool (v1-arch 3.1)")
    # conv state: MPK's pool is HF's [..., 1:] transposed (M2-I4)
    cs_ref = load_dump(dumps, "gdn.conv_state_after")[0][:, 1:].transpose(0, 1)
    rep.cmp("conv state after", b.conv_state[b._gdn_slot[i], 0], cs_ref,
            LIMITS["gdn_conv"])

    h2_ref = load_dump(dumps, "gdn.post_attention_layernorm.input").reshape(
        -1, c.hidden_size)
    rep.cmp("out_proj + residual", buf[f"layer_{i}_attn_resid"], h2_ref,
            LIMITS["mixer_residual"])
    check_moe(rep, b, dumps, "gdn", i, h2_ref, c)

    pk.finalize()
    return finish(rep, json_out, "GDN")


def phase_attn(snapshot, dumps, json_out):
    rep = Report()
    i = ATTN_LAYER
    h = load_dump(dumps, "attn.input_layernorm.input").reshape(-1, 2048).contiguous()
    pk, b, c, _ = build_and_run(snapshot, i, h,
                                os.path.join(_HERE, "test_output_q35_attn"))
    buf = b.buffers
    print(f"  tokens={h.shape[0]} layer={i} (full_attention)", flush=True)

    rep.cmp("pre_norm", buf[f"layer_{i}_pre_norm"],
            load_dump(dumps, "attn.input_layernorm.output").reshape(-1, c.hidden_size),
            LIMITS["pre_norm"])

    # the packed QKVG row, unpacked with the kernel's own addressing
    hd, nkv = c.head_dim, c.num_key_value_heads
    qo = c.num_attention_heads // nkv
    gw = qo * 2 * hd + 2 * hd
    qkvg = buf[f"layer_{i}_qkvg"]
    from mirage.mpk.models.qwen3_5.rope import rope_permutation_src
    p = torch.as_tensor(rope_permutation_src(hd, c.rotary_dim), dtype=torch.long,
                        device=DEV)
    inv = torch.argsort(p)
    q_ref = load_dump(dumps, "attn.q_split").reshape(-1, c.num_attention_heads, hd)
    g_ref = load_dump(dumps, "attn.gate_split").reshape(-1, c.num_attention_heads, hd)
    k_ref = load_dump(dumps, "attn.k_proj_out").reshape(-1, nkv, hd)
    v_ref = load_dump(dumps, "attn.v_proj_out").reshape(-1, nkv, hd)
    qs, gs, ks, vs = [], [], [], []
    for g in range(nkv):
        base = g * gw
        for hh in range(qo):
            s = base + hh * 2 * hd
            qs.append(qkvg[:, s:s + hd].index_select(1, inv))
            gs.append(qkvg[:, s + hd:s + 2 * hd])
        kb = base + qo * 2 * hd
        ks.append(qkvg[:, kb:kb + hd].index_select(1, inv))
        vs.append(qkvg[:, kb + hd:kb + 2 * hd])
    rep.cmp("qkvg -> q (unpermuted)", torch.stack(qs, 1), q_ref, LIMITS["proj"],
            "layout + RoPE permutation + fp8 rows in one check")
    rep.cmp("qkvg -> gate", torch.stack(gs, 1), g_ref, LIMITS["proj"])
    rep.cmp("qkvg -> k (unpermuted)", torch.stack(ks, 1), k_ref, LIMITS["proj"])
    rep.cmp("qkvg -> v", torch.stack(vs, 1), v_ref, LIMITS["proj"])

    rep.cmp("attention + gate epilogue", buf[f"layer_{i}_attn_out"],
            load_dump(dumps, "attn.gate_sigmoid_mul_out").reshape(
                -1, c.num_attention_heads * hd), LIMITS["attn_core"])
    # the KV cache the kernel wrote, back in the HF basis
    kc = b.k_cache[b._attn_slot[i], 0, :q_ref.shape[0]]        # [T, nkv, hd]
    rep.cmp("kv cache k (permuted basis)", kc.index_select(2, inv),
            load_dump(dumps, "attn.kv_cache_k_after_write")[0].transpose(0, 1),
            LIMITS["attn_core"], "write-before-read ordering (vllm-graph 2.2.5)")
    rep.cmp("kv cache v", b.v_cache[b._attn_slot[i], 0, :q_ref.shape[0]],
            load_dump(dumps, "attn.kv_cache_v_after_write")[0].transpose(0, 1),
            LIMITS["attn_core"])

    h2_ref = load_dump(dumps, "attn.post_attention_layernorm.input").reshape(
        -1, c.hidden_size)
    rep.cmp("o_proj + residual", buf[f"layer_{i}_attn_resid"], h2_ref,
            LIMITS["mixer_residual"])
    check_moe(rep, b, dumps, "attn", i, h2_ref, c)

    pk.finalize()
    return finish(rep, json_out, "ATTN")


def finish(rep, json_out, tag):
    if json_out:
        with open(json_out, "w") as f:
            json.dump({"phase": tag, "rows": rep.rows,
                       "failures": rep.failures}, f, indent=1)
    if rep.failures:
        print(f"\n{tag} FAILURES:")
        for f in rep.failures:
            print(" -", f)
        return 1
    print(f"\n  RESULT {tag} single-layer end-to-end PASSED "
          f"({len(rep.rows)} boundaries)")
    return 0


PHASES = {"gdn": phase_gdn, "attn": phase_attn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--phase", default=None, choices=list(PHASES))
    args = ap.parse_args()

    if args.phase:
        return PHASES[args.phase](args.snapshot, args.dumps, args.json)

    failures = []
    for name in ("gdn", "attn"):
        print(f"\n== single-layer end-to-end: {name} ==", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--phase", name,
               "--snapshot", args.snapshot, "--dumps", args.dumps]
        if args.json:
            cmd += ["--json", args.json.replace(".json", f".{name}.json")]
        p = subprocess.run(cmd, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                           capture_output=True, text=True)
        print(p.stdout, flush=True)
        if p.returncode != 0:
            failures.append(f"phase {name} failed (rc={p.returncode})")
            print("--- stderr tail ---\n" +
                  "\n".join(p.stderr.splitlines()[-20:]), flush=True)
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nSINGLE-LAYER END-TO-END GATES PASSED (both layer types)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
