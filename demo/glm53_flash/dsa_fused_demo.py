"""GLM-5.3-Flash DSA attention with the sublayer boundary inside the layer, as Kimi K3's fused layers (#811: o_proj, the TP
all-reduce, the residual and the norm in one kernel): one decode step of T = 8 requests on --tp GPUs in one process (static_host's
multicast exchange region), one kernel per GPU.

  input    the residual streams R [8, 4 x 4096] bf16 (random) and the attention's mHC pre before the layer (as SGLang's
           hc_attn_pre: x_attn = RMSNorm(sum_i pre_i R_i) input_layernorm, h_post, h_res): what SGLang hands the attention
  layer    dsa_layer.build_graph: the DSA layer (GPU r: TP shard r, 64 / tp heads; the caches and the indexer
           replicated) -> o_proj (this GPU's partial sum) -> allreduce_send (bf16, to every GPU) -> hc_post_mix (the sum
           over the GPUs in GPU order; R'_j = sum_i comb[i][j] R_i + post_j out; the FFN's mHC mixes) -> hc_pre (the FFN's
           mHC pre: x_ffn = RMSNorm(sum_i pre_i R'_i) post_attention_layernorm, the FFN's h_post / h_res finished)
  outputs  R', x_ffn, h_post_ffn, h_res_ffn on every GPU: what SGLang's attn_boundary.finish + ffn_boundary.prepare hand
           the FFN (MHCState: the streams, x, h_post, h_res)
  check    the nodes before o_proj (a decode step, node_checks): the indexer node by node (dsa_layer's references: the
           pooled keys and tail rows written, the scores from the kernel's q_c, the top-k against the kernel's own scores
           and the reference's selection) and MLA on the kernel's selection (the latent rows written, q, o_lat, v); the
           boundary node by node, from the kernel's own per-GPU partial sums (t["o"]): R' (bit-equal share, relRMS),
           x_ffn, h_post, h_res against SGLang's mHC formulas (mhc.py _mhc_post_torch / _mhc_pre_torch, the norm as
           RMSNorm.forward_native); every GPU's outputs equal; o_proj per GPU from the kernel's v; end to end: every GPU's
           DSA reference (dsa_mla.reference) on the kernel's own selection, then the boundary
  time     --reps launches, 512 MB read before each (L2 emptied): the span over the GPUs
  --verify N  the layer of a speculative verify step (SGLang's DSA target-verify: one request's N draft tokens as rows t <
           N at positions seq_lens[0] - 1 + t, sharing its page tables and request slot; the other rows idle, seq_lens 0):
           dsa_kpool_update runs the request's tokens in order (the pools they close, their tail entries). The checks as
           above on the N rows (the reference per token over the kernel's selection, the rows this step wrote included).

  plans    the round-robin plan of build_graph's grids (OUT/plan.json), or --plan FILE (one GPU's lists, repeated for the
           GPUs); --search COSTS: the plan search.search_plan finds from it with the prefetch cost model
  costs    --record FILE: the timing build, this run's cost records (cost_records.records_of_run, the launches after the
           first WARM) appended to FILE; --collect FILE: every plan of plan_gen.collection_plans and the round-robin plan
           run with --record FILE by this script (--jobs at a time: builds in parallel, GPU work one at a time); the
           records carry no context length, so each context set gets its own file. On one GPU (--gpus 1 --tp 4: shard 0,
           the same tasks per GPU as TP4) the all-reduce's wait for the other GPUs is not in the records (a few us).

python demo/glm53_flash/dsa_fused_demo.py --model-dir /raid/catalyst/models/GLM-5.3-Flash [--tp 4] [--layer 3] [--seq-lens ...]
    [--page 1] [--maxctx 8192] [--out /tmp/flash_dsa_fused] [--reps 20] [--verify N] [--nvcc-flag=-DSTATIC_RESET_IN_KERNEL]
    [--plan FILE] [--search COSTS ...] [--record FILE] [--collect FILE [--jobs 4] [--rounds 3]]
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsa_layer as dd  # noqa: E402
import dsa_mla as mla  # noqa: E402
from dsa_mla import H, KPOOL, QL, T, bf, relrms  # noqa: E402

from mirage.mpk import compiler  # noqa: E402
from mirage.mpk.static_harness import Harness  # noqa: E402
from mirage.mpk.static_megakernel import StaticMegakernel  # noqa: E402

HC = 4   # mHC streams (hc_mult)
WARM = 2   # timed launches left out of the cost records


def run_args(args) -> list:
    """The arguments that decide the graph and the state, for this script in another process (--collect)."""
    return (["--model-dir", args.model_dir, "--layer", str(args.layer), "--tp", str(args.tp), "--gpus", str(args.gpus),
             "--page", str(args.page), "--maxctx", str(args.maxctx), "--seq-lens", *map(str, args.seq_lens), "--splits",
             str(args.splits), "--seed", str(args.seed), "--iters", str(args.iters), "--reps", str(args.reps), "--verify",
             str(args.verify)] + [f"--nvcc-flag={f}" for f in args.nvcc_flag])


def collect(args, mpk, current: dict) -> None:
    """--collect: every plan of plan_gen.collection_plans and the current plan ("cur") written to
    OUT/plans/<name>.json and run by this script with --record (output in OUT/<name>/), --jobs at a time: the builds in
    parallel, the GPU work one at a time (the lock OUT/gpu.lock). A plan whose run left a result.json is skipped."""
    from concurrent.futures import ThreadPoolExecutor
    from mirage.mpk import plan_gen
    plans = [("cur", current)] + plan_gen.collection_plans(mpk, rounds=args.rounds, seed=args.seed)
    plan_dir = os.path.join(args.out, "plans")
    os.makedirs(plan_dir, exist_ok=True)
    print(f"collect: {len(plans)} plans -> {plan_dir}, records -> {args.collect}, {args.jobs} at a time", flush=True)

    def one(item):
        name, plan = item
        path, run_out = os.path.join(plan_dir, name + ".json"), os.path.join(args.out, name)
        compiler.write_plan_file(path, dict(plan, lists=plan["lists"][:1]))   # one GPU's lists (main repeats them)
        if os.path.exists(os.path.join(run_out, "result.json")):
            return name, "done before"
        cmd = [sys.executable, os.path.abspath(__file__)] + run_args(args) + [
            "--plan", path, "--record", args.collect, "--out", run_out, "--gpu-lock", os.path.join(args.out, "gpu.lock")]
        with open(os.path.join(plan_dir, name + ".log"), "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        return name, f"FAILED (exit {code}), log {plan_dir}/{name}.log" if code else "ok"

    failed = []
    with ThreadPoolExecutor(args.jobs) as ex:
        for i, (name, status) in enumerate(ex.map(one, plans)):
            print(f"collect {i + 1}/{len(plans)} {name}: {status}", flush=True)
            failed += [name] if status.startswith("FAILED") else []
    print(f"collect: {len(plans) - len(failed)} of {len(plans)} plans measured" + (f", failed: {failed}" if failed else ""),
          flush=True)


def load_mhc(model_dir: str, layer: int) -> dict:
    """The layer's mHC parameters (hc_attn_* / hc_ffn_*: fn [24, 4 H], base [24], scale [3] fp32) and its two RMSNorm weights."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    p = f"model.language_model.layers.{layer}."

    def get(name):
        with safe_open(os.path.join(model_dir, where[p + name]), framework="pt") as f:
            return f.get_tensor(p + name)
    m = {k: get(k) for k in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
                             "input_layernorm.weight", "post_attention_layernorm.weight")}
    for k in m:
        m[k] = m[k].float() if k.startswith("hc_") else m[k].to(torch.bfloat16)
    assert m["hc_attn_fn"].shape == (HC * (HC + 2), HC * H)
    return m


def hc_pre_ref(R: torch.Tensor, fn, base, scale, gamma, rms_eps, hc_eps, norm_eps, iters, post_mult=2.0,
               sglang_norm: bool = True):
    """SGLang's _mhc_pre_torch on R [T, HC, H] bf16, then the norm: sglang_norm, as SGLang's fused mhc_pre
    (mhc_pre_big_fuse_with_norm: the sum of squares of the fp32 weighted sum, x = bf16(bf16(sum) rsqrt w)); else
    RMSNorm.forward_native (of the bf16 sum, rounded after rsqrt, then x w): (x bf16 [T, H], h_post [T, HC], h_res
    [T, HC HC] fp32)."""
    n = R.shape[0]   # rows (a verify step: fewer than T)
    x = R.reshape(n, HC * H).float()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + hc_eps
    post = post_mult * torch.sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = (mixes[:, 2 * HC:].view(n, HC, HC) * scale[2] + base[2 * HC:].view(HC, HC)).softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    s = (pre.unsqueeze(-1) * R.float()).sum(1)
    li = bf(s).float()
    if sglang_norm:
        return bf(li * torch.rsqrt(s.square().mean(-1, keepdim=True) + norm_eps) * gamma.float()), post, \
            comb.reshape(n, HC * HC)
    y = bf(li * torch.rsqrt(li.square().mean(-1, keepdim=True) + norm_eps))
    return bf(y.float() * gamma.float()), post, comb.reshape(n, HC * HC)


def hc_post_ref(out: torch.Tensor, R: torch.Tensor, h_post, h_res) -> torch.Tensor:
    """SGLang's _mhc_post_torch: R'_j = post_j out + sum_i comb[i][j] R_i -> bf16 [rows, HC, H] (rows = R's, T or fewer)."""
    n = R.shape[0]
    comb = h_res.view(n, HC, HC)
    mixed = (comb.unsqueeze(-1) * R.float().unsqueeze(2)).sum(1)
    return bf(h_post.view(n, HC, 1) * out.float().unsqueeze(1) + mixed)


def fx_float(q: torch.Tensor) -> torch.Tensor:
    """runtime.cuh fx_to_float: int64 fixed point (scale 2^40) -> fp32."""
    return q.to(torch.float32) * 2.0 ** -40


def tp_sum(parts) -> torch.Tensor:
    """The all-reduce of allreduce_send + hc_post: each GPU's fp32 partial sum rounded to bf16, added in fp32 in GPU order,
    rounded to bf16."""
    acc = torch.zeros_like(parts[0], dtype=torch.float32)
    for p in parts:
        acc = acc + bf(p).float()
    return bf(acc)


def layer_inputs(args, cfg: dict):
    """What every GPU shares: the mHC parameters, the streams R, the attention's mHC pre (x_attn, h_post, h_res; reference,
    CPU), hc_small of the FFN's pre, the boundary's options for build_graph (fused)."""
    eps, hc_eps, iters = cfg["rms_norm_eps"], cfg["hc_eps"], args.iters or cfg["hc_sinkhorn_iters"]
    m = load_mhc(args.model_dir, args.layer)
    g = torch.Generator().manual_seed(args.seed + 7)
    R = bf(torch.randn(T, HC, H, generator=g))
    x_attn, post_a, comb_a = hc_pre_ref(R, m["hc_attn_fn"], m["hc_attn_base"], m["hc_attn_scale"], m["input_layernorm.weight"],
                                        eps, hc_eps, eps, iters)   # as SGLang's hc_attn_pre (its fused norm)
    hc_small = torch.zeros(32, dtype=torch.float32)
    hc_small[:HC * (HC + 2)], hc_small[HC * (HC + 2):HC * (HC + 2) + 3] = m["hc_ffn_base"], m["hc_ffn_scale"]
    fused = {"rms_eps": eps, "hc_eps": hc_eps, "norm_eps": eps, "iters": iters, "verify": bool(getattr(args, "verify", 0)),
             "requests": T // 8}   # --verify: T / 8 requests of 8 rows (the demo's request in rows 0 .. N - 1, the rest idle)
    return {"m": m, "R": R, "x_attn": x_attn, "post_a": post_a, "comb_a": comb_a, "hc_small": hc_small, "fused": fused}


def rank_tensors(args, inp: dict, r: int, tp: int, dev):
    """GPU r's layer: its TP shard's weights, the replicated indexer and history (the same seeds on every GPU), the graph
    tensors (the DSA layer's, then the boundary's). Returns (w, st, the history to restore before each launch, the tensors)."""
    m, L = inp["m"], args.seq_lens
    NV = getattr(args, "verify", 0)
    if NV:   # one request: its N draft tokens at positions seq_lens[0] - 1 + t, the other rows idle
        L = [args.seq_lens[0] + t for t in range(NV)] + [0] * (T - NV)
    w = mla.load_layer(args.model_dir, args.layer, tp, r, dev)
    wi = dd.load_indexer(args.model_dir, args.layer, dev)
    st = mla.make_state(w, L, args.maxctx, args.page, dd.SLOTS, args.seed, dev)
    st.update(dd.make_indexer_state(L, args.maxctx, max(16, len(L)), 4, args.seed + 1, dev))   # request slots >= T
    if NV:   # every row: the request's page tables and slot
        for k in ("block_table", "pooled_bt"):
            st[k][1:] = st[k][0]
        st["req_slots"][1:] = st["req_slots"][0]
    st["x"] = inp["x_attn"].to(dev)
    st["indices"] = st.pop("pool_idx")
    hist = {k: st[k].clone() for k in ("cache", "pcache", "tail_k", "tail_score")}
    t = mla.gpu_tensors(w, st)
    t["pool_idx"] = t.pop("indices")
    t.update({k: wi[k] for k in ("w_kgw", "w_iqb", "k_norm_w", "k_norm_b", "ape")})
    t.update({k: st[k] for k in ("req_slots", "pooled_bt", "pcache", "tail_k", "tail_score", "idx_scores")})
    t["q_c"] = torch.zeros(T, mla.QL, dtype=torch.bfloat16, device=dev)
    t["o"] = torch.zeros(T, H, dtype=torch.int64, device=dev)   # o_proj's K parts added: this GPU's partial sum
    t.update({"R_in": inp["R"].reshape(T, HC * H).to(dev), "h_post_in": inp["post_a"].contiguous().to(dev),
              "h_res_in": inp["comb_a"].contiguous().to(dev), "R_out": torch.zeros(T, HC * H, dtype=torch.bfloat16, device=dev),
              "hc_ffn_fn": m["hc_ffn_fn"].contiguous().to(dev), "hc_ffn_small": inp["hc_small"].to(dev),
              "ffn_norm": m["post_attention_layernorm.weight"].to(dev),
              "x_ffn": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
              "h_post_out": torch.zeros(T, HC, dtype=torch.float32, device=dev),
              "h_res_out": torch.zeros(T, HC * HC, dtype=torch.float32, device=dev)})
    return w, st, hist, t


def rank_refs(args, cfg: dict, w: dict, st: dict, hist: dict, pool_idx, o, v):
    """GPU r's part of the checks, on its device: o_proj from the kernel's v (relRMS against the kernel's o) and its DSA
    reference on the kernel's selection (the o_proj partial sum, fp32, CPU)."""
    dev = st["x"].device
    L = st["seq_lens"].tolist()   # (--verify: the request's positions, the idle rows' 0)
    NV = getattr(args, "verify", 0) or T
    o_node = relrms(o[:NV], mla.o_proj(v.to(dev), w).cpu()[:NV])
    sel = dict(st, indices=dd.expand(pool_idx.to(dev), L))
    o_e2e = mla.reference(w, sel, hist["cache"].clone(), L, args.page, cfg["rms_norm_eps"])["o"].cpu()
    return o_node, o_e2e


def node_checks(args, cfg: dict, w: dict, st: dict, hist: dict, t: dict, r: int) -> bool:
    """GPU r's nodes before o_proj in a decode step, on its device (the module docstring): the indexer node by node (the
    pooled keys of the pools this step closed, dequantized, relRMS: a last-bit difference of the k norm flips an e4m3 key
    now and then; the tail rows equal; the scores from the kernel's q_c; the top-k =
    the top 512 of the kernel's own scores, its overlap with the reference's selection) and MLA on the kernel's selection
    (the latent rows written; q, o_lat, v end to end and per node). Prints them; returns whether they pass."""
    L, eps, page = st["seq_lens"].tolist(), cfg["rms_norm_eps"], args.page
    live = [i for i, n in enumerate(L) if n > 0]
    pc, tk, ts_ = hist["pcache"].clone(), hist["tail_k"].clone(), hist["tail_score"].clone()
    kw = dd.kpool_reference(t, st["x"], st, L, pc, tk, ts_, 1e-6)
    pool_rows = [(int(st["pooled_bt"][i, (n - 1) // KPOOL // dd.PPAGE]), (n - 1) // KPOOL % dd.PPAGE) for i, n in enumerate(L)
                 if n > 0 and (n - 1) % KPOOL == KPOOL - 1]
    pool_keys = sum(int((t["pcache"][p, o * 128:(o + 1) * 128] == pc[p, o * 128:(o + 1) * 128]).all()) for p, o in pool_rows)
    sc = dd.PPAGE * 128   # a page's scales after its keys
    deq = lambda c: torch.cat([c[p, o * 128:(o + 1) * 128].view(torch.float8_e4m3fn).float() *
                               c[p, sc + 4 * o:sc + 4 * o + 4].view(torch.float32) for p, o in pool_rows])
    pool_err = relrms(deq(t["pcache"]), deq(pc)) if pool_rows else 0.0
    tail_eq = ((t["tail_k"].view(torch.int16) == tk.view(torch.int16)).float().mean().item(),
               (t["tail_score"].view(torch.int16) == ts_.view(torch.int16)).float().mean().item())
    scores, picks = dd.score_reference(t, t["q_c"], kw, st, L, t["pcache"])
    valid = torch.zeros_like(scores, dtype=torch.bool)
    for i, n in enumerate(L):
        valid[i, :max(n, 0) // KPOOL] = True
    s_err = relrms(t["idx_scores"][valid], scores[valid])
    topk_ok = bool((dd.topk_of(t["idx_scores"], L) == t["pool_idx"]).all())
    overlap = []
    for i in live:   # the share of the reference's pools the kernel selected (1 with no pools)
        want = set(picks[i].tolist()) - {-1}
        overlap.append(len(set(t["pool_idx"][i].tolist()) & want) / len(want) if want else 1.0)
    qc_err = relrms(t["q_c"], mla.rmsnorm(bf(mla.gemm(st["x"], w["w_qkva"], w["w_qkva_s"]))[:, :QL], w["q_a_norm"], eps))
    print(f"  GPU {r} indexer: pools closed bit-equal {pool_keys} / {len(pool_rows)} (relRMS {pool_err:.2e}), tail k / "
          f"gate equal {100 * tail_eq[0]:.3f}% / {100 * tail_eq[1]:.3f}% | scores relRMS {s_err:.2e} | top-k of its own scores "
          f"{topk_ok} | "
          f"overlap with the reference's selection " + " ".join(f"{x:.3f}" for x in overlap) + f" | q_c relRMS {qc_err:.2e}")
    sel = dict(st, indices=dd.expand(t["pool_idx"], L))
    ref_cache = hist["cache"].clone()
    ref = mla.reference(w, sel, ref_cache, L, page, eps)
    rows = [mla.cache_row(st, i, L[i] - 1, page) for i in live]
    same = (t["cache"][rows].view(torch.int16) == ref_cache[rows].view(torch.int16)).float().mean().item()
    node = {"q": ref["q"], "o_lat": mla.attention(t["q_mla"], t["cache"], sel, L, page), "v": mla.o_absorb(t["o_lat"], w)}
    errs = {}
    for k, n in (("q", "q_mla"), ("o_lat", "o_lat"), ("v", "v")):
        errs[k], errs[k + "_node"] = relrms(t[n][live], ref[k][live]), relrms(t[n][live], node[k][live])
    print(f"  GPU {r} MLA: latent rows written bit-equal {100 * same:.2f}% | " +
          " | ".join(f"{k} end to end {errs[k]:.2e}, alone {errs[k + '_node']:.2e}" for k in ("q", "o_lat", "v")))
    return pool_err < 2e-2 and min(tail_eq) > 0.999 and s_err < 1e-2 and topk_ok and min(overlap) > 0.95 and \
        same > 0.99 and errs["q"] < 1e-3 and errs["o_lat_node"] < 3e-3 and errs["v_node"] < 1e-3


def check_boundary(args, cfg: dict, inp: dict, outs, o, o_node, o_e2e) -> bool:
    """The checks of the module docstring. outs[r] = GPU r's {R_out, x_ffn, h_post_out, h_res_out, pool_idx} (CPU); o[r]: its
    o_proj partial sum (fp32, CPU); o_node[r], o_e2e[r]: rank_refs. Prints them; returns whether they pass. --verify: the
    end-to-end comparison on the request's rows (the idle rows' o is not defined)."""
    eps, hc_eps, iters, tp = cfg["rms_norm_eps"], cfg["hc_eps"], args.iters or cfg["hc_sinkhorn_iters"], len(o)
    NV = getattr(args, "verify", 0) or T
    if NV < T:   # the idle rows: the kernel's own o (the boundary check), 0 for the reference's end to end
        o_e2e = [torch.cat([v[:NV], torch.zeros(T - NV, H, dtype=v.dtype)]) for v in o_e2e]
        o_cmp = [torch.cat([v[:NV], torch.zeros(T - NV, H, dtype=v.dtype)]) for v in o]
    else:
        o_cmp = o
    m, R, post_a, comb_a = inp["m"], inp["R"], inp["post_a"], inp["comb_a"]
    ffn = lambda Rx: hc_pre_ref(Rx, m["hc_ffn_fn"], m["hc_ffn_base"], m["hc_ffn_scale"], m["post_attention_layernorm.weight"],
                                eps, hc_eps, eps, iters)
    R_ref = hc_post_ref(tp_sum(o), R, post_a, comb_a)
    R_k = outs[0]["R_out"].view(T, HC, H)
    x_ref, post_f, comb_f = ffn(R_k)
    same_gpus = all(torch.equal(outs[r][k], outs[0][k]) for r in range(1, tp) for k in outs[0])
    eqR = (R_k.view(torch.int16) == R_ref.view(torch.int16)).float().mean().item()
    eqx = (outs[0]["x_ffn"].view(torch.int16) == x_ref.view(torch.int16)).float().mean().item()
    e = {"R": relrms(R_k, R_ref), "x_ffn": relrms(outs[0]["x_ffn"], x_ref),
         "h_post": relrms(outs[0]["h_post_out"], post_f), "h_res": relrms(outs[0]["h_res_out"], comb_f)}
    print(f"  boundary (from the kernel's partial sums): R' relRMS {e['R']:.2e} (bit-equal {100 * eqR:.3f}%) | "
          f"x_ffn relRMS {e['x_ffn']:.2e} (bit-equal {100 * eqx:.3f}%) | h_post {e['h_post']:.2e} | h_res {e['h_res']:.2e} | "
          f"every GPU's outputs equal: {same_gpus}")
    R_e2e = hc_post_ref(tp_sum(o_e2e), R, post_a, comb_a)
    x_e2e, post_e2e, _ = ffn(R_e2e)
    ee = {"o": max(relrms(o_cmp[r], o_e2e[r]) for r in range(tp)), "R'": relrms(R_k[:NV], R_e2e[:NV]),
          "R' - R": relrms(R_k[:NV].float() - R[:NV].float(), R_e2e[:NV].float() - R[:NV].float()),
          "x_ffn": relrms(outs[0]["x_ffn"][:NV], x_e2e[:NV]), "h_post": relrms(outs[0]["h_post_out"][:NV], post_e2e[:NV])}
    print("  o_proj per GPU (from the kernel's v): relRMS " + " ".join(f"{x:.2e}" for x in o_node))
    print("  end to end (each GPU's DSA reference on the kernel's selection, then the boundary): " +
          " | ".join(f"{k} {x:.2e}" for k, x in ee.items()))
    return same_gpus and e["R"] < 1e-3 and e["x_ffn"] < 1e-2 and e["h_post"] < 1e-4 and e["h_res"] < 1e-4 and \
        max(o_node) < 1e-3 and ee["x_ffn"] < 3e-2


def add_args(ap):
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=0, help="GPUs of the layer (default --tp): GPU r holds TP shard r of --tp, the "
                    "all-reduce adds these GPUs' partial sums (fewer than --tp: a test of the mechanism on fewer GPUs)")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--maxctx", type=int, default=8192)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[20, 60, 300, 2051, 2052, 4100, 6001, 8192],
                    help="the T requests' context lengths (fewer: repeated with + 1 up to T)")
    ap.add_argument("--splits", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=0, help="Sinkhorn iterations of the mHC pre steps (default: the config's; "
                    "a timing experiment)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--nvcc-flag", action="append", default=[])
    ap.add_argument("--verify", type=int, default=0, help="the verify layer: one request's N draft tokens (1 .. 8) at "
                    "positions --seq-lens[0] - 1 + t")
    ap.add_argument("--plan", help="a plan file (compiler.write_plan_file) instead of the round-robin plan")
    ap.add_argument("--search", nargs="+", metavar="COSTS", help="cost record files: the plan search.search_plan finds from "
                    "the round-robin plan (or --plan) with the prefetch cost model")
    ap.add_argument("--time", type=float, default=300, help="with --search: the solver's time limit (s)")
    ap.add_argument("--costs-code", default="current", help="with --search: the records of these kernel templates: current "
                    "(static_schedule.code_tag), any, or a tag")
    ap.add_argument("--record", metavar="FILE", help="the timing build; this run's cost records appended to FILE")
    ap.add_argument("--collect", metavar="FILE", help="a cost collection: each plan run with --record FILE")
    ap.add_argument("--jobs", type=int, default=4, help="with --collect: plans run at once")
    ap.add_argument("--rounds", type=int, default=3, help="with --collect: random plans per candidate grid of each node")
    ap.add_argument("--gpu-lock", help="a lock file held around this run's GPU work (--collect gives it to its runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.add_argument("--out", default="/tmp/flash_dsa_fused")
    args = ap.parse_args()
    while len(args.seq_lens) < T:   # T rows (STATIC_TOKENS): more requests, context lengths one shorter
        args.seq_lens = args.seq_lens + [max(1, v - 1) for v in args.seq_lens[:T - len(args.seq_lens)]]
    args.seq_lens = args.seq_lens[:T]
    tp, n, L = args.tp, args.gpus or args.tp, args.seq_lens
    assert n <= tp and torch.cuda.device_count() >= n, f"{torch.cuda.device_count()} GPUs visible, --gpus {n} of --tp {tp}"
    torch.backends.cuda.matmul.allow_tf32 = False
    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)["text_config"]
    assert cfg["layer_types"][args.layer] == "deepseek_sparse_attention" and cfg["hc_mult"] == HC
    eps, ln_eps = cfg["rms_norm_eps"], 1e-6
    inp = layer_inputs(args, cfg)
    ws, sts, hists, ts = [], [], [], []
    for r in range(n):
        w, st, hist, t = rank_tensors(args, inp, r, tp, torch.device("cuda", r))
        ws.append(w)
        sts.append(st)
        hists.append(hist)
        ts.append(t)
    mpk = StaticMegakernel(num_gpus=n)
    grids = dd.build_graph(mpk, ts[0], args.page, args.splits, eps, ln_eps, fused=inp["fused"])
    if os.environ.get("STATIC_L2_PREFETCH_MB"):   # the layer's GEMM weights into L2 at its start (core.cuh kernel_begin)
        mpk.l2_prefetch_weights(float(os.environ["STATIC_L2_PREFETCH_MB"]))
    os.makedirs(args.out, exist_ok=True)
    plan = mla.round_robin_plan(mpk, grids) if not args.plan else compiler.plan_from_file(args.plan)
    if args.collect:
        del ws, sts, hists, ts
        torch.cuda.empty_cache()
        collect(args, mpk, plan)
        return
    if args.search:
        from mirage.mpk import plan_viewer, search, static_schedule
        code = {"current": static_schedule.code_tag(), "any": None}.get(args.costs_code, args.costs_code)
        costs = search.Costs(args.search, code=code, gpu=torch.cuda.get_device_name(0), model="prefetch")
        plan, summary = search.search_plan(mpk, costs, time_limit_s=args.time, start_from=dict(plan, lists=plan["lists"][:1]))
        plan_viewer.save(os.path.join(args.out, "plan_view.json"), plan, summary, f"searched from {args.plan or 'round robin'}")
    if len(plan["lists"]) == 1 and n > 1:
        plan["lists"] = plan["lists"] * n   # every GPU: the same lists (no node splits its rows over the GPUs)
    compiler.write_plan_file(os.path.join(args.out, "plan.json"), plan)
    paths, info = compiler.compile_plan(mpk, compiler.plan_from_file(os.path.join(args.out, "plan.json")),
                                        os.path.join(args.out, "schedules"))
    timing = bool(args.record)
    sk = mpk.compile_static(paths, gpu_tensors=ts, out_dir=os.path.join(args.out, "build"), extra_flags=args.nvcc_flag,
                            profile=timing)
    if args.gpu_lock:   # held to the end of the process (the OS lets go of it at exit)
        import fcntl
        lock = open(args.gpu_lock, "a")
        fcntl.flock(lock, fcntl.LOCK_EX)
    h = Harness(sk)
    try:
        h.launch(dd.FLUSH_BYTES, timeout_s=60)
    except (TimeoutError, RuntimeError) as e:
        print("KERNEL DID NOT FINISH:", e, flush=True)
        os._exit(2)
    for r in range(n):
        torch.cuda.synchronize(r)
    print(f"layer {args.layer}, {n} GPUs with TP{tp} shards 0..{n - 1} ({ws[0]['w_uk'].shape[0]} heads per GPU), page "
          f"{args.page}, seq_lens {sts[0]['seq_lens'].tolist()}" + (f" (verify: {args.verify} draft tokens of one request)"
                                                                      if args.verify else ""))
    keys = ("R_out", "x_ffn", "h_post_out", "h_res_out", "pool_idx")
    outs = [{k: ts[r][k].cpu() for k in keys} for r in range(n)]
    torch.save({"outs": outs[0]}, os.path.join(args.out, "outputs.pt"))
    o = [fx_float(ts[r]["o"].cpu()) for r in range(n)]
    nodes_ok = args.verify or all([node_checks(args, cfg, ws[r], sts[r], hists[r], ts[r], r) for r in range(n)])
    refs = [rank_refs(args, cfg, ws[r], sts[r], hists[r], outs[r]["pool_idx"], o[r], ts[r]["v"].cpu()) for r in range(n)]
    ok = check_boundary(args, cfg, inp, outs, o, [x[0] for x in refs], [x[1] for x in refs]) and nodes_ok
    if not ok:
        print("CHECK FAILED: no timing")
        sk.finalize()
        sys.exit(1)
    print("CHECK PASSED")
    spans, launches = [], []
    for _ in range(args.reps):
        for r in range(n):
            for k in ("cache", "pcache", "tail_k", "tail_score"):
                ts[r][k].copy_(hists[r][k])
        h.launch(dd.FLUSH_BYTES, timeout_s=60)
        spans.append(h.span_us())
        if timing:
            launches.append((h.task_times(), h.timing()))
    if args.record:
        from mirage.mpk import cost_records
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        machine = {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc}
        records = cost_records.records_of_run(mpk, paths, launches[WARM:], os.path.basename(os.path.normpath(args.out)), machine)
        cost_records.append_records(args.record, records)
        print(f"cost records: {len(records)} added to {args.record} ({machine})")
    print(f"machine {socket.gethostname()} {torch.cuda.get_device_name(0)} x {n}, grids {info['grids']}, {args.reps} launches: "
          f"span min {min(spans):.1f} median {statistics.median(spans):.1f} max {max(spans):.1f} us")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"plan": args.plan or "round robin", "grids": {str(k): v for k, v in info["grids"].items()}, "seq_lens": L,
                   "gpus": n, "timing_build": timing, "spans_us": spans}, f, indent=1)
    sk.finalize()


if __name__ == "__main__":
    main()
