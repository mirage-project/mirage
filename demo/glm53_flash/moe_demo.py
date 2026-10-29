"""GLM-5.3-Flash MoE (a sparse layer: 288 routed experts, top 8, one shared expert) with the static-schedule compiler: one
decode step of T tokens on TP shard --rank of --tp (--gpus N: ranks 0 .. N - 1 on N GPUs), the sublayer's end in the kernel.
The experts are #799's GLM-5.3 MoE tasks (demo/glm53_moe) with Flash's shapes:

  shapes      hidden 4096, 288 experts: the router padded to 384 rows (whole 128-row tiles; topk_route reads the first 288);
              the expert intermediate 2048, 512 per GPU at TP4, as 2 sub-experts of 256 each (scaled_moe_experts takes at most
              256): the bank has 576 experts, and topk_route gives both sub-experts of a chosen expert its weight (16 pairs
              per token); the shared expert is the bank's last 2 sub-experts (topk_route gives every token them with weight
              1), run in the experts node's queue
  activation  swiglu_limit 10: SiLU(min(gate, 10)) x clamp(up, -10, 10), routed and shared
  graph       router (gemm_tile, K parts in slots) -> topk_route (sigmoid + bias, top 8, renorm x 2.5, + the shared expert);
              group_quant(x); scaled_moe_experts -> allreduce_send (a token's 18 rows added, bf16 to every GPU) -> the end
              (--end): mhc: hc_post with the finished mHC coefficients (SGLang's MHCState): R'_j = sum_i comb[i][j] R_i +
              post_j (sum over the GPUs); mhc_pre: hc_post_mix + hc_pre: that and the next layer's attention mHC pre (its
              x_attn, h_post, h_res), as SGLang runs it (MOE_STATIC_END)
  check       a PyTorch reference with the kernel's rounding points: fp64 routing; group_quant e4m3 (s = amax / 448); the
              checkpoint's FP8 blocks; h in fp32 -> e4m3 per 128 features; R in fp32, y = bf16(sum R). Checked: the experts
              chosen and their weights, the routed rows of each chosen expert and the shared expert's (this node alone: from
              the kernel's own pairs), the new streams against SGLang's mHC formulas on the ranks' y added in rank order
              (with mhc_pre: the next attention's x_attn, h_post, h_res too), every rank's new streams equal; random streams
              and coefficients
  time        --reps launches, L2 emptied before each (--save-y / --same-y: the new streams saved, or required bit-equal to
              saved ones)
              --prof-experts: the experts node with its per-SM stamps (scaled_moe_experts.cuh moe_prof_event): per phase,
              the medians over the SMs (W13 items, the ring switch, W2 tiles and what each waits for)
              --timeline: the timing build; per node of the last launch, its first task start and last task end (from the
              start barrier) and its median task time
              --record FILE: the timing build; the per-task times become cost records (cost_records.py) appended to FILE
  plan        the round-robin plan (round_robin_plan), or --plan FILE; --search COSTS: the solver's plan from the cost record
              files COSTS, starting from that plan (search.search_plan, as demo/glm53_moe/demo.py) -> <out>/plan.json and
              <out>/plan_view.json (its predicted timeline)

python demo/glm53_flash/moe_demo.py --model-dir /raid/catalyst/models/GLM-5.3-Flash [--layer 3] [--tp 4 --rank 0] [--gpus N]
    [--end mhc | mhc_pre] [--out /tmp/flash_moe] [--reps 20] [--plan PLAN] [--timeline] [--record FILE] [--search COSTS ...]
"""
import argparse
import contextlib
import itertools
import json
import os
import socket
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

from mirage.core import bfloat16, float32, uint8
from mirage.mpk import compiler
from mirage.mpk.fp8_fp32scale_weights import pack_w13, pack_w2
from mirage.mpk.static_harness import Harness
from mirage.mpk.static_megakernel import HC_MIX_STRIDE, StaticMegakernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsa_mla import bf, fp8_gemm, gemm, group_quant, relrms  # noqa: E402

T = int(os.environ.get("STATIC_TOKENS", "8"))   # tokens (rows) per step: the kernel's T (8 or 16)
H, I, E, TOPK, SCALE, LIMIT = 4096, 2048, 288, 8, 2.5, 10.0
HC = 4               # mHC: the residual is 4 streams [T, HC H]
SUB = 2              # sub-experts per expert shard (TP4: 512 features -> 2 x 256)
ROUTER_ROWS = 384    # the router padded to whole 128-row tiles
FLUSH_BYTES = 512 << 20


def load_layer(model_dir: str, layer: int, tp: int, rank: int, dev) -> dict:
    """The MoE tensors of `layer` for shard `rank` of `tp` on dev, as the checkpoint has them (e4m3, fp32 128 x 128 block
    scales): router [E, H] bf16, bias [E] fp32; the routed experts' shards gate_up [E, 2 IR, H] (the gate rows of features
    [rank IR, rank IR + IR), then their up rows), gate_up_s [E, 2 IR / 128, H / 128], down [E, H, IR] (the same features'
    columns), down_s [E, H / 128, IR / 128]; the shared expert's shard alike (sgu, sgu_s [2 IR, H], sd, sd_s [H, IR])."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    p = f"model.language_model.layers.{layer}.mlp."
    IR = I // tp
    rows, blk = slice(rank * IR, rank * IR + IR), slice(rank * IR // 128, (rank * IR + IR) // 128)
    with contextlib.ExitStack() as stack:
        files = {}

        def get(name, r=None, c=None):
            fn = where[p + name]
            if fn not in files:
                files[fn] = stack.enter_context(safe_open(os.path.join(model_dir, fn), framework="pt"))
            sl = files[fn].get_slice(p + name)
            v = sl[r] if c is None else sl[:, c]
            return v.to(dev)

        def shard(pre):
            u8 = lambda v: v.view(torch.uint8)
            gu = torch.cat([u8(get(pre + "gate_proj.weight", rows)), u8(get(pre + "up_proj.weight", rows))]).view(
                torch.float8_e4m3fn)
            gs = torch.cat([get(pre + "gate_proj.weight_scale_inv", blk), get(pre + "up_proj.weight_scale_inv", blk)])
            return gu, gs.float(), get(pre + "down_proj.weight", c=rows).contiguous(), \
                get(pre + "down_proj.weight_scale_inv", c=blk).float().contiguous()

        w = {"router": get("gate.weight", slice(None)), "bias": get("gate.e_score_correction_bias", slice(None)).float()}
        assert w["router"].shape == (E, H) and w["router"].dtype == torch.bfloat16
        w["sgu"], w["sgu_s"], w["sd"], w["sd_s"] = shard("shared_experts.")
        gu, gs, d, ds = shard("experts.0.")
        w["gate_up"] = torch.empty((E,) + gu.shape, dtype=gu.dtype, device=dev)
        w["gate_up_s"] = torch.empty((E,) + gs.shape, dtype=torch.float32, device=dev)
        w["down"] = torch.empty((E,) + d.shape, dtype=d.dtype, device=dev)
        w["down_s"] = torch.empty((E,) + ds.shape, dtype=torch.float32, device=dev)
        for e in range(E):
            w["gate_up"][e], w["gate_up_s"][e], w["down"][e], w["down_s"][e] = shard(f"experts.{e}.") if e else (gu, gs, d, ds)
    assert w["gate_up"].dtype == torch.float8_e4m3fn and w["gate_up"].shape == (E, 2 * IR, H)
    return w


def bank(w: dict, sub: int = SUB) -> dict:
    """The kernel's expert bank from load_layer's tensors: each expert's IR features as `sub` sub-experts of SR = IR / sub
    features (expert e -> bank experts e sub + j), the shared expert's as the last `sub` (E sub .. E sub + sub - 1: the
    experts node runs them, topk_route_layer(shared_experts=sub)): w13 [E sub + sub, 2 SR, H] uint8 pieces, w13_sf
    [E sub + sub, SR / 128, H / 128, 2], w2 [E sub + sub, H, SR], w2_sf [E sub + sub, H / 128, SR / 128]
    (mirage.mpk.fp8_fp32scale_weights); the router padded to ROUTER_ROWS rows and its bias."""
    NE, IR2, HH = w["gate_up"].shape
    IR = IR2 // 2
    SR, dev = IR // sub, w["gate_up"].device
    assert SR % 128 == 0
    NB = NE * sub + sub
    b = {"w13": torch.empty(NB, 2 * SR, HH, dtype=torch.uint8, device=dev),
         "w13_sf": torch.empty(NB, SR // 128, HH // 128, 2, dtype=torch.float32, device=dev),
         "w2": torch.empty(NB, HH, SR, dtype=torch.uint8, device=dev),
         "w2_sf": torch.empty(NB, HH // 128, SR // 128, dtype=torch.float32, device=dev)}
    u8 = lambda v: v.view(torch.uint8)
    experts = [(u8(w["gate_up"][e]), w["gate_up_s"][e], u8(w["down"][e]), w["down_s"][e]) for e in range(NE)]
    experts.append((u8(w["sgu"]), w["sgu_s"], u8(w["sd"]), w["sd_s"]))
    for e, (gu, gs, d, ds) in enumerate(experts):
        for j in range(sub):
            r, s = slice(j * SR, (j + 1) * SR), slice(j * SR // 128, (j + 1) * SR // 128)
            g, gsc = gu[:IR][r], gs[:IR // 128][s]
            u, usc = gu[IR:][r], gs[IR // 128:][s]
            q, sf = pack_w13(g, gsc, u, usc)
            b["w13"][e * sub + j], b["w13_sf"][e * sub + j] = q.view(2 * SR, HH), sf
            q, sf = pack_w2(d[:, r].contiguous(), ds[:, s].contiguous())
            b["w2"][e * sub + j], b["w2_sf"][e * sub + j] = q.view(HH, SR), sf
    router = torch.zeros(ROUTER_ROWS, HH, dtype=torch.bfloat16, device=dev)
    router[:NE] = w["router"]
    b["router"], b["bias"] = router, w["bias"]
    return b


def gpu_tensors(b: dict, x: torch.Tensor, attn_next: dict = None) -> dict:
    """The graph tensors by name: the bank, x, and the outputs the check reads (nodes write them: these graph tensors bind
    the node buffers): the pairs [T, (K + 1) SUB] (the shared expert's last), the routed rows R [T (K + 1) SUB, H] fp32;
    the residual streams [T, HC H] bf16, their finished mHC coefficients h_post [T, HC] and h_res [T, HC HC] fp32
    (SGLang's MHCState), the new streams [T, HC H] bf16; moe_out and residual [T, H] bf16 are not read (they keep the
    graph's numbering, which the plan files use). attn_next (build_graph): the next layer's attention mHC parameters and
    its outputs x_attn [T, H] bf16, h_post_next, h_res_next."""
    dev, KS, T = x.device, (TOPK + 1) * SUB, x.shape[0]
    t = dict(b)
    t.update(x=x, routing_pairs=torch.zeros(T, KS, dtype=torch.int64, device=dev),
             routed_rows=torch.zeros(T * KS, H, dtype=torch.float32, device=dev),
             moe_out=torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
             residual=torch.zeros(T, H, dtype=torch.bfloat16, device=dev))
    t.update(streams=torch.zeros(T, HC * H, dtype=torch.bfloat16, device=dev),
             h_post=torch.zeros(T, HC, dtype=torch.float32, device=dev),
             h_res=torch.zeros(T, HC * HC, dtype=torch.float32, device=dev),
             streams_out=torch.zeros(T, HC * H, dtype=torch.bfloat16, device=dev))
    if attn_next is not None:   # the next layer's attention mHC pre at the end (build_graph): its parameters, the outputs
        hc_small = torch.zeros(32, dtype=torch.float32)
        hc_small[:HC * (HC + 2)], hc_small[HC * (HC + 2):HC * (HC + 2) + 3] = \
            attn_next["hc_base"].float().cpu(), attn_next["hc_scale"].float().cpu()
        t.update(hc_attn_fn=attn_next["hc_fn"].float().contiguous().to(dev), hc_attn_small=hc_small.to(dev),
                 attn_norm=attn_next["norm"].to(torch.bfloat16).contiguous().to(dev),
                 x_attn=torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
                 h_post_next=torch.zeros(T, HC, dtype=torch.float32, device=dev),
                 h_res_next=torch.zeros(T, HC * HC, dtype=torch.float32, device=dev))
    return t


def build_graph(mpk, t: dict, attn_next: dict = None, valid_rows: int = None) -> dict:
    """The layer's nodes (mpk.num_gpus GPUs, one TP shard each); returns the grids of the plan {graph_idx: grid} for the
    GEMMs whose default grid is not used. The end: allreduce_send (R: a token's rows added, bf16 to every GPU) ->
      t without "x_attn"  hc_post with the finished coefficients (MODE 1): R'_j = comb[i][j] R_i over i + post_j sum over
                          the GPUs -> streams_out
      attn_next (t with "x_attn", gpu_tensors attn_next; {"rms_eps", "hc_eps", "norm_eps", "iters"}): the next layer's
                          attention mHC pre too (beyond #811: no framework kernel between the two layers): hc_post_mix (the
                          sum over the GPUs, the mHC post, the next attention's mixes with hc_attn_fn) -> hc_pre (the next
                          layer's x_attn = RMSNorm(sum_i pre_i R'_i) input_layernorm, its h_post / h_res exported) -> x_attn,
                          h_post_next, h_res_next
    valid_rows: the live rows of a speculative verify step (the rows past them are not routed)."""
    a = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in t.items()}
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    T = t["x"].shape[0]
    logits, z_q = nt((T, ROUTER_ROWS), float32, "router_logits"), nt((T, H), uint8, "x_fp8")
    mpk.gemm_tile_layer(input=a["x"], weight=a["router"], output=logits)                             # logits = x . Wg^T
    mpk.topk_route_layer(input=logits, bias=a["bias"], output=a["routing_pairs"],                    # sigmoid + bias, top 8,
                         routed_scaling_factor=SCALE, sub_experts=SUB,                               # renorm x 2.5
                         shared_experts=SUB,                                                         # + the shared expert
                         valid_rows=valid_rows)                                                      # idle rows: no experts
    mpk.group_quant_layer(input=a["x"], output=z_q)                                                  # x in FP8, fp32 scales
    mpk.scaled_moe_experts_layer(z_q=z_q, pairs=a["routing_pairs"], w13=a["w13"], w13_scales=a["w13_sf"],
                                 w2=a["w2"], w2_scales=a["w2_sf"], output=a["routed_rows"], swiglu_limit=LIMIT,
                                 prof=a.get("moe_prof"))                                             # --prof-experts
    r_sent = nt((T, H), bfloat16, "r_sent")
    mpk.allreduce_send_layer(input=a["routed_rows"], output=r_sent)                                  # a token's 18 rows added
    if "x_attn" in t:   # the mHC post and the next layer's attention mHC pre
        assert attn_next is not None, "the next layer's attention pre: its eps and iterations"
        mix = nt((T, HC_MIX_STRIDE), float32, "attn_hc_mix")
        mpk.hc_post_mix_layer(sent=r_sent, residual=a["streams"], coef=(a["h_post"], a["h_res"]), fn=a["hc_attn_fn"],
                              output=a["streams_out"], mix=mix)
        mpk.hc_pre_layer(mix=mix, residual=a["streams_out"], hc_small=a["hc_attn_small"], gamma=a["attn_norm"],
                         output=a["x_attn"], rms_eps=attn_next["rms_eps"], hc_eps=attn_next["hc_eps"],
                         norm_eps=attn_next["norm_eps"], sinkhorn_iters=attn_next["iters"],
                         export=(a["h_post_next"], a["h_res_next"]))
    else:
        mpk.hc_post_layer(sent=r_sent, residual=a["streams"], output=a["streams_out"],
                          coef=(a["h_post"], a["h_res"]))                                            # sum over the GPUs, mHC post
    router = next(n for n in compiler.graph_nodes(mpk) if n.name == "gemm_tile")
    # the router: 64-row tiles x K in 8 parts (4 K tiles each)
    return {router.graph_idx: (ROUTER_ROWS // 64, 8, 1)}


def round_robin_plan(mpk, grids: dict) -> dict:
    """grids {graph_idx: grid} for the nodes they name (the others: their layer's grid); the tasks dealt to the SMs in turn, in
    graph order (x slowest); a one_task_per_sm node's task x on SM x. The same lists on every GPU (mpk.num_gpus)."""
    num_sms = mpk._num_sms()
    per_sm, cursor = [[] for _ in range(num_sms)], 0
    for n in compiler.graph_nodes(mpk):
        for pos in itertools.product(*(range(d) for d in grids.get(n.graph_idx, n.grid))):
            if n.info["one_task_per_sm"]:
                per_sm[pos[0]].append((n.graph_idx, pos))
                continue
            per_sm[cursor % num_sms].append((n.graph_idx, pos))
            cursor += 1
    # the in-memory plan (as compiler.plan_from_file returns it: grids by graph_idx; search.search_plan looks them up so)
    return {"grids": {k: tuple(v) for k, v in grids.items()}, "lists": [per_sm] * mpk.num_gpus}


# ---- the reference ----
def route(x: torch.Tensor, w: dict):
    """fp64 logits of bf16 x and the router, sigmoid, + bias, the top K (by score + bias), the K sigmoid scores renormalised
    x the routed scaling factor -> experts [T, K] int64 (in rank order), weights [T, K] fp64."""
    s = torch.sigmoid(x.double() @ w["router"].double().t())
    idx = torch.topk(s + w["bias"].double(), TOPK, dim=-1, sorted=True).indices
    wt = s.gather(1, idx)
    return idx, wt / wt.sum(-1, keepdim=True) * SCALE


def swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return F.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT)


def expert(x: torch.Tensor, gu, gu_s, d, d_s, act_fp8: bool = True) -> torch.Tensor:
    """x bf16 [n, H] through one expert shard (gate_up e4m3 [2 IR, H]: gate rows, then up rows; down [H, IR]): fp64 [n, H].
    act_fp8: the kernel's rounding (x and h in e4m3 per 128), else exact activations (fp64 h, the weights' own rounding
    only)."""
    g = gemm(x, gu, gu_s, act_fp8)
    IR = gu.shape[0] // 2
    if not act_fp8:
        return gemm(swiglu(g[:, :IR], g[:, IR:]), d, d_s, act_fp8=False)
    h = swiglu(g[:, :IR].float(), g[:, IR:].float())
    return fp8_gemm(*group_quant(h), d, d_s)


def routed(x: torch.Tensor, w: dict, experts: torch.Tensor, weights: torch.Tensor, act_fp8: bool = True) -> torch.Tensor:
    """fp64 [T, K, H]: weight x the expert's output, for each token's K chosen experts."""
    out = torch.zeros(x.shape[0], experts.shape[1], H, dtype=torch.float64, device=x.device)
    for e in experts.unique().tolist():
        tt, kk = (experts == e).nonzero(as_tuple=True)
        y = expert(x[tt], w["gate_up"][e], w["gate_up_s"][e], w["down"][e], w["down_s"][e], act_fp8)
        out[tt, kk] = weights[tt, kk].double().unsqueeze(1) * y
    return out


def shared(x: torch.Tensor, w: dict, act_fp8: bool = True) -> torch.Tensor:
    """The shared expert, in the experts node's queue as a routed expert."""
    return expert(x, w["sgu"], w["sgu_s"], w["sd"], w["sd_s"], act_fp8)


def reference(x: torch.Tensor, w: dict, act_fp8: bool = True) -> dict:
    """The layer from x: the routing, R (fp64 [T, K, H]), S (fp64 [T, H]), y = bf16(sum R + S) (the rows added in fp32)."""
    experts, weights = route(x, w)
    R, S = routed(x, w, experts, weights, act_fp8), shared(x, w, act_fp8)
    return {"experts": experts, "weights": weights, "R": R, "S": S, "y": bf(R.sum(1) + S)}


def kernel_pairs(pairs: torch.Tensor):
    """The kernel's pairs [T, (K + 1) SUB] (bank expert << 32 | fp32 weight bits) -> experts [T, K], weights [T, K] fp32,
    and whether each expert's SUB pairs are its sub-experts in order with one weight and the shared expert's the bank's
    last SUB experts with weight 1."""
    p32 = pairs.contiguous().view(torch.int32).view(pairs.shape[0], -1, SUB, 2)   # little-endian: the weight bits, then
                                                                                 # the expert
    e, wt = p32[..., 1].long(), p32[..., 0].contiguous().view(torch.float32)
    ok = bool((e == e[..., :1] // SUB * SUB + torch.arange(SUB, device=pairs.device)).all() and (e[..., 0] % SUB == 0).all()
              and (wt == wt[..., :1]).all())
    ok = ok and bool((e[:, TOPK, 0] == E * SUB).all() and (wt[:, TOPK] == 1.0).all())
    return e[:, :TOPK, 0] // SUB, wt[:, :TOPK, 0], ok


def check(t: dict, w: dict, x: torch.Tensor = None) -> dict:
    """The experts node's outputs in t against the reference (from x, default t["x"]; the rows of x): the pairs, the
    weights, the routed rows of each chosen expert and the shared expert's (S: its rows of R; this node alone, from the
    kernel's own pairs). Prints them; returns the errors and "ok"."""
    x = t["x"] if x is None else x
    T = x.shape[0]   # the rows checked
    r = reference(x, w)
    ke, kw, pairs_ok = kernel_pairs(t["routing_pairs"][:T])
    same_sets = [set(ke[i].tolist()) == set(r["experts"][i].tolist()) for i in range(T)]
    same_order = bool((ke == r["experts"]).all())
    by_expert = lambda e, v: v.gather(1, e.sort(1).indices)   # the weights in expert order (a tie may swap the ranks)
    err = {"weights": relrms(by_expert(ke, kw), by_expert(r["experts"], r["weights"])) if all(same_sets) else float("nan")}
    # a chosen expert's 2 sub-expert rows added
    rows = t["routed_rows"].view(-1, (TOPK + 1), SUB, H).double().sum(2)[:T]
    err["R_node"] = relrms(rows[:, :TOPK], routed(x, w, ke, kw))
    err["S"] = relrms(rows[:, TOPK], r["S"])
    ok = pairs_ok and all(same_sets) and err["weights"] < 1e-5 and err["R_node"] < 3e-3 and err["S"] < 3e-3
    print(f"  pairs: sub-experts in order with one weight: {pairs_ok}; experts as the reference: {sum(same_sets)}/{T} tokens "
          f"(same order: {same_order}); weights relRMS {err['weights']:.2e} | routed rows (this node alone) relRMS "
          f"{err['R_node']:.2e} | S {err['S']:.2e}")
    return dict(err, ok=ok)


def check_end(ts: list, ws: list, x: torch.Tensor, streams, h_post, h_res, attn_next: dict, n: int) -> bool:
    """Every GPU's new streams (and with attn_next the next attention's x_attn, h_post, h_res), rows 0 .. n - 1, against
    SGLang's mHC formulas (dsa_fused_demo.hc_post_ref / hc_pre_ref) on the ranks' y (each rank's rounding reference, bf16 as
    sent) added in rank order in fp32, and against the exact ones (unquantized activations); every GPU's equal. Prints
    them; returns whether they pass."""
    from dsa_fused_demo import hc_post_ref, hc_pre_ref
    xs = [x[:n].to(w["router"].device) for w in ws]
    part = lambda fp8: sum(reference(xr, w, act_fp8=fp8)["y"].float().cpu() for xr, w in zip(xs, ws))
    R = streams[:n].view(n, HC, H)
    want = hc_post_ref(part(True), R, h_post[:n], h_res[:n])
    want_exact = hc_post_ref(part(False), R, h_post[:n], h_res[:n])
    ok = True
    for r, t in enumerate(ts):
        so = t["streams_out"][:n].float().cpu().view(n, HC, H)
        eq = r == 0 or bool((t["streams_out"].cpu().view(torch.int16) == ts[0]["streams_out"].cpu().view(torch.int16)).all())
        e_f, e_x = relrms(so, want.float()), relrms(so, want_exact.float())
        line = f"  GPU {r}: new streams relRMS vs the formulas on the ranks' sums {e_f:.2e}, vs exact {e_x:.2e}, = GPU 0: {eq}"
        ok = ok and eq and e_f < 2e-2
        if attn_next is not None:   # the next layer's attention input and coefficients from the new streams
            x_ref, hp_ref, hr_ref = hc_pre_ref(want, attn_next["hc_fn"].float().cpu(), attn_next["hc_base"].float().cpu(),
                                               attn_next["hc_scale"].float().cpu(), attn_next["norm"].cpu(),
                                               attn_next["rms_eps"], attn_next["hc_eps"], attn_next["norm_eps"],
                                               attn_next["iters"])
            e_a = relrms(t["x_attn"][:n].float().cpu(), x_ref.float())
            line += (f" | next attention pre: x_attn {e_a:.2e}, h_post {relrms(t['h_post_next'][:n].cpu(), hp_ref):.2e}, h_res "
                     f"{relrms(t['h_res_next'][:n].cpu(), hr_ref):.2e}")
            ok = ok and e_a < 2e-2
        print(line, flush=True)
    return ok


def print_timeline(mpk, schedule_path: str, h: Harness):
    """Per node of the last launch (timing build): its tasks, first start and last end from the start barrier, the median of
    its task times (start to the last warp's end), us."""
    from mirage.mpk.cost_records import stamps_of_gpu
    with open(schedule_path) as f:
        sched = json.load(f)
    bar = h.timing()[0][0]
    per = {}
    for tid, st in stamps_of_gpu(sched, h.task_times()[0]).items():
        per.setdefault(sched["all_tasks"][tid]["node"], []).append(st)
    names = {n.graph_idx: n.name for n in compiler.graph_nodes(mpk)}
    print(f"  timeline (us from the start barrier; span {h.span_us():.1f}):")
    for k in sorted(per, key=lambda k: min(st["start"] for st in per[k])):
        v = per[k]
        print(f"    {names[k]:20s}#{k:<3d} {len(v):4d} tasks  {(min(st['start'] for st in v) - bar) / 1e3:7.1f} .. "
              f"{(max(st['end'] for st in v) - bar) / 1e3:7.1f}  median task {statistics.median((st['end'] - st['start']) / 1e3 for st in v):6.2f}")
    pool = next(k for k in per if names[k] == "scaled_moe_experts")
    ends = sorted((st["end"] - bar) / 1e3 for st in per[pool])
    q = lambda f: ends[min(len(ends) - 1, int(f * len(ends)))]
    print(f"  experts: per-SM end min {ends[0]:.1f}, 10% {q(0.1):.1f}, median {q(0.5):.1f}, 90% {q(0.9):.1f}, max {ends[-1]:.1f}")
    return min(st["start"] for st in per[pool]) - bar, ends[-1] * 1e3


def prof_report(prof: torch.Tensor, bar: int, t: dict):
    """The experts node's stamps of the last launch (scaled_moe::moe_prof_event; ns): per SM its W13 items (seen -> h_q
    written) and W2 tiles (seen -> CHQ and scale loads -> accumulator full -> rows stored), the ring switch, the W2 loaders'
    waits; printed as medians (p10 / p90) over the SMs, in us. Bytes: a W13 item 128 x H, a W2 tile 128 x its K (+ h_q)."""
    p = prof.numpy()
    w13_bytes, w2_bytes = 128 * H, 128 * t["w2"].shape[2] + T * t["w2"].shape[2]
    per = {k: [] for k in ("task started", "first barrier", "z_q + pairs landed", "switch done (t0)", "switch done (t64)",
                           "table: 1st barrier", "table: slots numbered", "start",
                           "end", "W13 items", "W13 item (seen-done)", "W13 done-done", "last W13 done",
                           "switch", "last W13 -> 1st W2 done", "W2 tiles", "W2: scales (+ CHQ, new slot)", "W2: wait acc full",
                           "W2: drain + store", "W2 done-done", "W2 loaders: wait stage (sum)", "W2 loaders: wait CHQ (sum)")}
    for sm in range(p.shape[0]):
        r = p[sm]
        if r[0] == 0:
            continue
        ev = [(int(v) >> 56, (int(v) & ((1 << 56) - 1)) / 1e3) for v in r[16:16 + int(r[8])]]
        for k, i in (("task started", 9), ("first barrier", 13), ("z_q + pairs landed", 10), ("switch done (t0)", 11),
                     ("switch done (t64)", 12), ("table: 1st barrier", 14), ("table: slots numbered", 15)):
            per[k].append((r[i] - bar) / 1e3)
        per["start"].append((r[0] - bar) / 1e3)
        per["end"].append((r[1] - bar) / 1e3)
        seen13 = [x for k, x in ev if k == 1]
        done13 = [x for k, x in ev if k == 2]
        per["W13 items"].append(len(done13))
        per["W13 item (seen-done)"] += [b - a for a, b in zip(seen13, done13)]
        per["W13 done-done"] += [b - a for a, b in zip(done13, done13[1:])]
        if done13:
            per["last W13 done"].append(done13[-1] + (r[0] - bar) / 1e3)
        if r[2]:
            per["switch"].append((r[3] - r[2]) / 1e3)
        w2 = {k: [x for kk, x in ev if kk == k] for k in (3, 4, 5, 6)}
        per["W2 tiles"].append(len(w2[6]))
        per["W2: scales (+ CHQ, new slot)"] += [b - a for a, b in zip(w2[3], w2[4])]
        per["W2: wait acc full"] += [b - a for a, b in zip(w2[4], w2[5])]
        per["W2: drain + store"] += [b - a for a, b in zip(w2[5], w2[6])]
        per["W2 done-done"] += [b - a for a, b in zip(w2[6], w2[6][1:])]
        if done13 and w2[6]:
            per["last W13 -> 1st W2 done"].append(w2[6][0] - done13[-1])
        per["W2 loaders: wait stage (sum)"].append((r[4] + r[6]) / 1e3)
        per["W2 loaders: wait CHQ (sum)"].append((r[5] + r[7]) / 1e3)
    q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
    names = {1: "13seen", 2: "13done", 3: "2seen", 4: "2scales", 5: "2full", 6: "2done"}
    by_start = sorted(range(p.shape[0]), key=lambda sm: p[sm][0])
    for sm in (by_start[0], by_start[len(by_start) // 2], by_start[-1]):   # the first, median and last SM to start
        r = p[sm]
        ev = [(int(v) >> 56, (int(v) & ((1 << 56) - 1)) / 1e3) for v in r[16:16 + int(r[8])]]
        print(f"  SM {sm}: starts {(r[0] - bar) / 1e3:.1f} us, switch {(r[2] - r[0]) / 1e3:.1f}..{(r[3] - r[0]) / 1e3:.1f}, "
              f"end {(r[1] - r[0]) / 1e3:.1f} (from its start): " +
              " ".join(f"{names[k]}@{x:.1f}" for k, x in ev if k in (1, 2, 6))[:900])
    for k, v in per.items():
        if v:
            print(f"    {k:30s} {statistics.median(v):8.2f}   ({q(v, 0.1):.2f} / {q(v, 0.9):.2f})")
    for k, nbytes in (("W13 done-done", w13_bytes), ("W2 done-done", w2_bytes)):
        if per[k]:
            print(f"    -> one SM in steady {k.split()[0]}: {nbytes / 1e3 / statistics.median(per[k]):.1f} GB/s "
                  f"({nbytes // 1024} KB per {k.split()[0]} unit), x {len(per['start'])} SMs = "
                  f"{nbytes / 1e3 / statistics.median(per[k]) * len(per['start']) / 1e3:.2f} TB/s")


def run_args(args) -> list:
    """The arguments that decide the graph and the state, for this script in another process (--collect)."""
    return (["--model-dir", args.model_dir, "--layer", str(args.layer), "--tp", str(args.tp), "--rank", str(args.rank), "--seed",
             str(args.seed), "--x-scale", str(args.x_scale), "--reps", str(args.reps), "--end", args.end]
            + [f"--nvcc-flag={f}" for f in args.nvcc_flag])


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
        compiler.write_plan_file(path, plan)
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


def run_tp(args, w0: dict, x: torch.Tensor, attn_next: dict) -> None:
    """--gpus N (2 .. --tp): ranks 0 .. N-1 of TP --tp on GPUs 0 .. N-1 of this process, one kernel each: the all-reduce over
    the N GPUs through the runtime's exchange region (multicast), as SGLang's TP layers (the sum over N ranks, not all --tp:
    the mechanism, not the model's numbers). x and the random streams and coefficients are the same on every rank. Checks
    every rank's new streams (check_end), then --reps timed launches."""
    n = args.gpus
    assert 1 < n <= args.tp and torch.cuda.device_count() >= n, \
        f"--gpus {n}: 2 .. --tp {args.tp} GPUs ({torch.cuda.device_count()} GPUs visible)"
    g = torch.Generator().manual_seed(args.seed + 7)
    streams = bf(torch.randn(T, HC * H, generator=g))
    h_post = 2 * torch.rand(T, HC, generator=g)
    h_res = torch.softmax(torch.randn(T, HC, HC, generator=g), -1).reshape(T, HC * HC)
    ws, ts = [], []
    for r in range(n):
        dev = torch.device("cuda", r)
        w = w0 if r == 0 else load_layer(args.model_dir, args.layer, args.tp, r, dev)
        t = gpu_tensors(bank(w), x.to(dev), attn_next=attn_next)
        t["streams"].copy_(streams)
        t["h_post"].copy_(h_post)
        t["h_res"].copy_(h_res)
        ws.append(w)
        ts.append(t)
    mpk = StaticMegakernel(num_gpus=n)
    grids = build_graph(mpk, ts[0], attn_next=None if attn_next is None else {k: attn_next[k] for k in ("rms_eps", "hc_eps",
                                                                                                        "norm_eps", "iters")})
    if os.environ.get("STATIC_L2_PREFETCH_MB"):   # the layer's GEMM weights into L2 at its start (core.cuh kernel_begin)
        mpk.l2_prefetch_weights(float(os.environ["STATIC_L2_PREFETCH_MB"]))
    plan = compiler.plan_from_file(args.plan) if args.plan else round_robin_plan(mpk, grids)
    if len(plan["lists"]) == 1:   # a one-GPU plan file: every rank the same lists
        plan["lists"] = plan["lists"] * n
    os.makedirs(args.out, exist_ok=True)
    compiler.write_plan_file(os.path.join(args.out, "plan.json"), plan)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(args.out, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=ts, out_dir=os.path.join(args.out, "build"), extra_flags=args.nvcc_flag,
                            profile=args.timeline)
    h = Harness(sk)
    try:
        h.launch(FLUSH_BYTES, timeout_s=60)
    except (TimeoutError, RuntimeError) as e:
        print("KERNEL DID NOT FINISH:", e, flush=True)
        os._exit(2)
    torch.cuda.synchronize()
    print(f"layer {args.layer}, TP{args.tp} ranks 0..{n - 1} on {n} GPUs, the end {args.end} in the kernel", flush=True)
    ok = check_end(ts, ws, x, streams, h_post, h_res, attn_next, T)
    first = ts[0]["streams_out"].clone()
    spans = []
    for i in range(args.reps):
        try:
            h.launch(FLUSH_BYTES, timeout_s=60)
        except (TimeoutError, RuntimeError) as e:
            print(f"KERNEL DID NOT FINISH (timed launch {i + 1}):", e, flush=True)
            os._exit(2)
        spans.append(h.span_us())
    torch.cuda.synchronize()
    same = bool((ts[0]["streams_out"].view(torch.int16) == first.view(torch.int16)).all())
    print(f"  new streams of the last timed launch bit-equal to the first: {same}")
    print(f"machine {socket.gethostname()} {torch.cuda.get_device_name(0)} x {n}, grids {info['grids']}, {args.reps} launches: "
          f"span min {min(spans):.1f} median {statistics.median(spans):.1f} max {max(spans):.1f} us")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"plan": args.plan or "round robin", "grids": {str(k): v for k, v in info["grids"].items()}, "end": args.end,
                   "gpus": n, "timing_build": args.timeline, "spans_us": spans}, f, indent=1)
    print("PASS" if ok and same else "FAIL", flush=True)
    sys.exit(0 if ok and same else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, default=3, help="a sparse (MoE) layer: 3..44")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=1, help="1 (--rank alone, the all-reduce writing its own copy), or 2 .. --tp: "
                    "ranks 0 .. N-1 on this process's GPUs 0 .. N-1, the all-reduce over them in the kernel (run_tp)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--x-scale", type=float, default=1.0, help="x = N(0, 1) x this (bf16)")
    ap.add_argument("--plan", help="a plan file of this graph (default: the round-robin plan)")
    ap.add_argument("--end", choices=("mhc", "mhc_pre"), default="mhc",
                    help="the graph's end: the mHC post with random streams and coefficients, or that and the next layer's "
                    "attention mHC pre (what SGLang runs with MOE_STATIC_END=mhc_pre)")
    ap.add_argument("--collect", metavar="FILE", help="a cost collection: every plan of plan_gen.collection_plans and the "
                    "round-robin plan run by this script with --record FILE (--jobs at a time; a run with a result.json is skipped)")
    ap.add_argument("--jobs", type=int, default=4, help="with --collect: plans run at once")
    ap.add_argument("--rounds", type=int, default=3, help="with --collect: random plans per candidate grid of each node")
    ap.add_argument("--gpu-lock", help="a lock file held around this run's GPU work (--collect gives it to its runs)")
    ap.add_argument("--out", default="/tmp/flash_moe")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--nvcc-flag", action="append", default=[])
    ap.add_argument("--timeline", action="store_true", help="the timing build: per node, its window in the last launch")
    ap.add_argument("--record", metavar="FILE", help="the timing build; this run's cost records appended to FILE")
    ap.add_argument("--search", nargs="+", metavar="COSTS", help="cost record files: the solver's plan, from the plan")
    ap.add_argument("--costs-code", default="current", help="with --search: the records of these kernel templates: current "
                    "(static_schedule.code_tag), any, or a tag")
    ap.add_argument("--time", type=float, default=300, help="with --search: the solver's time limit (s)")
    ap.add_argument("--save-y", metavar="FILE", help="save the new streams of the first launch (torch.save): a bit-exact "
                    "baseline")
    ap.add_argument("--same-y", metavar="FILE", help="the new streams of the first launch must equal these saved ones bit for "
                    "bit (a kernel change that keeps the rounding, same --layer --rank --seed)")
    ap.add_argument("--prof-experts", action="store_true", help="the experts node's per-SM stamps (scaled_moe_experts_layer "
                    "prof=): where its time goes, for the last timed launch")
    ap.add_argument("--valid-rows", type=int, default=0, help="the live rows (a speculative verify step): the rows past "
                    "them are not routed (topk_route valid_rows); the checks on the live rows")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda", 0)
    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)["text_config"]
    assert cfg["mlp_layer_types"][args.layer] == "sparse", f"layer {args.layer} is {cfg['mlp_layer_types'][args.layer]}"
    assert (cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"], cfg["num_experts_per_tok"],
            cfg["n_shared_experts"], cfg["routed_scaling_factor"], cfg["swiglu_limit"], cfg["scoring_func"],
            cfg["topk_method"], cfg["n_group"], cfg["norm_topk_prob"]) == \
        (H, I, E, TOPK, 1, SCALE, LIMIT, "sigmoid", "noaux_tc", 1, True), "not GLM-5.3-Flash's MoE"
    assert I // args.tp == 256 * SUB, f"TP{args.tp}: {I // args.tp} features per GPU; the bank takes {SUB} sub-experts of 256"

    w = load_layer(args.model_dir, args.layer, args.tp, args.rank, dev)
    g = torch.Generator().manual_seed(args.seed)
    x = bf(torch.randn(T, H, generator=g) * args.x_scale).to(dev)
    attn_next = None
    if args.end == "mhc_pre":   # the next layer's attention mHC parameters and norm (dsa_fused_demo.load_mhc)
        from dsa_fused_demo import load_mhc
        m = load_mhc(args.model_dir, args.layer + 1)
        attn_next = {"hc_fn": m["hc_attn_fn"], "hc_base": m["hc_attn_base"], "hc_scale": m["hc_attn_scale"],
                     "norm": m["input_layernorm.weight"], "rms_eps": float(cfg["rms_norm_eps"]), "hc_eps": float(cfg["hc_eps"]),
                     "norm_eps": float(cfg["rms_norm_eps"]), "iters": int(cfg["hc_sinkhorn_iters"])}
    if args.gpus > 1:
        return run_tp(args, w, x, attn_next)
    t = gpu_tensors(bank(w), x, attn_next=attn_next)
    streams, h_post = bf(torch.randn(T, HC * H, generator=g)), 2 * torch.rand(T, HC, generator=g)   # random streams and
    h_res = torch.softmax(torch.randn(T, HC, HC, generator=g), -1).reshape(T, HC * HC)              # finished coefficients
    t["streams"].copy_(streams)
    t["h_post"].copy_(h_post)
    t["h_res"].copy_(h_res)
    if args.prof_experts:
        t["moe_prof"] = torch.zeros(torch.cuda.get_device_properties(0).multi_processor_count, 256, dtype=torch.int64,
                                    device=dev)
    mpk = StaticMegakernel(num_gpus=1)
    grids = build_graph(mpk, t, attn_next=None if attn_next is None else {k: attn_next[k] for k in ("rms_eps", "hc_eps",
                                                                                                        "norm_eps", "iters")},
                        valid_rows=args.valid_rows or None)
    plan = compiler.plan_from_file(args.plan) if args.plan else round_robin_plan(mpk, grids)
    os.makedirs(args.out, exist_ok=True)
    if args.collect:
        del w, t
        torch.cuda.empty_cache()
        collect(args, mpk, plan)
        return
    if args.search:
        from mirage.mpk import plan_viewer, search as solver, static_schedule
        code = {"current": static_schedule.code_tag(), "any": None}.get(args.costs_code, args.costs_code)
        costs = solver.Costs(args.search, code=code, gpu=torch.cuda.get_device_name(0), model="prefetch")   # the prefetch
                                                                                                           # cost model
        plan, summary = solver.search_plan(mpk, costs, time_limit_s=args.time, start_from=plan)
        plan_viewer.save(os.path.join(args.out, "plan_view.json"), plan, summary,
                         f"searched from {args.plan or 'the round-robin plan'}")
    compiler.write_plan_file(os.path.join(args.out, "plan.json"), plan)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(args.out, "schedules"))
    profile = args.timeline or args.record is not None
    sk = mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(args.out, "build"), extra_flags=args.nvcc_flag,
                            profile=profile)
    if args.gpu_lock:   # held to the end of the process (the OS lets go of it at exit)
        import fcntl
        lock = open(args.gpu_lock, "a")
        fcntl.flock(lock, fcntl.LOCK_EX)
    h = Harness(sk)
    try:
        h.launch(FLUSH_BYTES, timeout_s=60)
    except (TimeoutError, RuntimeError) as e:
        print("KERNEL DID NOT FINISH:", e, flush=True)
        os._exit(2)
    torch.cuda.synchronize()
    print(f"layer {args.layer}, TP{args.tp} rank {args.rank}: {E} experts x {SUB} sub-experts of {I // args.tp // SUB}, "
          f"x = N(0, 1) x {args.x_scale}, the end {args.end} in the kernel (one GPU)")
    n = args.valid_rows or T   # the live rows checked
    err = check(t, w, x[:n])
    ok = err["ok"] and check_end([t], [w], x, streams, h_post, h_res, attn_next, n)
    if args.valid_rows:   # the idle rows: no expert of their own (the shared ones at weight 0), their routed rows 0
        pr = t["routing_pairs"][n:].view(torch.int64)
        e_idle, w_idle = (pr >> 32).int(), (pr & 0xFFFFFFFF).int().view(torch.float32)
        KS = t["routing_pairs"].shape[1] - SUB
        idle_ok = bool((e_idle[:, :KS] >= E * SUB).all() and (w_idle[:, :KS] == 0).all())
        rr_idle = t["routed_rows"].view(T, -1, H)[n:, :KS].float().abs().max().item()
        print(f"  idle rows {n} .. {T - 1}: routed pairs on the shared experts at weight 0: {idle_ok}; max |routed row| "
              f"{rr_idle:.1e}")
        ok = ok and idle_ok and rr_idle == 0.0
    first = t["streams_out"].clone()
    spans, launches = [], []   # launches (--record): per timed launch, the per-task times and the span
    for _ in range(args.reps):
        h.launch(FLUSH_BYTES, timeout_s=60)
        spans.append(h.span_us())
        if args.record:
            launches.append((h.task_times(), h.timing()))
    same = bool((t["streams_out"].view(torch.int16) == first.view(torch.int16)).all())
    if args.prof_experts:
        prof_report(t["moe_prof"].cpu(), h.timing()[0][0], t)
    if args.timeline:
        p0, p1 = print_timeline(mpk, paths[0], h)
        n_e = len(kernel_pairs(t["routing_pairs"])[0].unique())
        mb = n_e * 3 * t["w2"].shape[2] * SUB * H / 1e6   # each chosen expert: gate, up and down of this GPU's features
        print(f"  experts: {n_e} distinct, {mb:.0f} MB of their weights over {(p1 - p0) / 1e3:.1f} us of the node: "
              f"{mb / ((p1 - p0) / 1e3) / 1e6:.2f} TB/s")
    if args.record:
        from mirage.mpk import cost_records
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        machine = {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc}
        records = cost_records.records_of_run(mpk, paths, launches, os.path.basename(os.path.normpath(args.out)), machine)
        cost_records.append_records(args.record, records)
        print(f"cost records: {len(records)} added to {args.record} (code {records[0]['code'] if records else '-'})")
    print(f"  new streams of the last timed launch bit-equal to the first: {same}")
    print(f"machine {socket.gethostname()} {torch.cuda.get_device_name(0)}, grids {info['grids']}, {args.reps} launches: "
          f"span min {min(spans):.1f} median {statistics.median(spans):.1f} max {max(spans):.1f} us")
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"plan": args.plan or "round robin", "grids": {str(k): v for k, v in info["grids"].items()}, "end": args.end,
                   "timing_build": profile, "spans_us": spans}, f, indent=1)
    ok = ok and same
    if args.save_y:
        torch.save(first.cpu(), args.save_y)
        print(f"  the new streams saved to {args.save_y}")
    if args.same_y:
        base = torch.load(args.same_y)
        eq = bool((first.cpu().view(torch.int16) == base.view(torch.int16)).all())
        print(f"  the new streams bit-equal to {args.same_y}: {eq}")
        ok = ok and eq
    print("PASS" if ok else "FAIL")
    sk.finalize()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
