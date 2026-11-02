"""One K3 KDA layer and its MoE as ONE static megakernel on AMD GPUs (TP8, 8 tokens): the KDA graph of test_kda_layer.py, then
the MoE graph of test_layer_one_gpu.py (demo.build_graph) reading the KDA layer's hidden and new prefix (attn_residual's outputs,
device-scope stores, its counter 0) in the same kernel. The plan: per CU the KDA plan's tasks (simple_plan with --grids), then
the MoE plan's (the best TP8 MoE plan: the solver plan, RMSNorm folded, pack 34:4, regrid 37), node numbers mapped to the
fused graph: a CU starts its MoE tasks (their weight loads) as soon as its KDA tasks are done.
Checks: test_kda_layer.check_all on the KDA part, test_layer_one_gpu.check_all on the MoE part with x = the kernel's hidden and
prefix = its new prefix; launch 1 == launch 0; then the e2e latency like the two tests.

python amd/tests/test_kda_moe_layer.py --gpus 8 --gemm-shuffled [--flush-mb 1024] [--e2e-reps 40] [--flag -DX ...] [--profile]
MK_BACKEND=hip python amd/tests/test_kda_moe_layer.py --codegen ...     (no GPU: layer.cu + hipcc only)
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "amd", "tools"))

MOE_PACK, MOE_REGRID = (34, 4), (37, (64, 1, 1), {36})
AR_CHUNKS = [0]   # attn_residual tasks per token (--ar-chunks)
FOLD_DROP = {38, 41}
# MPK_MOE_NFOLD=1: R summed once per token (sum_rmsnorm) and latent_up's last K part adding the parts and pushing them
# (test_layer_one_gpu.py --lup-finish) instead of the RMSNorm folded into latent_up; MoE nodes 38 sum_rmsnorm, 39 sum_gpus,
# 40 latent_up, 41 residual_add (folded: 38 sum_gpus, 39 latent_up, 40 residual_add). MI355X dec88, MoE alone T = 64 with S
# sent after the shared down and sum_gpus over the experts' CUs: 281.2 / 276.5 vs 301.4 / 298.0 us folded
NFOLD = os.environ.get("MPK_MOE_NFOLD") == "1"


def compute_nodes(mpk):
    from mirage.mpk import compiler
    return [n.graph_idx for n in sorted(compiler.graph_nodes(mpk), key=lambda n: n.graph_idx)]


def build_kda(mpk, w, kda, x_in=None, prefix_in=None):
    """the KDA nodes (test_kda_layer.main's graph); returns (hidden, new_prefix) DTensors. x_in / prefix_in: graph tensors of
    the layer before in the same kernel (--order moe: its MoE's aggregation 1 output and y), else the inputs x / prefix"""
    from mirage.core import bfloat16, float32
    T, H, HD, PROJ_PAD = kda.T, kda.H, kda.HD, kda.PROJ_PAD
    given = {"x": x_in, "prefix": prefix_in}
    att = {k: given[k] if given.get(k) is not None else mpk.attach_input(torch_tensor=w[k], name=k) for k in (
        "x", "wproj_k", "conv_w", "conv_pool", "icw", "ring_k", "ring_v", "info", "wfb", "prm", "state_pool", "ring_g",
        "ring_beta", "oproj_k", "bank", "prefix", "q", "gain", "ow", "hidden", "new_prefix", "slot", "icw_row")}
    proj = mpk.new_tensor(dims=(T, PROJ_PAD), dtype=float32, name="proj_partial")
    rec_in = mpk.new_tensor(dims=(T, PROJ_PAD), dtype=bfloat16, name="rec_in")
    y_rec = mpk.new_tensor(dims=(T, HD), dtype=bfloat16, name="y_rec")
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="oproj_sent")
    mpk.gemm_tile_layer(input=att["x"], weight=att["wproj_k"], output=proj, combine="slots",
                        hybrid=os.environ.get("MPK_KDA_PROJ_HYB") == "1")   # (gemm_tile.cuh gemm_wave HYB; grid x 50)
    mpk.short_conv_silu_layer(input=proj, conv_weight=att["conv_w"], conv_pool=att["conv_pool"], output=rec_in,
                              icw=att["icw"], ring_k=att["ring_k"], ring_v=att["ring_v"], info=att["info"], slot=att["slot"],
                              icw_row=att["icw_row"])
    mpk.delta_rule_recurrence_layer(rec_in=rec_in, wfb=att["wfb"], prm=att["prm"], state_pool=att["state_pool"], output=y_rec,
                                    ring_g=att["ring_g"], ring_beta=att["ring_beta"], info=att["info"], slot=att["slot"],
                                    col_g=kda.COL_G, col_fa=kda.COL_FA, col_beta=kda.COL_BETA, onorm_eps=kda.ONORM_EPS)
    mpk.gemm_tile_layer(input=y_rec, weight=att["oproj_k"], output=sent, combine="push_all", input_polled=True)
    mpk.attn_residual_layer(input=sent, bank=att["bank"], prefix=att["prefix"], q=att["q"], gain=att["gain"], ow=att["ow"],
                            output=att["hidden"], u_output=att["new_prefix"], info=att["info"], eps=kda.RES_EPS, chunks=AR_CHUNKS[0],
                            prefix_wait=agg1_prefix_wait(mpk) if prefix_in is not None else None)
    return att["hidden"], att["new_prefix"]


def agg1_prefix_wait(mpk, last=False):
    """(counter, need) of the posted moe_agg1 node's second count (its y, the bank row, agg_prefix written): the next
    attention's attn_residual waits for it before reading its prefix and bank rows"""
    n = [v for v in mpk._static_nodes.values() if v["name"] == "moe_agg1"]
    assert (last or len(n) == 1) and n[-1]["counter"] >= 0, "--order moe: one posted moe_agg1 node"
    return (n[-1]["counter"] + 1, mpk.tokens)   # (one count per moe_agg1 task: a task per token)


def moe_plan(mpk_moe, plan_path, aggn=None):
    """the best TP8 MoE plan on a MoE-only graph (test_layer_one_gpu.py --fold --pack 34:4 --regrid 37:64,1,1:36); a graph of
    other than 8 tokens: the plan file's per-token nodes re-dealt (plan_one_gpu.tokens_plan)"""
    from mirage.mpk import compiler
    from plan_one_gpu import drop_plan, fewer_gpus_plan, follow_plan, lup_finish_plan, pack_plan, regrid_plan, tokens_plan
    plan = (lup_finish_plan if NFOLD else lambda p_: drop_plan(p_, FOLD_DROP))(compiler.plan_from_file(plan_path))
    res = 41 if NFOLD else 40
    if aggn is not None and aggn != res:   # (--agg: the plan's last node, 40 = residual_add (NFOLD 41), is moe_agg1 in this graph)
        ren = lambda k: aggn if int(k) == res else int(k)
        plan = {**plan, "grids": {ren(k): v for k, v in plan["grids"].items()},
                "lists": [[[(ren(t[0]), tuple(t[1])) for t in cu] for cu in gpu] for gpu in plan["lists"]]}
    if mpk_moe.tokens != 8:
        plan = tokens_plan(mpk_moe, plan, 8)
    plan = fewer_gpus_plan(mpk_moe, plan)
    # (MPK_MOE_PACK: the shared down's tasks per CU; dec45, MoE alone T = 1: 2 per CU 68.5 vs 4 75.6 / 83.8 us)
    plan = pack_plan(plan, MOE_PACK[0], int(os.environ.get("MPK_MOE_PACK", MOE_PACK[1])))
    if os.environ.get("MPK_MOE_SEND_FOLLOW") == "1":   # S's send (35) right after the down tasks (34) on their CUs
        plan = follow_plan(plan, 35, 34)
    rn, rg, ra = MOE_REGRID   # (MPK_MOE_R_TASKS: R's allreduce_send tasks; dec42, T = 64: 128 tasks 293.5 vs 64 308.1 us)
    plan = regrid_plan(mpk_moe, plan, rn, (int(os.environ.get("MPK_MOE_R_TASKS", rg[0])), 1, 1), ra)
    for rg_ in os.environ.get("MPK_MOE_REGRID", "").replace(";", " ").split():   # NODE:GX,GY,GZ[:AFTER,...] (plan_one_gpu.regrid_plan; MoE numbers)
        f = rg_.split(":")
        plan = regrid_plan(mpk_moe, plan, int(f[0]), tuple(int(v) for v in f[1].split(",")),
                           {int(v) for v in f[2].split(",")} if len(f) > 2 and f[2] else set())
    for fo in os.environ.get("MPK_MOE_FOLLOW", "").split():   # NODE:LEAD[:AFTER,...] (plan_one_gpu.follow_plan; MoE numbers)
        f = fo.split(":")
        plan = follow_plan(plan, int(f[0]), int(f[1]), {int(v) for v in f[2].split(",")} if len(f) > 2 and f[2] else set(),
                           {int(v) for v in f[3].split(",")} if len(f) > 3 and f[3] else set())
    return plan


def fused_plan(kplan, kmap, mplan, mmap, early_front=None):
    """per GPU and CU: the attention plan's tasks, then the MoE plan's, node numbers mapped (kmap / mmap: old -> fused).
    early_front = (late attention nodes, the MoE experts node; old numbers): the MoE tasks before each CU's experts task (the
    front: router, latent_down, shared expert, ...) taken out of the CUs' lists in the plan's order (by position, then CU) and
    dealt round-robin to the CUs without a late attention task (free after the projection), so they start (their weights
    in flight, STATIC_GEMM_LDSW=4: the whole slice in LDS) while the attention's tail runs; each list stays in graph order."""
    grids = {kmap[int(k)]: tuple(v) for k, v in kplan["grids"].items()}
    grids.update({mmap[int(k)]: tuple(v) for k, v in mplan["grids"].items()})
    lists = []
    for kl, ml in zip(kplan["lists"], mplan["lists"]):
        if early_front is None:
            lists.append([[(kmap[t[0]], tuple(t[1])) for t in a] + [(mmap[t[0]], tuple(t[1])) for t in b] for a, b in zip(kl, ml)])
            continue
        late, experts = early_front
        fronts, rests = [], []
        for b in ml:
            i = next((j for j, t in enumerate(b) if t[0] == experts), len(b))
            fronts.append(b[:i])
            rests.append(b[i:])
        pool = [f[k] for k in range(max(len(f) for f in fronts)) for f in fronts if k < len(f)]
        early = [c for c, a in enumerate(kl) if not any(t[0] in late for t in a)] or list(range(len(kl)))
        mine = [[] for _ in kl]
        for i, t in enumerate(pool):
            mine[early[i % len(early)]].append(t)
        gl = []
        for c, a in enumerate(kl):
            front = sorted(mine[c], key=lambda t: t[0])   # (graph order within the CU)
            gl.append([(kmap[t[0]], tuple(t[1])) for t in a] + [(mmap[t[0]], tuple(t[1])) for t in front + rests[c]])
        lists.append(gl)
    return {"grids": grids, "lists": lists}


def moe_first_plan(mplan, mmap, kplan, kmap):
    """--order moe: per GPU and CU the MoE plan's tasks, then the attention plan's (node numbers mapped): a CU starts its
    attention tasks (the projection's weight loads) as soon as its MoE tasks are done, while the MoE's tail runs"""
    grids = {mmap[int(k)]: tuple(v) for k, v in mplan["grids"].items()}
    grids.update({kmap[int(k)]: tuple(v) for k, v in kplan["grids"].items()})
    lists = [[[(mmap[t[0]], tuple(t[1])) for t in b] + [(kmap[t[0]], tuple(t[1])) for t in a] for a, b in zip(kl, ml)]
             for kl, ml in zip(kplan["lists"], mplan["lists"])]
    return {"grids": grids, "lists": lists}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", type=int, default=8, choices=(8,))
    p.add_argument("--out", default="kda_moe_layer")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-moe-in", default=None, help="the MoE part's x / prefix of launch 1 to this .pt (test_layer_one_gpu.py --moe-in)")
    p.add_argument("--wait", type=float, default=60)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--flush-mb", type=int, default=0)
    p.add_argument("--e2e-reps", type=int, default=20)
    p.add_argument("--ar-chunks", type=int, default=0, help="attn_residual tasks per token (0: H / 1024; 4: 1792-column chunks)")
    p.add_argument("--gemm-shuffled", action="store_true")
    p.add_argument("--attn", default="kda", choices=("kda", "mla"), help="the attention part: KDA or MLA (test_mla_layer.py's graph)")
    p.add_argument("--order", default="attn", choices=("attn", "moe"), help="moe: the MoE layer with the next layer's "
                   "aggregation 1 first (--agg NB,WRITE,2), then the NEXT layer's attention part on its outputs (x = "
                   "aggregation 1's output, prefix = the MoE's y); split loops at the first attention node (MPK_SPLIT_LOOP)")
    p.add_argument("--grids", default=None, help="the attention plan's grids (that test's node numbers; default: KDA "
                   '{"27":[100,7,1],"30":[224,1,1]}, MLA {"29":[34,7,1],"31":[144,1,1],"32":[24,7,1],"37":[224,1,1]})')
    p.add_argument("--first", default=None, help="comma-separated attention node numbers: plan_one_gpu.first_plan")
    p.add_argument("--early-front", action="store_true", help="fused_plan's early_front: the MoE front on the CUs free after the "
                   "projection (no recurrence / o_proj / residual task; MLA: no absorb .. residual task)")
    p.add_argument("--today", action="store_true", help="KDA: light 28:27 + apart 31,30 (use --grids with o_proj 112); MoE: apart "
                   "39,37,38, regrid 28:56,8,1 and 40:64,1,1")
    p.add_argument("--kv-rows", type=int, default=37)
    p.add_argument("--die-group", type=int, default=None, help="MLA: plan_one_gpu.die_plan (test_mla_layer.py numbering)")
    p.add_argument("--rank", default=None, help="MLA: N,F1,F2..: plan_one_gpu.rank_plan (test_mla_layer.py numbering)")
    p.add_argument("--kv-splits", type=int, default=1)
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--norm-groups", type=int, default=1)
    p.add_argument("--plan", default=os.path.join(ROOT, "demo", "kimi_k3_moe", "solver_plan_256cu.json"))
    p.add_argument("--codegen", action="store_true")
    p.add_argument("--decode", action="store_true", help="plain decode: the attention part of test_kda_decode.py / "
                   "test_mla_decode.py (per-request slots / KV rows) with --tokens requests; grids in those tests' numbering")
    p.add_argument("--tokens", type=int, default=8, help="--decode: requests (batch size)")
    p.add_argument("--rpt", type=int, default=1, help="--decode KDA: requests per delta_rule_decode task")
    p.add_argument("--agg", default=None, metavar="NB,WRITE[,KIND]", help="the MoE part ends with the next layer's "
                   "aggregation 1 (test_layer_one_gpu.py --agg; KIND 2: tasks/moe_agg1.cuh)")
    p.add_argument("--attn-only", action="store_true", help="a probe: the fused build, only the attention part's tasks run "
                   "(no MoE task in the lists; checks skipped, timing only)")
    p.add_argument("--dump-stamps", action="store_true", help="the timed launches' per-CU stamps to <out>/stamps_timed.npy")
    p.add_argument("--save-outs", default=None, help="launch 1's outputs of every GPU to this .pt (test_rank_lib.py compares)")
    a = p.parse_args()
    AR_CHUNKS[0] = a.ar_chunks
    os.environ["MPK_LATENT_UP_FINISH" if NFOLD else "MPK_RMSNORM_FOLD"] = "1"
    if a.decode:
        os.environ["MPK_TOKENS"] = str(a.tokens)
    else:
        assert a.tokens == 8, "--tokens: --decode only"
    if a.codegen and not torch.cuda.is_available():
        for fn in ("empty", "zeros", "ones", "full", "arange", "tensor", "empty_like", "zeros_like", "randn", "randint", "cat",
                   "randperm"):
            def on_cpu(*args, _orig=getattr(torch, fn), **kw):
                if str(kw.get("device", "")).startswith("cuda"):
                    kw["device"] = "cpu"
                return _orig(*args, **kw)
            setattr(torch, fn, on_cpu)

        class Props:
            major, minor, multi_processor_count, gcnArchName = 9, 5, 256, "gfx950"
            name, total_memory = "no GPU (codegen check)", 288 << 30
        torch.cuda.get_device_properties = lambda *_: Props
        torch.cuda.get_device_name = lambda *_: Props.name
        torch.cuda.current_device = lambda: 0
        torch.cuda.device_count = lambda: 8
        torch.cuda.set_device = lambda *_: None
        _gen = torch.Generator
        torch.Generator = lambda device=None: _gen()
    import test_kda_layer as kda
    import test_layer_one_gpu as moe
    import test_mla_layer as mla
    if a.decode:   # (the decode tests' modules in the verify ones' place: their T is the batch size)
        import test_kda_decode as kda
        import test_mla_decode as mla
        kda.T = mla.T = a.tokens
        kda.H = 7168
    if a.grids is None:
        a.grids = ('{"21":[100,7,1],"24":[112,1,1]}' if a.decode else '{"27":[100,7,1],"30":[224,1,1]}') if a.attn == "kda" \
            else '{"29":[34,7,1],"31":[144,1,1],"32":[24,7,1],"37":[224,1,1]}'
    from mirage.mpk import compiler
    from mirage.mpk.static_harness import Harness
    from mirage.mpk.static_megakernel import StaticMegakernel, shuffle_gemm_weight
    from plan_one_gpu import simple_plan
    moe.FLAGS[:] = a.flag
    if a.agg:   # (as test_layer_one_gpu.py --agg: the MoE part ends with the next layer's aggregation 1)
        f_ = [int(v) for v in a.agg.split(",")]
        moe.AGG[:] = f_[:2]
        os.environ["MPK_MOE_AGG"] = str(f_[2] if len(f_) > 2 else 2)
    d = moe.load_demo()
    n = a.gpus
    devs = [torch.device("cpu")] if a.codegen else [torch.device("cuda", g) for g in range(n)]
    if a.attn == "kda":
        wk = [kda.random_weights(devs[0], a.seed)]
        wk += [kda.random_weights(devs[g], a.seed + g, common=wk[0]) for g in range(1, len(devs))]
        pools0 = [(w["conv_pool"].cpu().clone(), w["state_pool"].cpu().clone()) for w in wk]
        for w in wk:
            for name in ("wproj", "oproj"):
                w[name + "_k"] = shuffle_gemm_weight(w[name]) if a.gemm_shuffled else w[name]
        if a.decode:
            build_attn = lambda m, w, **kw: kda.build_graph(m, w, a.rpt, **kw)
        else:
            build_attn = lambda m, w, **kw: build_kda(m, w, kda, **kw)
    else:
        if a.decode:
            lay = mla.layout(a.seed, a.kv_rows)
            wk = [mla.random_weights(devs[0], a.seed, lay)]
            wk += [mla.random_weights(devs[g], a.seed + g, lay, common=wk[0]) for g in range(1, len(devs))]
        else:
            wk = [mla.random_weights(devs[0], a.seed, a.kv_rows)]
            wk += [mla.random_weights(devs[g], a.seed + g, a.kv_rows, common=wk[0]) for g in range(1, len(devs))]
        cache0 = wk[0]["cache"].cpu().clone()
        mla.prep_weights(wk, a.gemm_shuffled)
        build_attn = lambda m, w, **kw: mla.build_graph(m, w, a, **kw)
    ts = float(os.environ.get("MPK_TOKEN_SHARE", "0.6"))   # (the MoE input's common share: test_layer_one_gpu.py --token-share)
    wm = [moe.random_weights(d, devs[0], a.seed, token_share=ts)]
    wm += [moe.random_weights(d, devs[g], a.seed + g, common=wm[0], token_share=ts) for g in range(1, len(devs))]
    rr = os.environ.get("MPK_REAL_ROUTE", "")   # (PATH:STEP - the server's routing: one decode step's MoE input of a real layer and
    if rr:                                       #  that layer's router weight / bias, saved by the server hook MPK_K3_SAVE_H)
        path, step = rr.rsplit(":", 1)
        R = torch.load(path)
        x = R["moe_in"][int(step)]
        assert tuple(x.shape) == tuple(wm[0]["moe_in"].shape), (x.shape, wm[0]["moe_in"].shape)
        for w in wm:
            w["moe_in"].copy_(x)
            w["router_weight"].copy_(R["router_weight"])
            w["score_correction_bias"].copy_(R["score_correction_bias"])
        print(f"real routing: {path} layer {R['layer']} step {step} of {R['moe_in'].shape[0]}", flush=True)
    for w in wm:
        for name in moe.GEMM_WEIGHTS:
            if a.gemm_shuffled:
                sh = shuffle_gemm_weight(w[name])
                moe.ROWMAJOR[sh.data_ptr()] = w[name]
                w[name] = sh
    torch.cuda.set_device(0)
    # the two plans on their own graphs, then the fused graph
    mk = StaticMegakernel(num_gpus=n)
    if a.gemm_shuffled:
        mk.gemm_weight_layout = "shuffled"
    build_attn(mk, wk[0])
    kplan = simple_plan(mk, json.loads(a.grids))
    if a.first:
        from plan_one_gpu import first_plan
        kplan = first_plan(kplan, {int(x) for x in a.first.split(",")})
    if a.die_group is not None:   # (as test_mla_layer.py: plan_one_gpu.die_plan on the attention plan)
        from plan_one_gpu import die_plan
        kplan = die_plan(kplan, a.die_group)
    if a.rank:   # (as test_mla_layer.py: plan_one_gpu.rank_plan on the attention plan)
        from plan_one_gpu import rank_plan
        rk = [int(x) for x in a.rank.split(",")]
        kplan = rank_plan(kplan, rk[0], 1, set(rk[1:]))
    mm = StaticMegakernel(num_gpus=n)
    if a.gemm_shuffled:
        mm.gemm_weight_layout = "shuffled"
    d.build_graph(mm, wm[0])
    # (--agg: the plan's last node, 40 = residual_add, is moe_agg1 in this graph: after its attached inputs)
    aggn = next(x.graph_idx for x in compiler.graph_nodes(mm) if x.name == "moe_agg1") if a.agg else None
    mplan = moe_plan(mm, a.plan, aggn)
    if a.today:   # 10-15's passes on the two plans (test_kda_layer.py / test_layer_one_gpu.py numbering)
        from plan_one_gpu import apart_plan, light_plan, regrid_plan
        if a.attn == "kda" and not a.decode:   # (MLA: its plan options --die-group / --rank instead)
            kplan = apart_plan(light_plan(kplan, 28, 27), 31, {30})
        # (MPK_LUP_APART_EXTRA="n,m": latent_up also off the CUs of these MoE nodes - T 8, dec228: the latent_up tasks behind S's
        # send (35) started ~6 us later and ended last)
        xa = {int(v) for v in os.environ.get("MPK_LUP_APART_EXTRA", "").split(",") if v}
        mplan = apart_plan(mplan, 40, {37, 38, 39} | xa) if NFOLD else apart_plan(mplan, 39, {37, 38} | xa)
        # (MPK_LDOWN_GRID: latent_down's grid, default 56 x 8: 7 row tiles per GPU x 8 K parts)
        mplan = regrid_plan(mm, mplan, 28, tuple(int(v) for v in os.environ.get("MPK_LDOWN_GRID", "56,8,1").split(",")), set())
        last = next(x.graph_idx for x in compiler.graph_nodes(mm) if x.name in ("moe_agg1", "residual_add"))
        mplan = regrid_plan(mm, mplan, last, (mm.tokens, 1, 1) if a.agg else (64, 1, 1), {36, 37, 38, 39} | ({40} if NFOLD else set()))   # (moe_agg1: a task per token)
        if os.environ.get("MPK_MOE_AGG_PACK"):   # moe_agg1's tasks k per CU: with MPK_ATTN_OFF_AGG fewer CUs without a projection
            from plan_one_gpu import pack_plan    # task (T = 8: 252 CUs hold the KDA projection's 252 q / k / v tasks at once)
            mplan = pack_plan(mplan, last, int(os.environ["MPK_MOE_AGG_PACK"]))
            print(f"MPK_MOE_AGG_PACK: node {last}'s tasks {os.environ['MPK_MOE_AGG_PACK']} per CU", flush=True)
    for fo in os.environ.get("MPK_MOE_FOLLOW_LATE", "").split():   # as MPK_MOE_FOLLOW, after --today's passes (its apart of 39)
        from plan_one_gpu import follow_plan
        f = fo.split(":")
        mplan = follow_plan(mplan, int(f[0]), int(f[1]), {int(v) for v in f[2].split(",")} if len(f) > 2 and f[2] else set(),
                            {int(v) for v in f[3].split(",")} if len(f) > 3 and f[3] else set())
        print(f"MPK_MOE_FOLLOW_LATE: node {f[0]} over node {f[1]}'s CUs", flush=True)
    mpk = StaticMegakernel(num_gpus=n)
    if a.gemm_shuffled:
        mpk.gemm_weight_layout = "shuffled"
    moe_first = a.order == "moe"
    if moe_first:   # the MoE (+ aggregation 1, posted: its output read in this kernel), then the next layer's attention
        assert a.agg and os.environ.get("MPK_MOE_AGG") == "2", "--order moe: --agg NB,WRITE,2"
        TT = a.tokens if a.decode else kda.T
        for w in wm:   # (the next attention's prefix and the MoE's output without the prefix, written by moe_agg1)
            w["agg_prefix"] = torch.empty(TT, 7168, dtype=torch.bfloat16, device=w["moe_in"].device)
            w["agg_bare"] = torch.empty(TT, 7168, dtype=torch.bfloat16, device=w["moe_in"].device)
            if os.environ.get("MPK_AGG_MIX") == "1":   # (the pre-norm mixture: demo.build_graph agg_mix)
                w["agg_mix"] = torch.empty(TT, 7168, dtype=torch.bfloat16, device=w["moe_in"].device)
        y_m, hid_m = d.build_graph(mpk, wm[0], agg_posted=True)
        if a.decode:   # (the decode builders take attn_residual's wait on moe_agg1 as an argument)
            build_attn(mpk, wk[0], x_in=hid_m, prefix_in=y_m, prefix_wait=agg1_prefix_wait(mpk))
        elif a.attn == "kda":
            build_kda(mpk, wk[0], kda, x_in=hid_m, prefix_in=y_m)
        else:
            mla.build_graph(mpk, wk[0], a, x_in=hid_m, prefix_in=y_m)
    else:
        hidden, new_prefix = build_attn(mpk, wk[0])
        d.build_graph(mpk, wm[0], x_in=hidden, prefix_in=new_prefix)
    kn, mn, fn = compute_nodes(mk), compute_nodes(mm), compute_nodes(mpk)
    assert len(fn) == len(kn) + len(mn), (len(fn), len(kn), len(mn))
    late = {29, 30, 31} if a.attn == "kda" else {34, 35, 36, 37, 38}
    if moe_first:
        plan = moe_first_plan(mplan, dict(zip(mn, fn[:len(mn)])), kplan, dict(zip(kn, fn[len(mn):])))
        split = fn[len(mn)]
        if os.environ.get("MPK_SPLIT_LOOP", str(split)) != str(split):
            raise SystemExit(f"--order moe: MPK_SPLIT_LOOP must be the first attention node, {split}")
        os.environ["MPK_SPLIT_LOOP"] = str(split)
        print(f"--order moe: MoE nodes {fn[:len(mn)]}, {a.attn.upper()} nodes {fn[len(mn):]}; MPK_SPLIT_LOOP={split}", flush=True)
        off = os.environ.get("MPK_ATTN_OFF_AGG", "")   # "1": the attention's first node off moe_agg1's CUs; "n,m,..": those nodes
        if off and off != "0" and aggn is not None:
            from plan_one_gpu import off_plan
            agg_f = dict(zip(mn, fn[:len(mn)]))[aggn]
            for nd in [split] if off == "1" else [int(x) for x in off.split(",")]:
                assert nd in fn[len(mn):], f"MPK_ATTN_OFF_AGG: {nd} is not an attention node {fn[len(mn):]}"
                plan = off_plan(plan, nd, {agg_f})
                print(f"MPK_ATTN_OFF_AGG: node {nd}'s tasks off the CUs of node {agg_f} (moe_agg1)", flush=True)
        for oo in os.environ.get("MPK_ATTN_OFF2", "").replace(",", " ").split():   # NODE:LEAD[:MOST], comma-separated (fused numbers): NODE's tasks off the CUs with
            from plan_one_gpu import off_plan                       # more than MOST (default 1) LEAD tasks
            f = [int(x) for x in oo.split(":")]
            nd, ld, most = f[0], f[1], (f[2] if len(f) > 2 else 1)
            pref, pmin = (f[3], f[4] if len(f) > 4 else 1) if len(f) > 3 else (None, 1)   # (NODE:LEAD:MOST:PREF[:PMIN])
            plan = off_plan(plan, nd, {ld}, most=most, prefer=pref, pmin=pmin)
            print(f"MPK_ATTN_OFF2: node {nd}'s tasks off the CUs with more than {most} tasks of node {ld}"
                  + (f", onto CUs with >= {pmin} tasks of node {pref} first" if pref is not None else ""), flush=True)
        for oo in os.environ.get("MPK_ATTN_OFF_SET", "").replace(",", " ").split():   # NODE:A1+A2+.. (fused numbers; entries by , or space): NODE's tasks off the CUs holding a
            from plan_one_gpu import off_plan                         # task of any of A1, A2, .. in one pass (two OFF passes in a row
            f = oo.split(":")                                         # can move tasks back onto the first one's CUs: dec239, T 16 HYB4
            nd, av = int(f[0]), {int(x) for x in f[1].split("+")}     # off latent_up's CUs put 16 back on moe_agg1's, ending +20 us)
            plan = off_plan(plan, nd, av)
            print(f"MPK_ATTN_OFF_SET: node {nd}'s tasks off the CUs of nodes {sorted(av)}", flush=True)
        sp = os.environ.get("MPK_KDA_SPLIT_PROJ", "")
        if sp in ("1", "2") and a.attn == "kda":   # the projection's g / f_a / beta rows last
            from plan_one_gpu import late_plan
            gx = {int(k): v for k, v in plan["grids"].items()}[split][0]
            rows0 = kda.COL_G // (kda.PROJ_PAD // gx)   # (the first such row block of the grid)
            if sp == "1":
                plan = late_plan(plan, split, lambda pos: pos[0] >= rows0)
                print(f"MPK_KDA_SPLIT_PROJ: node {split}'s row blocks >= {rows0} in its last slots", flush=True)
            else:   # 2: q / k / v, then f_a / beta (the recurrence's start reads them), then g (only before its output)
                rows1 = kda.COL_FA // (kda.PROJ_PAD // gx)
                plan = late_plan(plan, split, lambda pos: 0 if pos[0] < rows0 else 2 if pos[0] < rows1 else 1)
                print(f"MPK_KDA_SPLIT_PROJ=2: node {split}'s row blocks {rows0}..{rows1 - 1} (g) in its last slots, "
                      f">= {rows1} (f_a / beta) before them", flush=True)
    else:
        plan = fused_plan(kplan, dict(zip(kn, fn[:len(kn)])), mplan, dict(zip(mn, fn[len(kn):])),
                          early_front=(late, 36) if a.early_front else None)
    if os.environ.get("MPK_EMPTY_PLAN") == "1":   # (a probe: no task at all - the kernel's fixed cost: launch, kernel_begin's
        # resets, kernel_end's, completion; checks fail by design, MPK_TIME_ANYWAY=1 times it)
        plan = {**plan, "lists": [[[] for cu in gpu] for gpu in plan["lists"]]}
    if a.attn_only:   # (a probe: the fused build with only the attention part's tasks in the lists; no check, timing only)
        kset = set(fn[:len(kn)])
        plan = {**plan, "lists": [[[t for t in cu if t[0] in kset] for cu in gpu] for gpu in plan["lists"]]}
    print(f"fused graph: {a.attn.upper()} nodes {fn[:len(kn)]}, MoE nodes {fn[len(kn):]}; tasks per CU (GPU 0) max "
          f"{max(len(l) for l in plan['lists'][0])}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    for ap in os.environ.get("MPK_FUSED_APART", "").split(";"):   # NODE:AVOID,..;.. (fused node numbers; plan_one_gpu.apart_plan:
        # every task of NODE on a CU holding none of AVOID - e.g. 30:26,28, the shared gate_up off the router / latent_down CUs)
        if ap:   # (as many of NODE's tasks as there are free CUs move; the rest stay where they are)
            nd, av = ap.split(":")
            nd, av = int(nd), {int(v) for v in av.split(",")}
            out, nmv = [], []
            for lists in plan["lists"]:
                bad = [any(t[0] in av for t in cu) for cu in lists]
                new = [list(cu) for cu in lists]
                free = sorted((i for i in range(len(lists)) if not bad[i] and not any(t[0] == nd for t in lists[i])),
                              key=lambda c: (len(new[c]), c))
                free = free[:max(0, len(free) - 24)]   # (CUs left as they were:
                # STATIC_RESET_FIRST_NODE needs CUs that start with the experts node)
                moved = 0
                for i, cu in enumerate(lists):
                    for task in [t for t in cu if t[0] == nd] if bad[i] else []:
                        if moved == len(free):
                            break
                        new[i].remove(task)
                        f = free[moved]
                        new[f].insert(max((j + 1 for j, t in enumerate(new[f]) if t[0] <= task[0]), default=0), task)
                        moved += 1
                out.append(new)
                nmv.append(moved)
            plan = {**plan, "lists": out}
            print(f"MPK_FUSED_APART: node {nd} off the CUs of {sorted(av)}: {nmv} tasks moved per GPU", flush=True)
    compiler.write_plan_file(os.path.join(a.out, f"plan_{n}gpu.json"), plan)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
    ws = [{**x, **y} for x, y in zip(wk, wm)]
    if a.codegen:
        import re
        import subprocess
        from mirage.mpk import static_schedule
        code = static_schedule.generate_code(mpk, [json.load(open(q)) for q in paths], a.profile)
        cu = os.path.join(a.out, "layer.cu")
        with open(cu, "w") as f:
            f.write(code)
        cmd = static_schedule.compile_command(cu, os.path.join(a.out, "layer.so"), a.flag + ["--save-temps"], arch="gfx950")
        r = subprocess.run(cmd, cwd=a.out, capture_output=True, text=True)
        print(f"layer.cu {len(code.splitlines())} lines; hipcc exit {r.returncode}", flush=True)
        if r.returncode:
            print("\n".join(l for l in r.stderr.splitlines() if "error" in l)[:6000])
        else:
            sasm = open(os.path.join(a.out, "layer-hip-amdgcn-amd-amdhsa-gfx950.s")).read()
            print({k: re.search(rf"; {k}: (\d+)", sasm).group(1) for k in ("NumVgprs", "NumAgprs", "ScratchSize", "Occupancy")})
        return
    # (the router's and latent_down's K-part slots are reset in the kernel only: test_layer_one_gpu.py binds them otherwise)
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag, profile=a.profile)
    h = Harness(sk)
    if os.environ.get("MPK_WARM_TENSORS"):   # (an experiment: these weights read after each flush, in the caches at the launch)
        names = os.environ["MPK_WARM_TENSORS"].split(",")
        h.warm = [[w[n] for n in names] for w in ws]
        print(f"MPK_WARM_TENSORS: {names}, {sum(w[n].numel() * w[n].element_size() for n in names for w in ws[:1]) / 1e6:.1f} MB "
              f"per GPU read after each flush", flush=True)
    outs = ((("hidden", "new_prefix", "conv_pool", "state_pool") if a.decode else
             ("hidden", "new_prefix", "ring_k", "ring_v", "ring_g", "ring_beta", "icw")) if a.attn == "kda" else
            ("hidden", "new_prefix", "cache")) + ("moe_out", "routing_pairs") + (("agg_out",) if a.agg else ()) + \
        (("agg_prefix", "agg_bare") if moe_first else ()) + \
        (("agg_mix",) if moe_first and os.environ.get("MPK_AGG_MIX") == "1" else ())
    bank0 = [w["agg_bank"].clone() for w in ws] if a.agg else None
    runs = []

    def restore():   # (--decode KDA: the pools are written in place; launch 1 == launch 0 needs the same pools before each)
        if a.decode and a.attn == "kda":
            for w, (cp, sp) in zip(wk, pools0):
                w["conv_pool"].copy_(cp)
                w["state_pool"].copy_(sp)
            for g in range(n):
                torch.cuda.synchronize(g)
    if os.environ.get("MPK_BOUND_FRONT_READY") == "1":   # (a bound: pairs / z_q never reset; two launches fill both exchange sets)
        for _ in range(2):
            restore()
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
            for g in range(n):
                torch.cuda.synchronize(g)
    for launch in range(2):
        restore()
        try:
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
        except (TimeoutError, RuntimeError) as e:
            print(f"KERNEL DID NOT FINISH (launch {launch}): {e}", flush=True)
            os._exit(2)
        for g in range(n):
            torch.cuda.synchronize(g)
        runs.append([{k: w[k].clone() for k in outs} for w in ws])
        print(f"launch {launch}: e2e {h.e2e_us():.1f} us, span {h.span_us():.1f} us", flush=True)
        if a.profile and launch == 1:
            moe.node_times(h, paths, compiler.graph_nodes(mpk))
            try:   # the moe item trace (as test_layer_one_gpu.py): <out>/trace
                moe.read_curve(h, paths, compiler.graph_nodes(mpk), 8 * 16 * (384 // 32), dump_dir=os.path.join(a.out, "trace"))
            except Exception as e:
                print(f"  (no moe trace: {e})", flush=True)
            import numpy as np
            dd = os.path.join(a.out, "trace")   # every GPU's stamps and task times (as test_layer_one_gpu.py), for xgpu_an.py
            os.makedirs(dd, exist_ok=True)
            hl = h.layout   # (not `lay`: the MLA build's layout of the same name is checked below)
            for g, (st, _) in enumerate(h.timing()):
                np.save(os.path.join(dd, f"stamps_gpu{g}.npy"),
                        h._array(g, "stamps", np.int64).reshape(hl["num_sms"], hl["num_stamps"]))
                np.save(os.path.join(dd, f"task_times_all_gpu{g}.npy"), np.array(h.task_times()[g], dtype=np.uint64))
                with open(os.path.join(dd, f"start_all_gpu{g}.txt"), "w") as f:
                    f.write(str(st))
            print("XGDUMP", dd, flush=True)
    rp = runs[1][0]["routing_pairs"].cpu()   # (the distinct experts launch 1 routed to: the server's bs 8 ~105-113, srvc1)
    print(f"routing: {int(torch.unique(rp >> 32).numel())} distinct experts over {rp.shape[0]} tokens (MPK_TOKEN_SHARE "
          f"{os.environ.get('MPK_TOKEN_SHARE', '0.6')})", flush=True)
    rep = moe.Report()
    if os.environ.get("MPK_SKIP_CHECKS") != "1":   # (MPK_SKIP_CHECKS=1: a timing probe whose outputs are not meant to be right)
        if moe_first:   # (the attention part's inputs: launch 1's aggregation 1 output and agg_prefix, the same in every launch)
            for w, r in zip(wk, runs[1]):
                w["x"], w["prefix"] = r["agg_out"], r["agg_prefix"]
            rep.check("agg_bare: y = bf16(prefix + agg_bare)",
                      all(moe.same_bits([(r["agg_bare"].float() + w["moe_prefix"].float()).to(torch.bfloat16), r["moe_out"]])
                          for r, w in zip(runs[1], ws)), f"bits, all {n} GPUs")
            rep.check("agg_prefix (the attention's prefix): " + ("zero (bank row written)" if moe.AGG[1] else "y"),
                      all(moe.same_bits([r["agg_prefix"], torch.zeros_like(r["moe_out"]) if moe.AGG[1] else r["moe_out"]])
                          for r in runs[1]), f"bits, all {n} GPUs")
        print(f"checks of launch 1, {a.attn.upper()} part{' (x = the MoE part' + chr(39) + 's aggregation 1, prefix = its y)' if moe_first else ''}:")
        if a.attn == "kda":
            kda.check_all(wk, h, rep, pools0)
        elif a.decode:
            mla.check_all(wk, h, rep, cache0, lay)
        else:
            mla.check_all(wk, h, rep, cache0, a.kv_rows)
        print("checks of launch 1, MoE part (x = the kernel's hidden, prefix = its new prefix):" if not moe_first else
              "checks of launch 1, MoE part (x, prefix: the graph inputs):")
        for w in ws:
            if not moe_first:
                w["moe_in"], w["moe_prefix"] = w["hidden"], w["new_prefix"]
        if a.save_moe_in:   # the MoE part's inputs (x, prefix) of launch 1 per GPU: test_layer_one_gpu.py --moe-in runs the MoE alone on them
            torch.save([{"moe_in": w["moe_in"].cpu(), "moe_prefix": w["moe_prefix"].cpu()} for w in ws], a.save_moe_in)
        moe.check_all(d, ws, rep, fold=not NFOLD)
        if a.agg:   # (check_agg takes the bank as it was before the launches)
            moe.check_agg({**ws[0], **runs[1][0]}, [{**w, **r} for w, r in zip(ws, runs[1])], rep, bank0[0])
        for k in outs:
            rep.check(f"launch 1 == launch 0: {k}", all(moe.same_bits([r0[k], r1[k]]) for r0, r1 in zip(runs[0], runs[1])),
                      f"bits, all {n} GPUs")
    else:
        rep.failed = ["checks skipped (MPK_SKIP_CHECKS)"]
    print("ALL CHECKS PASSED" if not rep.failed else f"FAILED: {rep.failed}", flush=True)
    if a.attn_only:   # (the MoE part never ran: its checks fail by design)
        rep.failed = []
    if a.save_outs:
        torch.save([{k: v.cpu() for k, v in r.items()} for r in runs[1]], a.save_outs)
        print("saved launch 1's outputs to", a.save_outs, flush=True)
    if a.e2e_reps and (not rep.failed or os.environ.get("MPK_TIME_ANYWAY") == "1"):   # (MPK_TIME_ANYWAY: a timing bound)
        e2e, spans, dstamps = [], [], []
        for i in range(a.e2e_reps):
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
            e2e.append(max(sk.elapsed_us()))
            spans.append(h.span_us())
            if a.dump_stamps:   # this launch's per-CU stamps, every GPU
                import numpy as np
                lay = h.layout
                dstamps.append(np.stack([h._array(g, "stamps", np.int64).reshape(lay["num_sms"], lay["num_stamps"])
                                         for g in range(n)]))
        if a.dump_stamps:
            import numpy as np
            os.makedirs(a.out, exist_ok=True)
            np.save(os.path.join(a.out, "stamps_timed.npy"), np.stack(dstamps))
            print(f"saved the timed launches' per-CU stamps to {os.path.join(a.out, 'stamps_timed.npy')}", flush=True)
        for g in range(n):
            torch.cuda.synchronize(g)
        last = [{k: w[k].clone() for k in outs} for w in ws]
        cmp = [k for k in outs if not (a.decode and k in ("conv_pool", "state_pool", "hidden", "new_prefix",
                                                                               "moe_out", "routing_pairs", "agg_out", "agg_prefix",
                                                                               "agg_bare"))]
        ok = all(all(moe.same_bits([r0[k], r1[k]]) for r0, r1 in zip(runs[0], last)) for k in cmp)   # (decode KDA: the
        # timed launches ran on the pools as the launches before left them: only the pool-free outputs compare)
        print("CHECK OF THE LAST TIMED LAUNCH " + ("PASSED" if ok else "FAILED"), flush=True)
        e2e.sort()
        spans.sort()
        print(f"LATENCY ({a.e2e_reps} launches, {n} GPUs, {a.attn.upper()} + MoE, {'timing build' if a.profile else 'normal build'}, "
              f"{a.flush_mb} MB flushed before each): e2e median {e2e[len(e2e) // 2]:.1f} us (min {e2e[0]:.1f}, max "
              f"{e2e[-1]:.1f}); span median {spans[len(spans) // 2]:.1f} us", flush=True)


if __name__ == "__main__":
    main()
