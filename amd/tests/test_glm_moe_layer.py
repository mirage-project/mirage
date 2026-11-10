"""GLM-5.3-Flash MoE sublayer on AMD GPUs (TP4, plain decode): x_ffn -> the router (gemm_tile, padded to 384 rows, K parts in
slots) -> glm_topk_route (sigmoid + bias, top 8, renormalised x 2.5, + the shared expert) -> glm_moe_experts (FP8 experts,
this GPU's 512 of each expert's 2048 features) -> allreduce_send (a token's 9 rows added, bf16 to every GPU) -> hc_post_mix
(the FFN's mHC post, the next layer's attention pre mixes) -> hc_pre (x_attn of layer L + 1, its h_post / h_res).
Real weights of MoE layer --layer (router, bias, 288 experts + the shared one, e4m3 + fp32 block scales) and of layer L + 1's
mHC; random x_ffn and streams. Checked per launch: the pairs (experts = the reference's top 8, weights), R per pair (vs
float64 of the dequantized weights and the kernel's own pairs), the all-reduced output, R', x_attn; GPUs equal; relaunches
bit-identical.

    python amd/tests/test_glm_moe_layer.py --tokens 8 [--gpus 4] [--layer 4] [--router-grid 3,4,1] [--codegen]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_glm_mhc import H, HC, bf, cpu_standins, hc_post_ref, hc_pre_ref, load_mhc, relrms  # noqa: E402

E, TOPK, I_FULL, SCALE, NL, NE_PAD = 288, 8, 2048, 2.5, 384, 320


class Ckpt:
    def __init__(self, d):
        self.d = d
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def get(self, name):
        from safetensors import safe_open
        f = self.idx[name]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.d, f), "pt")
        return self.files[f].get_tensor(name)


def load_moe(ck: Ckpt, layer: int):
    p = f"model.language_model.layers.{layer}.mlp."
    w = {"router": ck.get(p + "gate.weight"), "bias": ck.get(p + "gate.e_score_correction_bias").float(), "experts": []}
    for e in range(E + 1):
        q = p + (f"experts.{e}." if e < E else "shared_experts.")
        w["experts"].append({k: (ck.get(q + f"{k}_proj.weight"), ck.get(q + f"{k}_proj.weight_scale_inv").float())
                             for k in ("gate", "up", "down")})
    return w


def pack_experts(w, g: int, n: int):
    """GPU g's bank: W13 [NB 2 IR, H] (unit u: 16 gate rows, then their 16 up rows; each unit's bytes in the load order),
    W13 scales [NB][IR / 128][H / 128][2], W2 [NB H, IR] (each 16-row tile's bytes in the load order), W2 scales [NB][H / 128][IR / 128] (tasks/glm_moe_experts.cuh), and the dequantized fp32 weights of
    each expert's shard for the reference (gate, up [IR, H], down [H, IR])"""
    IR = I_FULL // n
    fs, bs = slice(g * IR, (g + 1) * IR), slice(g * IR // 128, (g + 1) * IR // 128)
    w13, s13, w2, s2, raw = [], [], [], [], []
    for ex in w["experts"]:
        (gq, gs), (uq, us), (dq, ds) = ex["gate"], ex["up"], ex["down"]
        gq, uq, dq = gq.view(torch.uint8)[fs], uq.view(torch.uint8)[fs], dq.view(torch.uint8)[:, fs]
        gs, us, ds = gs[bs], us[bs], ds[:, bs]
        u13 = torch.stack([gq.reshape(IR // 16, 16, H), uq.reshape(IR // 16, 16, H)], 1)   # [unit][i][16 rows][H]
        # (a wave's 16-B load per lane = 1 KB contiguous: [unit][K block][i][half][lane = 16 grp + r16][16 B] holds
        # row 16 i + r16, K block 128 kb + 32 grp + 16 half ..)
        w13.append(u13.reshape(IR // 16, 2, 16, H // 128, 4, 2, 16).permute(0, 3, 1, 5, 4, 2, 6).reshape(2 * IR, H))
        s13.append(torch.stack([gs, us], -1))
        # W2: [16-row tile][K block][half][lane][16 B]
        w2.append(dq.reshape(H // 16, 16, IR // 128, 4, 2, 16).permute(0, 2, 4, 3, 1, 5).reshape(H, IR))
        s2.append(ds.contiguous())
        raw.append(((gq, gs), (uq, us), (dq, ds)))
    return (torch.cat(w13, 0).contiguous(), torch.stack(s13).reshape(-1).contiguous(), torch.cat(w2, 0).contiguous(),
            torch.stack(s2).reshape(-1).contiguous(), raw)


_DEQ = {}


def dequant(raw, g, e):
    """expert e's shard on GPU g, dequantized to fp64 (gate, up [IR, H], down [H, IR]), cached"""
    if (g, e) not in _DEQ:
        f8 = lambda q: q.view(torch.float8_e4m3fn).double()
        blk = lambda q, s: f8(q) * s.double().repeat_interleave(128, 0).repeat_interleave(128, 1)
        _DEQ[(g, e)] = tuple(blk(q, s) for q, s in raw[g][e])
    return _DEQ[(g, e)]


def route_ref(x, router, bias):
    """SGLang's noaux_tc for GLM-5.3-Flash: sigmoid scores, + bias for the choice, top 8, the scores renormalised x 2.5"""
    logits = x.float() @ router.float().t()
    sc = torch.sigmoid(logits)
    idx = torch.topk(sc + bias, TOPK, dim=-1).indices
    wt = sc.gather(1, idx)
    return idx, wt / wt.sum(-1, keepdim=True) * SCALE


def node_times(h, paths, nodes):
    """Timing build, after a launch: per GPU and node its tasks, median / longest task us, first start .. last end us after
    the start barrier; glm_moe_experts: its stage stamps 0 (the table built) and 1 (its poller saw every W13 unit done)."""
    import numpy as np
    names = {q.graph_idx: q.name for q in nodes}
    times, starts = h.task_times(), [b for b, _ in h.timing()]
    for g, path in enumerate(paths[: h.sk.num_gpus]):
        sched = json.load(open(path))
        per, st, rel = {}, {0: [], 1: []}, {}
        for sm, queue in enumerate(sched["worker_task_queues"]):
            for i, tid in enumerate(queue):
                s0, e0, a0, a1 = times[g][sm][i]
                if s0 and e0:
                    node = sched["all_tasks"][tid]["node"]
                    per.setdefault(node, []).append((s0, e0))
                    for k, v in ((0, a0), (1, a1)):   # stage stamps inside the task: us after its start
                        if v and s0 <= v <= e0:
                            rel.setdefault((node, k), []).append((int(v) - int(s0)) / 1e3)
                    if names[node] == "glm_topk_route" and a0:   # (stage stamp 0 packed: keys / picked / end - landed, 10 ns)
                        rel.setdefault((node, "tk"), []).append(((a0 & 0xFFFF) / 100, ((a0 >> 16) & 0xFFFF) / 100,
                                                                 ((a0 >> 32) & 0xFFFF) / 100, (int(e0) - int(s0)) / 1e3))
                    if names[node] == "glm_moe_experts":
                        for k, v in ((0, a0), (1, a1)):
                            if v:
                                st[k].append((int(v) - int(starts[g])) / 1e3)
        print(f"GPU {g}: node: tasks, median / longest task us, first start .. last end us after the start barrier", flush=True)
        for node in sorted(per):
            d = sorted((e - s) / 1e3 for s, e in per[node])
            fs = (min(s for s, _ in per[node]) - int(starts[g])) / 1e3
            le = (max(e for _, e in per[node]) - int(starts[g])) / 1e3
            sst = "".join(f", stage {k} at +{sorted(rel[(node, k)])[len(rel[(node, k)]) // 2]:.1f}" for k in (0, 1)
                          if (node, k) in rel)
            print(f"  {names[node]}#{node}: {len(d)}, {d[len(d) // 2]:.1f} / {d[-1]:.1f}, {fs:.1f} .. {le:.1f}{sst}", flush=True)
            if (node, "tk") in rel:
                tk = sorted(rel[(node, "tk")], key=lambda v: v[2])[len(rel[(node, "tk")]) // 2]
                print(f"    topk after its K parts landed (median task): keys +{tk[0]:.2f}, picked +{tk[1]:.2f}, end +{tk[2]:.2f} us "
                      f"(task {tk[3]:.1f} us: landed at +{tk[3] - tk[2]:.1f})", flush=True)
        dn = os.environ.get("MPK_TASK_DUMP")   # (a node index: its 12 latest-ending tasks - CU, start, end, stage stamps, the
        if dn:                                  # task before it on that CU: node, end; us after the start barrier)
            dn = int(dn)
            rows = []
            for sm, queue in enumerate(sched["worker_task_queues"]):
                prev = None
                for i, tid in enumerate(queue):
                    s0, e0, a0, a1 = times[g][sm][i]
                    node = sched["all_tasks"][tid]["node"]
                    if node == dn and s0 and e0:
                        rows.append((int(e0), sm, int(s0), a0, a1, prev))
                    if s0 and e0:
                        prev = (node, int(e0))
            rel0 = lambda v: (int(v) - int(starts[g])) / 1e3
            print(f"  latest-ending tasks of node {dn} (CU, start .. end, stage 0 / 1, before it on the CU: node end):", flush=True)
            for e0, sm, s0, a0, a1, prev in sorted(rows, reverse=True)[:12]:
                ps = f"{names[prev[0]]}#{prev[0]} {rel0(prev[1]):.1f}" if prev else "-"
                print(f"    CU {sm}: {rel0(s0):.1f} .. {rel0(e0):.1f}, stages {rel0(a0) if a0 else 0:.1f} / {rel0(a1) if a1 else 0:.1f};"
                      f" before: {ps}", flush=True)
        for k in (0, 1):
            if st[k]:
                v = np.array(st[k])
                print(f"  glm_moe_experts stage stamp {k}: {len(v)} CUs, min / median / max {v.min():.1f} / "
                      f"{np.median(v):.1f} / {v.max():.1f} us", flush=True)
        if g == 0 and h.sk.num_gpus > 1 and os.environ.get("MPK_ALL_GPUS") != "1":   # (MPK_ALL_GPUS=1: every GPU's lines)
            break


def pair_send_postmix(plan, snd, pm):
    """STATIC_ARS_LDSHAND's placement: hc_post_mix (node pm) task x right after allreduce_send (node snd) task x on the same CU
    (H / 32 send slices: one send task per hc_post_mix task)"""
    new_lists = []
    for lists in plan["lists"]:
        mine = {tuple(t[1])[0]: t for cu in lists for t in cu if t[0] == pm}
        rest = [[t for t in cu if t[0] != pm] for cu in lists]
        for cu in rest:
            for j in [j for j, t in enumerate(cu) if t[0] == snd][::-1]:
                cu.insert(j + 1, mine.pop(tuple(cu[j][1])[0]))
        assert not mine, "STATIC_ARS_LDSHAND placement: one send task per hc_post_mix task (H / 32 send slices)"
        new_lists.append(rest)
    return {**plan, "lists": new_lists}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--model-dir", default="/dev/shm/glm53f")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--router-grid", default="12,8,1")   # (MI355X 11-07 glmmhc3, T 1: 3x4 router 11.6 us, table at 17.0; 12x8 5.1 us, table at 10.6)
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--out", default="glm_moe")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--x-from", default="", help="x_ffn from a server dump (sglang_glm MPK_GLM_DUMP: r0_t<T>_l<layer>_B.pt): the real "
                   "routing of a decode step (concentrated: fewer experts, more tokens each than random x)")
    p.add_argument("--serving", action="store_true", help="with --codegen: the server library's graph (scratch tensors inside)")
    p.add_argument("--codegen", action="store_true")
    p.add_argument("--profile", action="store_true", help="the timing build: after the timed launches, per node its task times")
    p.add_argument("--random", action="store_true", help="random router / experts in the real shapes (no checkpoint read)")
    p.add_argument("--stop", default="", choices=("", "topk", "experts"), help="the graph up to this node only (bisecting)")
    a = p.parse_args()
    T, n = a.tokens, a.gpus
    KP, IR = TOPK + 1, I_FULL // n
    os.environ["MPK_TOKENS"] = str(T)
    if a.codegen:
        cpu_standins()
    from mirage.core import bfloat16, float32, int32, int64  # noqa: F401
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import HC_MIX_STRIDE, StaticMegakernel
    from plan_one_gpu import simple_plan
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    rms_eps, hc_eps, iters, limit = cfg["rms_norm_eps"], cfg["hc_eps"], cfg["hc_sinkhorn_iters"], cfg["swiglu_limit"]
    ck = Ckpt(a.model_dir)
    if a.random:
        rg = torch.Generator().manual_seed(23)
        f8r = lambda *sh: (torch.randn(*sh, generator=rg) * 2).clamp(-448, 448).to(torch.float8_e4m3fn)
        scl = lambda *sh: torch.rand(*sh, generator=rg) * 0.02 + 0.005
        exs = [{"gate": (f8r(I_FULL, H), scl(16, 32)), "up": (f8r(I_FULL, H), scl(16, 32)), "down": (f8r(H, I_FULL), scl(32, 16))}
               for _ in range(E + 1)]
        W = {"router": (torch.randn(E, H, generator=rg) * H ** -0.5).to(torch.bfloat16), "bias": torch.randn(E, generator=rg) * 0.1,
             "experts": exs}
    elif a.codegen:   # (the shapes only)
        z = lambda *sh: torch.zeros(*sh, dtype=torch.uint8).view(torch.float8_e4m3fn)
        ex = {"gate": (z(I_FULL, H), torch.ones(16, 32)), "up": (z(I_FULL, H), torch.ones(16, 32)),
              "down": (z(H, I_FULL), torch.ones(32, 16))}
        W = {"router": torch.zeros(E, H, dtype=torch.bfloat16), "bias": torch.zeros(E), "experts": [ex] * (E + 1)}
    else:
        W = load_moe(ck, a.layer)
    m = load_mhc(a.model_dir, a.layer + 1, "attn")
    gen = torch.Generator().manual_seed(17)
    x = bf(torch.randn(T, H, generator=gen))
    if a.x_from:
        x = torch.load(a.x_from)["x_ffn"].to(torch.bfloat16).cpu()
        assert tuple(x.shape) == (T, H), f"--x-from: {tuple(x.shape)} vs ({T}, {H})"
    R = bf(torch.randn(T, HC, H, generator=gen))
    hp = (torch.rand(T, HC, generator=gen) * 2).float()
    hr = torch.rand(T, HC * HC, generator=gen).float()
    hc_small = torch.zeros(32, dtype=torch.float32)
    hc_small[:HC * (HC + 2)], hc_small[HC * (HC + 2):HC * (HC + 2) + 3] = m["base"], m["scale"]
    router = torch.zeros(NL, H, dtype=torch.bfloat16)
    router[:E] = W["router"]
    bias = torch.full((NE_PAD,), float("-inf"))
    bias[:E] = W["bias"]
    ws, deqs = [], []
    for g in range(n):
        dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
        w13, s13, w2, s2, raw = pack_experts(W, g, n)
        deqs.append(raw)
        ws.append({"x_ffn": x.to(dev), "router": router.to(dev), "bias": bias.to(dev), "w13": w13.to(dev),
                   "w13_sf": s13.to(dev), "w2": w2.to(dev), "w2_sf": s2.to(dev),
                   "pairs": torch.zeros(T, KP, dtype=torch.int64, device=dev),
                   "R_moe": torch.zeros(T * KP, H, dtype=torch.float32, device=dev),
                   "R_in": R.reshape(T, HC * H).to(dev), "h_post_in": hp.to(dev), "h_res_in": hr.to(dev),
                   "hc_attn_fn": m["fn"].to(dev), "hc_attn_small": hc_small.to(dev), "attn_norm": m["gamma"].to(dev),
                   "R_out": torch.zeros(T, HC * H, dtype=torch.bfloat16, device=dev),
                   "x_attn": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
                   "h_post_out": torch.zeros(T, HC, dtype=torch.float32, device=dev),
                   "h_res_out": torch.zeros(T, HC * HC, dtype=torch.float32, device=dev)})
        if any("STATIC_GME_DBG" in f for f in a.flag):
            ws[-1]["gme_dbg"] = torch.zeros(2048, dtype=torch.int32).pin_memory()
        if any("STATIC_GME_TIMES" in f for f in a.flag):   # per wave 8 times (long long), on the GPU
            ws[-1]["gme_dbg"] = torch.zeros(1024 * 8 * 2, dtype=torch.int32, device=dev)
    if not a.codegen:
        torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    # --serving (the server libraries): the scratch tensors inside the kernel (node buffers the kernel resets every launch;
    # bound as run tensors they would keep the last launch's values: a polled reader could read them before the new ones)
    SCRATCH = ['pairs', 'R_moe'] if a.serving else []
    _dt = {torch.bfloat16: bfloat16, torch.float32: float32, torch.int64: int64, torch.int32: int32}
    at = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ws[0] if k not in SCRATCH}
    for k in SCRATCH:
        at[k] = mpk.new_tensor(dims=tuple(ws[0][k].shape), dtype=_dt[ws[0][k].dtype], name=k)
    logits = mpk.new_tensor(dims=(T, NL), dtype=float32, name="router_logits")
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="moe_sent")
    mix = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name="attn_hc_mix")
    mpk.gemm_tile_layer(input=at["x_ffn"], weight=at["router"], output=logits, combine="slots")
    mpk.glm_topk_route_layer(input=logits, bias=at["bias"], output=at["pairs"], k=TOPK, shared_experts=1, shared_first=E,
                             routed_scaling_factor=SCALE)
    if a.stop != "topk":
      mpk.glm_moe_experts_layer(x=at["x_ffn"], pairs=at["pairs"], w13=at["w13"], w13_sf=at["w13_sf"], w2=at["w2"],
                              w2_sf=at["w2_sf"], output=at["R_moe"], swiglu_limit=limit, dbg=at.get("gme_dbg"))
    # MPK_ARS_HAND (default 1, T >= MPK_ARS_HAND_MIN_T = the kernel's STATIC_ARS_LDSHAND_MIN_T): H / 32 send slices and
    # hc_post_mix task x right after send task x on the same CU (the kernel's STATIC_ARS_LDSHAND hand-off in LDS)
    hand = os.environ.get("MPK_ARS_HAND", "1") == "1" and T >= int(os.environ.get("MPK_ARS_HAND_MIN_T", "4"))
    if not a.stop:
      mpk.allreduce_send_layer(input=at["R_moe"], output=sent,
                               num_tasks=int(os.environ.get("MPK_ARS_NSLICE", "0")) or (H // 32 if hand else None),
                               prefetch=(at["R_in"], at["hc_attn_fn"]) if hand and os.environ.get("MPK_ARS_PREFETCH") == "1" else None)   # (STATIC_ARS_COLMAJ: C8 % slices == 0)
      mpk.hc_post_mix_layer(sent=sent, residual=at["R_in"], coef=(at["h_post_in"], at["h_res_in"]), fn=at["hc_attn_fn"],
                          output=at["R_out"], mix=mix)
      mpk.hc_pre_layer(mix=mix, residual=at["R_out"], hc_small=at["hc_attn_small"], gamma=at["attn_norm"],
                     output=at["x_attn"], rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters,
                     export=(at["h_post_out"], at["h_res_out"]))
    nodes = compiler.graph_nodes(mpk)
    plan = simple_plan(mpk, {nodes[0].graph_idx: tuple(int(v) for v in a.router_grid.split(","))})
    ncu_e = int(os.environ.get("MPK_GME_NCU", "0"))
    if ncu_e:   # (the sends on the CUs without an experts task, round robin, each after the list's tasks of earlier nodes)
        snd = next(q.graph_idx for q in nodes if q.name == "allreduce_send")
        new_lists = []
        for lists in plan["lists"]:
            mine = [t for cu in lists for t in cu if t[0] == snd]
            rest = [[t for t in cu if t[0] != snd] for cu in lists]
            gme = next(q.graph_idx for q in nodes if q.name == "glm_moe_experts")
            free = [i for i, cu in enumerate(rest) if not any(t[0] == gme for t in cu)]   # (round robin: NCU distinct CUs)
            assert len(free) == len(rest) - ncu_e and all(sum(t[0] == gme for t in cu) <= 1 for cu in rest)
            for i, task in enumerate(mine):
                cu = rest[free[i % len(free)]]
                at_ = max((j + 1 for j, t in enumerate(cu) if t[0] <= task[0]), default=0)
                cu.insert(at_, task)
            new_lists.append(rest)
        plan = {**plan, "lists": new_lists}
    if os.environ.get("MPK_HCPM_FREE") == "1":   # (hc_post_mix on the CUs without a send task, round robin, in node order)
        snd = next(q.graph_idx for q in nodes if q.name == "allreduce_send")
        pm = next(q.graph_idx for q in nodes if q.name == "hc_post_mix")
        new_lists = []
        for lists in plan["lists"]:
            mine = [t for cu in lists for t in cu if t[0] == pm]
            rest = [[t for t in cu if t[0] != pm] for cu in lists]
            free = [i for i, cu in enumerate(rest) if not any(t[0] == snd for t in cu)] or list(range(len(rest)))
            for i, task in enumerate(mine):
                cu = rest[free[i % len(free)]]
                at_ = max((j + 1 for j, t in enumerate(cu) if t[0] <= task[0]), default=0)
                cu.insert(at_, task)
            new_lists.append(rest)
        plan = {**plan, "lists": new_lists}
    if os.environ.get("MPK_ARS_PAIRCU", "1" if hand and not a.stop else "0") == "1":   # (STATIC_ARS_LDSHAND: hc_post_mix task x right after send task x, same CU)
        snd = next(q.graph_idx for q in nodes if q.name == "allreduce_send")
        pm = next(q.graph_idx for q in nodes if q.name == "hc_post_mix")
        plan = pair_send_postmix(plan, snd, pm)
    print("nodes:", [(q.graph_idx, q.name, q.grid) for q in compiler.graph_nodes(mpk)], flush=True)
    os.makedirs(a.out, exist_ok=True)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
    if a.codegen:
        import subprocess
        from mirage.mpk import static_schedule
        code = static_schedule.generate_code(mpk, [json.load(open(q)) for q in paths], False)
        cu = os.path.join(a.out, "layer.cu")
        open(cu, "w").write(code)
        r = subprocess.run(static_schedule.compile_command(cu, os.path.join(a.out, "layer.so"), a.flag + ["--save-temps"],
                                                           arch="gfx950"), cwd=a.out, capture_output=True, text=True)
        print(f"hipcc exit {r.returncode}", r.stderr[-3000:] if r.returncode else "", flush=True)
        return
    from mirage.mpk.static_harness import Harness
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag,
                            profile=a.profile)
    h = Harness(sk)
    idx_ref, wt_ref = route_ref(x, W["router"], W["bias"])
    failed = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            failed.append(name)

    def expert_ref(xt, g, e, lim):
        gt, up, dn = dequant(deqs, g, e)
        gv, uv = xt.double() @ gt.t(), xt.double() @ up.t()
        gv, uv = gv.clamp(max=lim), uv.clamp(-lim, lim)
        return (gv * torch.sigmoid(gv) * uv) @ dn.t()
    first = None
    for launch in range(a.launches):
        try:
            h.launch(512 << 20, timeout_s=60)
        except TimeoutError:
            if os.environ.get("GDB_STOP") == "1":   # (run under rocgdb: stop here, the waves still where they hang)
                import signal
                import time
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(30)
            if "gme_dbg" in ws[0]:
                for g in range(n):
                    d = ws[g]["gme_dbg"][:256].tolist()
                    hist = {}
                    for v in d:
                        k = (f"unit phase {v % 10}" if v >= 100000 else v if v < 4 or v == 1000 else "queue")
                        hist[k] = hist.get(k, 0) + 1
                    print(f"GPU {g}: CUs per phase {hist}; max item {max(d)}", flush=True)
                    wv = ws[g]["gme_dbg"][256:256 + 1024].view(256, 4)
                    from collections import Counter
                    print(f"GPU {g}: (wave 0..3 phases) of the CUs: {Counter(tuple(int(v) % 10 for v in r) for r in wv).most_common(6)}",
                          flush=True)
            raise
        for g in range(n):
            torch.cuda.synchronize(g)
        pk = ws[0]["pairs"].cpu()
        ek, wk = (pk >> 32).int(), (pk & 0xFFFFFFFF).int().view(torch.float32)
        same_set = all(set(ek[t, :TOPK].tolist()) == set(idx_ref[t].tolist()) for t in range(T))
        if launch == 0:
            from collections import Counter
            cnt = Counter(ek[:, :TOPK].flatten().tolist())
            print(f"routing: {len(cnt)} routed experts for {T} tokens, tokens per expert max {max(cnt.values())}, "
                  f"median {sorted(cnt.values())[len(cnt) // 2]}", flush=True)
            check("pairs: the routed experts = the reference's top 8", same_set,
                  f"tokens differing: {[t for t in range(T) if set(ek[t, :TOPK].tolist()) != set(idx_ref[t].tolist())]}")
            wr = torch.stack([wt_ref[t][[idx_ref[t].tolist().index(e) for e in ek[t, :TOPK].tolist()]] if same_set else
                              wt_ref[t] for t in range(T)])
            check("pairs: weights", relrms(wk[:, :TOPK], wr) < 1e-5 and bool((ek[:, TOPK] == E).all()) and
                  bool((wk[:, TOPK] == 1.0).all()), f"relRMS {relrms(wk[:, :TOPK], wr):.2e}, shared pair ok "
                  f"{bool((ek[:, TOPK] == E).all())}")
            # R per pair on GPU 0 vs float64 of the dequantized shard (the kernel's own pairs)
            Rk = ws[0]["R_moe"].cpu().view(T, KP, H)
            ref = torch.zeros(T, KP, H, dtype=torch.float64)
            for t in range(T):
                for j in range(KP):
                    ref[t, j] = expert_ref(x[t:t + 1], 0, int(ek[t, j]), limit)[0] * float(wk[t, j])
            if a.stop != "topk" and relrms(Rk, ref) >= 1e-2:   # (which pairs: per expert, token, position)
                er = ((Rk.double() - ref).norm(dim=2) / ref.norm(dim=2).clamp_min(1e-30))
                bad = (er > 1e-2).nonzero().tolist()
                from collections import Counter
                print(f"  bad pairs {len(bad)} of {T * KP}: per expert {sorted(Counter(int(ek[t, j]) for t, j in bad).items())[:12]}; "
                      f"pair slots j {sorted(Counter(j for t, j in bad).items())}; tokens {sorted(set(t for t, j in bad))[:40]}; "
                      f"rows wrong (first bad pair) {int(((Rk[bad[0][0], bad[0][1]].double() - ref[bad[0][0], bad[0][1]]).abs() > 1e-2 * ref[bad[0][0], bad[0][1]].abs().max()).sum()) if bad else 0}", flush=True)
            if a.stop != "topk":
                check("GPU 0: R per pair vs float64 (dequantized weights)", relrms(Rk, ref) < 1e-2,
                      f"relRMS {relrms(Rk, ref):.2e}")
        if a.stop:
            continue
        G0 = {k: ws[0][k].cpu().clone() for k in ("R_out", "x_attn", "h_post_out", "h_res_out", "pairs")}
        if launch == 0:   # the chain: every GPU's shard, the all-reduce, mHC
            acc = torch.zeros(T, H)
            for g in range(n):
                part = torch.zeros(T, H, dtype=torch.float64)
                for t in range(T):
                    for j in range(KP):
                        part[t] += expert_ref(x[t:t + 1], g, int(ek[t, j]), limit)[0] * float(wk[t, j])
                acc = acc + bf(part.float()).float()
            out = bf(acc)
            R2 = hc_post_ref(out, R, hp, hr)
            x_ref, _, _ = hc_pre_ref(R2, m["fn"], m["base"], m["scale"], m["gamma"], rms_eps, hc_eps, rms_eps, iters)
        R2k = G0["R_out"].view(T, HC, H)
        check(f"launch {launch}: R' vs the reference chain", relrms(R2k, R2) < 2e-2, f"relRMS {relrms(R2k, R2):.2e}")
        check(f"launch {launch}: x_attn vs the reference chain", relrms(G0["x_attn"], x_ref) < 3e-2,
              f"relRMS {relrms(G0['x_attn'], x_ref):.2e}")
        same = all(torch.equal(ws[g][k].cpu(), G0[k]) for g in range(1, n) for k in G0)
        check(f"launch {launch}: every GPU's outputs equal", same)
        if first is None:
            first = G0
        else:
            check(f"launch {launch}: bit-identical to launch 0", all(torch.equal(G0[k], first[k]) for k in G0))
    if (not failed or os.environ.get("MPK_TIME_ANYWAY") == "1") and a.reps:   # (MPK_TIME_ANYWAY: a timing probe, results wrong)
        us = []
        for _ in range(a.reps):
            h.launch(512 << 20, timeout_s=60)
            us.append(max(h.sk.elapsed_us()))
        us.sort()
        print(f"LATENCY e2e median {us[len(us) // 2]:.1f} us (min {us[0]:.1f}, max {us[-1]:.1f}; {a.reps} launches, L2 "
              f"emptied, slowest GPU)", flush=True)
        if a.profile:
            node_times(h, paths, compiler.graph_nodes(mpk))
        if any("STATIC_GME_TIMES" in f for f in a.flag):   # GPU 0's waves, us after its start barrier (the last launch)
            import numpy as np
            st = int([b for b, _ in h.timing()][0])
            tm = ws[0]["gme_dbg"].cpu().view(torch.int64).view(1024, 8).numpy().astype(np.float64)
            np.save(os.path.join(a.out, "gme_times.npy"), tm - st)   # (ns after GPU 0's start barrier, per wave gw)
            ok = tm[:, 1] > 0
            rel = (tm - st) / 1e3
            q = lambda v: f"{np.percentile(v, 5):.1f} / {np.median(v):.1f} / {v.max():.1f}"
            print(f"GME_TIMES GPU 0 ({ok.sum()} waves with a W13 item; 5 % / median / max us): table built {q(rel[:, 0])}; "
                  f"first item issued {q(rel[ok, 1])}; its sums done {q(rel[ok, 2])}; W13 end counted {q(rel[:, 3])}; "
                  f"item compute+wait {q(rel[ok, 2] - rel[ok, 1])}; first generic-loop item entered {q(rel[tm[:, 7] > 0, 7])}", flush=True)
            w2 = tm[:, 6] > 0
            if w2.any():
                print(f"GME_TIMES GPU 0 W2 ({w2.sum()} waves, items per wave max {int(tm[:, 6].max())}): first W2 item past its "
                      f"wait {q(rel[w2, 4])}; last item done {q(rel[w2, 5])}; W2 time per item "
                      f"{q((rel[w2, 5] - rel[w2, 4]) / tm[w2, 6])}", flush=True)
    if any("STATIC_PRE_STAMPS" in f for f in a.flag) and a.reps:   # (normal build: per CU entry / start / stored / end, ns)
        import numpy as np
        lay = h.layout
        for g in range(n):
            st_ = h._array(g, "stamps", np.int64).reshape(lay["num_sms"], lay["num_stamps"]).astype(np.float64)
            ent, sta, sto, end = st_[:, 4], st_[:, 0], st_[:, 6], st_[:, 3]
            ok = (ent > 0) & (end > 0)
            e0 = ent[ok].min()
            q = lambda v: f"{np.median(v):.1f} / {v.max():.1f}"
            print(f"PRE GPU {g} (us after the first CU's entry, median / max over CUs): entry {q((ent[ok] - e0) / 1e3)}; start "
                  f"{q((sta[ok] - e0) / 1e3)}; resets stored {q((sto[ok] - e0) / 1e3)}; end {q((end[ok] - e0) / 1e3)}", flush=True)
    print("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}", flush=True)


if __name__ == "__main__":
    main()
