"""The K3 MLA layer (TP8, 8 tokens, 12 of the 96 heads per GPU) on AMD GPUs as one static megakernel, with random weights in the
real shapes and a random paged KV cache, checked step by step against float64 references (mpk-kernels tests/mla_reference.py),
then timed like amd/tests/test_layer_one_gpu.py.

Graph per GPU (amd/notes/kda_mi355x.md, MLA section):
  x -> fused q_a / kv_a / k_pe projection (gemm_tile, [2112 -> 2176, 7168] bf16, K parts in slots)
    -> mla_norm (per token: q_a = rmsnorm; the KV row [rmsnorm(kv_a) | k_pe] into the paged cache at its row)
    -> q_b (gemm_tile "local", polled: q [T, 12 x 192] bf16)
    -> mla_attn (per head: q_abs = q_nope . w_kc (w_kc in LDS), attention over the request's KV rows, online softmax)
  x -> g_proj (gemm_tile, [1536, 7168], K parts in slots)
    -> mla_vgate (per head: v = attn . w_vc (w_vc in LDS), o_in = bf16(v bf16(sigmoid(gate))))
    -> o_proj (gemm_tile push_all, polled) -> attn_residual (KDA's)

python amd/tests/test_mla_layer.py --gpus 8 [--kv-rows 37] [--gemm-shuffled] [--flush-mb 1024] [--e2e-reps 40] [--flag ...]
MK_BACKEND=hip python amd/tests/test_mla_layer.py --codegen ...   (no GPU: code + hipcc only)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "amd", "tools"))
from test_layer_one_gpu import Report, bf16, node_times, relrms, same_bits  # noqa: E402

T, H = 8, 7168
HEADS, NOPE, RP, KV, VD, QL = 12, 128, 64, 512, 128, 1536
NA, NA_PAD = QL + KV + RP, 2176
NQ, NG, D = HEADS * (NOPE + RP), HEADS * VD, KV + RP
POOL_ROWS, REQS, REQ, MAX_CTX = 10240, 3, 1, 9216   # SGLang's KV pool rows, req_to_token [REQS, MAX_CTX] (page size 1)
SCALE = (NOPE + RP) ** -0.5
EPS = 1e-5
NB_ALLOC, NB = 4, 3
SHARDED = ("wqb", "wkc", "wvc", "wg", "wo")


def random_weights(dev, seed: int, kv_rows: int, common: dict = None) -> dict:
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *shape: torch.randn(shape, generator=g, device=dev)
    b = lambda t: t.to(torch.bfloat16)
    perm = torch.randperm(POOL_ROWS, generator=g, device=dev)[:kv_rows + T].to(torch.int32)   # the request's rows, random order
    table = torch.full((REQS, MAX_CTX), -1, dtype=torch.int32, device=dev)
    table[REQ, :kv_rows + T] = perm
    w = {
        "x": b(rn(T, H)),
        "wa": b(torch.cat([rn(NA, H) * H ** -0.5, torch.zeros(NA_PAD - NA, H, device=dev)])),
        "wq_norm": b(1 + 0.1 * rn(QL)), "wkv_norm": b(1 + 0.1 * rn(KV)),
        "wqb": b(rn(NQ, QL) * QL ** -0.5),
        "wkc": b(rn(HEADS * NOPE, KV) * NOPE ** -0.5),
        "wvc": b(rn(HEADS * KV, VD) * KV ** -0.5),
        "wg": b(rn(NG, H) * H ** -0.5),
        "wo": b(rn(H, NG) * NG ** -0.5),
        "cache": b(rn(POOL_ROWS, D)),
        "req_to_token": table,
        "seq_lens": torch.tensor([kv_rows, -1], dtype=torch.int64, device=dev),    # (SGLang: the rows before the step)
        "req_idx": torch.tensor([REQ, -1], dtype=torch.int64, device=dev),
        "out_loc": perm[kv_rows:kv_rows + T].long(),
        "bank": b(rn(T, NB_ALLOC, H)), "prefix": b(rn(T, H)), "q": b(rn(H) * H ** -0.5), "gain": b(1 + 0.1 * rn(H)),
        "ow": b(1 + 0.1 * rn(H)),
        "hidden": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
        "new_prefix": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
    }
    w["info"] = torch.tensor([NB, NB_ALLOC * H, H, MAX_CTX, 0, 0, 0, 0], dtype=torch.int32, device=dev)
    if common is not None:
        for k in w:
            if k not in SHARDED and k not in ("hidden", "new_prefix"):
                w[k] = common[k].to(dev).clone() if k == "cache" else common[k].to(dev)
    return w


def rmsnorm(x, wt, eps=EPS):
    x = x.double()
    return bf16(x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * wt.double()).double()


def ref_front(w):
    latent = bf16(w["x"].double() @ w["wa"][:NA].double().T).double()
    q_a = rmsnorm(latent[:, :QL], w["wq_norm"])
    kv_new = torch.cat([rmsnorm(latent[:, QL:QL + KV], w["wkv_norm"]), latent[:, QL + KV:]], -1)
    gate = bf16(w["x"].double() @ w["wg"].double().T).double()
    return q_a, kv_new, gate


def ref_attn(q, w, cache, kv_rows):
    """attn [T, 12 x 512] from the kernel's q [T, NQ] and the cache (with this step's rows)."""
    qh = q.double().view(T, HEADS, NOPE + RP)
    wkc = w["wkc"].double().view(HEADS, NOPE, KV)
    q_abs = bf16(torch.einsum("thn,hnl->thl", qh[..., :NOPE], wkc)).double()
    S = kv_rows + T
    rows = w["req_to_token"][REQ, :S].long()
    K = cache[rows].double()
    s = (torch.einsum("thl,jl->thj", q_abs, K[:, :KV]) + torch.einsum("thr,jr->thj", qh[..., NOPE:], K[:, KV:])) * SCALE
    j = torch.arange(S, device=cache.device).view(1, 1, -1)
    t = torch.arange(T, device=cache.device).view(-1, 1, 1)
    s = s.masked_fill(j > kv_rows + t, float("-inf"))
    e = torch.exp(s - s.amax(-1, keepdim=True))
    return bf16(torch.einsum("thj,jl->thl", bf16(e).double(), K[:, :KV]) / e.sum(-1, keepdim=True)).double().view(T, HEADS * KV)


def ref_residual(a, w):
    u = bf16(w["prefix"].double() + bf16(a).double()).double()   # (a: the all-reduce's bf16 output, SGLang ROCm)
    src = torch.cat([w["bank"][:, :NB].double(), u.unsqueeze(1)], 1)
    c = (w["q"].float() * w["gain"].float()).double()   # fp32, not rounded (SGLang ROCm cw2)
    sc = (src * c).sum(-1) / torch.sqrt((src * src).mean(-1) + EPS)
    h = (torch.softmax(sc, -1).unsqueeze(-1) * src).sum(1)
    return u, bf16(h / torch.sqrt((h * h).mean(-1, keepdim=True) + EPS) * w["ow"].double())


def read_buf(h, g, name, dtype, shape):
    a = np.frombuffer(h.sk.read(g, name), dtype=np.uint8).copy()
    n = int(np.prod(shape)) * torch.tensor([], dtype=dtype).element_size()
    return torch.from_numpy(a[:n]).view(dtype).view(*shape)


def check_all(ws, h, rep, cache0, kv_rows):
    n = len(ws)
    partials = []
    for gi, w in enumerate(ws):
        tag = f"GPU {gi}: "
        dev = w["x"].device
        q_a_ref, kv_ref, gate_ref = ref_front(w)
        q_a = read_buf(h, gi, "q_a", torch.bfloat16, (T, QL)).to(dev)
        rep.check(tag + "q_a", relrms(q_a, q_a_ref) < 2e-3, f"relRMS {relrms(q_a, q_a_ref):.2e}")
        rows = w["out_loc"]
        rep.check(tag + "KV rows written", relrms(w["cache"][rows], kv_ref) < 2e-3, f"relRMS {relrms(w['cache'][rows], kv_ref):.2e}")
        others = torch.ones(w["cache"].shape[0], dtype=torch.bool, device=dev)
        others[rows] = False
        rep.check(tag + "cache else unchanged", torch.equal(w["cache"][others].cpu(), cache0[others.cpu()]), "bits")
        q = read_buf(h, gi, "q_heads", torch.bfloat16, (T, NQ)).to(dev)
        q_ref = bf16(q_a.double() @ w["wqb"].double().T).double()
        rep.check(tag + "q (q_b)", relrms(q, q_ref) < 1e-3, f"relRMS {relrms(q, q_ref):.2e} (from the kernel's q_a)")
        attn = read_buf(h, gi, "attn", torch.bfloat16, (T, HEADS * KV)).to(dev)
        attn_ref = ref_attn(q, w, w["cache"], kv_rows)
        rep.check(tag + "attn", relrms(attn, attn_ref) < 5e-3, f"relRMS {relrms(attn, attn_ref):.2e} (from the kernel's q and cache)")
        o_in = read_buf(h, gi, "o_in", torch.bfloat16, (T, NG)).to(dev)
        v = bf16(torch.einsum("thl,hlv->thv", attn.double().view(T, HEADS, KV), w["wvc"].double().view(HEADS, KV, VD))).double()
        o_ref = bf16(v.reshape(T, NG) * bf16(torch.sigmoid(gate_ref)).double()).double()
        rep.check(tag + "o_in (w_vc, gate)", relrms(o_in, o_ref) < 5e-3, f"relRMS {relrms(o_in, o_ref):.2e} (from the kernel's attn)")
        partials.append(bf16(o_in.double() @ w["wo"].double().T).double())
    a = sum(p.to(ws[0]["x"].device) for p in partials)
    u_ref, hid_ref = ref_residual(a, ws[0])
    rep.check("u (new prefix)", relrms(ws[0]["new_prefix"], u_ref) < 2e-3, f"relRMS {relrms(ws[0]['new_prefix'], u_ref):.2e}")
    rep.check("hidden", relrms(ws[0]["hidden"], hid_ref) < 5e-3, f"relRMS {relrms(ws[0]['hidden'], hid_ref):.2e}")
    if n > 1:
        rep.check("hidden, u the same on every GPU", same_bits([w["hidden"] for w in ws]) and
                  same_bits([w["new_prefix"] for w in ws]), "bits")


def prep_weights(ws: list, gemm_shuffled: bool) -> None:
    """the kernel's weight forms: the GEMM weights (shuffled layout or as given), w_kc / w_vc transposed per head (SGLang's
    memory layout)"""
    from mirage.mpk.static_megakernel import shuffle_gemm_weight
    for w in ws:
        for name in ("wa", "wqb", "wg", "wo"):
            w[name + "_k"] = shuffle_gemm_weight(w[name]) if gemm_shuffled else w[name]
        w["wkc_k"] = w["wkc"].view(HEADS, NOPE, KV).transpose(1, 2).contiguous().view(HEADS * KV, NOPE)
        w["wvc_k"] = w["wvc"].view(HEADS, KV, VD).transpose(1, 2).contiguous().view(HEADS * VD, KV)


def _agg1_wait(mpk):
    """(as test_kda_moe_layer.agg1_prefix_wait) the posted moe_agg1's second count: its y / bank row / agg_prefix written"""
    n = [v for v in mpk._static_nodes.values() if v["name"] == "moe_agg1"]
    assert len(n) == 1 and n[0]["counter"] >= 0
    return (n[0]["counter"] + 1, mpk.tokens)   # (one count per moe_agg1 task: a task per token)


def build_graph(mpk, w: dict, a, x_in=None, prefix_in=None):
    """the MLA layer's nodes (a: kv_splits, group, norm_groups, ar_chunks); returns (hidden, new_prefix), the layer's outputs.
    x_in / prefix_in: graph tensors of the layer before in the same kernel (test_kda_moe_layer.py --order moe), else the
    inputs x / prefix"""
    from mirage.core import bfloat16, float32
    given = {"x": x_in, "prefix": prefix_in}
    att = {k: given[k] if given.get(k) is not None else mpk.attach_input(torch_tensor=w[k], name=k) for k in (
        "x", "wa_k", "wq_norm", "wkv_norm", "cache", "info", "out_loc", "wqb_k", "wkc_k", "req_to_token", "seq_lens", "req_idx",
        "wg_k", "wvc_k", "wo_k", "bank",
        "prefix", "q", "gain", "ow", "hidden", "new_prefix")}
    latent = mpk.new_tensor(dims=(T, NA_PAD), dtype=float32, name="latent")
    q_a = mpk.new_tensor(dims=(T, QL), dtype=bfloat16, name="q_a")
    qb = mpk.new_tensor(dims=(T, NQ), dtype=bfloat16, name="q_heads")   # (not "q": the residual's score weight)
    gp = mpk.new_tensor(dims=(T, NG), dtype=float32, name="gate_parts")
    attn = mpk.new_tensor(dims=(T, HEADS * KV), dtype=bfloat16, name="attn")
    o_in = mpk.new_tensor(dims=(T, NG), dtype=bfloat16, name="o_in")
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="oproj_sent")
    mpk.gemm_tile_layer(input=att["x"], weight=att["wa_k"], output=latent, combine="slots")
    mpk.mla_norm_layer(latent=latent, w_q=att["wq_norm"], w_kv=att["wkv_norm"], kv_cache=att["cache"], out_loc=att["out_loc"], q_a=q_a,
                       eps_q=EPS, eps_kv=EPS, groups=a.norm_groups)
    mpk.gemm_tile_layer(input=q_a, weight=att["wqb_k"], output=qb, combine="local", input_polled=True)
    mpk.gemm_tile_layer(input=att["x"], weight=att["wg_k"], output=gp, combine="slots")
    query = mpk.new_tensor(dims=(T, HEADS * (KV + RP)), dtype=bfloat16, name="query")
    mpk.mla_absorb_layer(q=qb, w_kc=att["wkc_k"], output=query)
    mpk.mla_attn_layer(query=query, norm_out=q_a, kv_cache=att["cache"], table=att["req_to_token"], info=att["info"],
                       seq_lens=att["seq_lens"], req_idx=att["req_idx"], out_loc=att["out_loc"], output=attn, page_size=1,
                       scale=SCALE, kv_splits=a.kv_splits, group=a.group)
    mpk.mla_vgate_layer(attn=attn, gate_parts=gp, w_vc=att["wvc_k"], output=o_in)
    mpk.gemm_tile_layer(input=o_in, weight=att["wo_k"], output=sent, combine="push_all", input_polled=True)
    mpk.attn_residual_layer(input=sent, bank=att["bank"], prefix=att["prefix"], q=att["q"], gain=att["gain"], ow=att["ow"],
                            output=att["hidden"], u_output=att["new_prefix"], info=att["info"], eps=EPS, chunks=a.ar_chunks,
                            prefix_wait=_agg1_wait(mpk) if prefix_in is not None else None)
    return att["hidden"], att["new_prefix"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", type=int, default=8, choices=(1, 2, 4, 8))
    p.add_argument("--kv-rows", type=int, default=37, help="the request's KV rows before this step")
    p.add_argument("--kv-splits", type=int, default=1, help="mla_attn's KV splits per head (grid y)")
    p.add_argument("--norm-groups", type=int, default=1, help="mla_norm's tasks per token (column ranges)")
    p.add_argument("--die-group", type=int, default=None, help="graph_idx of mla_attn: plan_one_gpu.die_plan (a KV split's head groups on one die)")
    p.add_argument("--group", type=int, default=1, help="mla_attn's heads per task (2: tasks/mla_attn_grp.cuh)")
    p.add_argument("--out", default="mla_layer")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--wait", type=float, default=60)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--flush-mb", type=int, default=0)
    p.add_argument("--e2e-reps", type=int, default=20)
    p.add_argument("--ar-chunks", type=int, default=0, help="attn_residual tasks per token (0: H / 1024; 4: 1792-column chunks)")
    p.add_argument("--gemm-shuffled", action="store_true")
    p.add_argument("--grids", default=None)
    p.add_argument("--save-outs", default=None, help="launch 1's outputs of every GPU to this .pt (test_rank_lib.py compares)")
    p.add_argument("--dedicate", default=None, help="comma-separated graph_idx: plan_one_gpu.dedicate_plan")
    p.add_argument("--rank", default=None, help="N,F1,F2..: plan_one_gpu.rank_plan (node N's low splits on the CUs with the least "
                   "work before it, nodes F* not counted)")
    p.add_argument("--light", default=None, help="N:H[,N:H]: plan_one_gpu.light_plan (node N off the CUs with the most H tasks)")
    p.add_argument("--apart", default=None, help="N,A1,A2..: plan_one_gpu.apart_plan (node N one per CU, off the CUs of A*)")
    p.add_argument("--first", default=None, help="comma-separated graph_idx: plan_one_gpu.first_plan")
    p.add_argument("--codegen", action="store_true")
    a = p.parse_args()
    assert a.kv_rows + T <= min(MAX_CTX, POOL_ROWS)
    if a.codegen and not torch.cuda.is_available():
        for fn in ("empty", "zeros", "ones", "full", "arange", "tensor", "empty_like", "zeros_like", "randn", "cat", "randperm"):
            def on_cpu(*args, _orig=getattr(torch, fn), **kw):
                if str(kw.get("device", "")).startswith("cuda"):
                    kw["device"] = "cpu"
                return _orig(*args, **kw)
            setattr(torch, fn, on_cpu)

        class Props:
            major, minor, multi_processor_count, gcnArchName = 9, 5, 256, "gfx950"
            name, total_memory = "no GPU (codegen check)", 288 << 30
        torch.cuda.get_device_properties = lambda *_: Props
        torch.cuda.current_device = lambda: 0
        torch.cuda.device_count = lambda: 8
        torch.cuda.set_device = lambda *_: None
        _gen = torch.Generator
        torch.Generator = lambda device=None: _gen()
    from mirage.core import bfloat16, float32
    from mirage.mpk import compiler
    from mirage.mpk.static_harness import Harness
    from mirage.mpk.static_megakernel import StaticMegakernel
    from plan_one_gpu import simple_plan
    n = a.gpus
    if a.codegen:
        ws = [random_weights(torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda", 0), a.seed, a.kv_rows)]
    else:
        ws = [random_weights(torch.device("cuda", 0), a.seed, a.kv_rows)]
        ws += [random_weights(torch.device("cuda", g), a.seed + g, a.kv_rows, common=ws[0]) for g in range(1, n)]
    cache0 = ws[0]["cache"].cpu().clone()
    torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    prep_weights(ws, a.gemm_shuffled)
    if a.gemm_shuffled:
        mpk.gemm_weight_layout = "shuffled"
    build_graph(mpk, ws[0], a)
    plan = simple_plan(mpk, json.loads(a.grids) if a.grids else None)
    if a.first:   # (plan_one_gpu.first_plan: these nodes' tasks first on CUs of their own)
        from plan_one_gpu import first_plan
        plan = first_plan(plan, {int(x) for x in a.first.split(",")})
    if a.dedicate:   # (plan_one_gpu.dedicate_plan: these nodes' tasks alone on CUs of their own, first)
        from plan_one_gpu import dedicate_plan
        plan = dedicate_plan(plan, {int(x) for x in a.dedicate.split(",")})
    if a.die_group is not None:   # (plan_one_gpu.die_plan: the head groups of a KV split on one die, sharing its L2)
        from plan_one_gpu import die_plan
        plan = die_plan(plan, a.die_group)
    if a.rank:   # (plan_one_gpu.rank_plan: node N's low indices along axis 1 on the CUs with the least work before it)
        from plan_one_gpu import rank_plan
        rk = [int(x) for x in a.rank.split(",")]
        plan = rank_plan(plan, rk[0], 1, set(rk[1:]))
    if a.light:   # (plan_one_gpu.light_plan: node N's tasks off the CUs with the most tasks of node H)
        from plan_one_gpu import light_plan
        for nh in a.light.split(","):
            plan = light_plan(plan, *(int(x) for x in nh.split(":")))
    if a.apart:   # (plan_one_gpu.apart_plan: node N's tasks one per CU, on CUs without tasks of the nodes A*)
        from plan_one_gpu import apart_plan
        ap = [int(x) for x in a.apart.split(",")]
        plan = apart_plan(plan, ap[0], set(ap[1:]))
    print("plan grids:", dict(plan["grids"]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
    if a.codegen:
        import re
        import subprocess
        from mirage.mpk import static_schedule
        code = static_schedule.generate_code(mpk, [json.load(open(q_)) for q_ in paths], a.profile)
        cu = os.path.join(a.out, "layer.cu")
        with open(cu, "w") as f:
            f.write(code)
        cmd = static_schedule.compile_command(cu, os.path.join(a.out, "layer.so"), a.flag + ["-Wall", "--save-temps"], arch="gfx950")
        r = subprocess.run(cmd, cwd=a.out, capture_output=True, text=True)
        print(f"layer.cu {len(code.splitlines())} lines; hipcc exit {r.returncode}, {r.stderr.count('warning:')} warnings", flush=True)
        if r.returncode or "warning:" in r.stderr:
            print("\n".join(l for l in r.stderr.splitlines() if "error" in l or "warning:" in l or "note:" in l)[:6000])
        if r.returncode == 0:
            sasm = open(os.path.join(a.out, "layer-hip-amdgcn-amd-amdhsa-gfx950.s")).read()
            print({k: re.search(rf"; {k}: (\d+)", sasm).group(1) for k in ("NumVgprs", "NumAgprs", "ScratchSize", "Occupancy")})
        return
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag, profile=a.profile)
    h = Harness(sk)
    outs = ("hidden", "new_prefix", "cache")
    runs = []
    for launch in range(2):
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
            node_times(h, paths, compiler.graph_nodes(mpk))
    rep = Report()
    print(f"checks of launch 1 ({a.kv_rows} KV rows before the step):")
    check_all(ws, h, rep, cache0, a.kv_rows)
    for k in outs:
        rep.check(f"launch 1 == launch 0: {k}", all(same_bits([r0[k], r1[k]]) for r0, r1 in zip(runs[0], runs[1])), f"bits")
    print("ALL CHECKS PASSED" if not rep.failed else f"FAILED: {rep.failed}", flush=True)
    if a.save_outs:
        torch.save([{k: v.cpu() for k, v in r.items()} for r in runs[1]], a.save_outs)
        print("saved launch 1's outputs to", a.save_outs, flush=True)
    if a.e2e_reps and not rep.failed:
        e2e, spans, per_gpu = [], [], []
        for i in range(a.e2e_reps):
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
            per_gpu.append(sk.elapsed_us())
            e2e.append(max(per_gpu[-1]))
            spans.append(h.span_us())
        for g in range(n):
            torch.cuda.synchronize(g)
        last = [{k: w[k].clone() for k in outs} for w in ws]
        ok = all(all(same_bits([r0[k], r1[k]]) for r0, r1 in zip(runs[0], last)) for k in outs)
        print("CHECK OF THE LAST TIMED LAUNCH " + ("PASSED" if ok else "FAILED"), flush=True)
        e2e.sort()
        spans.sort()
        med = [sorted(p_[g] for p_ in per_gpu)[len(per_gpu) // 2] for g in range(n)]
        print(f"LATENCY ({a.e2e_reps} launches, {n} GPUs, {a.kv_rows} KV rows, {'timing build' if a.profile else 'normal build'}, "
              f"{a.flush_mb} MB flushed before each): e2e median {e2e[len(e2e) // 2]:.1f} us (min {e2e[0]:.1f}, max {e2e[-1]:.1f}); "
              f"span median {spans[len(spans) // 2]:.1f} us; e2e median per GPU {[round(m, 1) for m in med]}", flush=True)


if __name__ == "__main__":
    main()
