"""The K3 MLA layer in plain decode (TP8, T requests with one new token each, 12 of the 96 heads per GPU) on AMD GPUs as one
static megakernel, with random weights in the real shapes and a random paged KV cache, checked against float64 references,
then timed like amd/tests/test_mla_layer.py.

Graph per GPU: test_mla_layer.py's, with mla_attn_decode (task = request x KV split, all 12 heads at once) in place of mla_attn,
mla_vgate in token groups of at most 16. Request t: its own table row (req_idx[t]), seq_lens[t] rows with this step's token
(SGLang decode), this step's row at out_loc[t]. With T > 1 the last row is padded as SGLang's captured graphs pad: seq_len
1, req_idx 0, out_loc 0.

Checks: q_a; the KV rows written, the cache else unchanged; q; attn of the live requests from the kernel's q and cache; o_in;
u and hidden from every GPU's o_in; hidden, u the same on every GPU; launch 1 == launch 0.

python amd/tests/test_mla_decode.py --tokens 64 --gpus 8 [--kv-rows 300] [--kv-splits S] [--flush-mb 512] [--e2e-reps 20]
MK_BACKEND=hip python amd/tests/test_mla_decode.py --tokens 64 --codegen
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
from test_layer_one_gpu import Report, bf16, node_times, relrms, same_bits  # noqa: E402
from test_mla_layer import (D, EPS, H, HEADS, KV, NA, NA_PAD, NB, NB_ALLOC, NG, NOPE, NQ, QL, RP, SCALE, SHARDED,  # noqa: E402
                            VD, prep_weights, read_buf, ref_residual, rmsnorm)

T = 8   # --tokens (main sets it)


def layout(seed: int, kv_rows: int) -> dict:
    """Per request: context rows before the step (kv_rows / 2 .. kv_rows), its table row, the pool size."""
    g = torch.Generator().manual_seed(seed + 2000)
    kv = torch.randint(max(1, kv_rows // 2), kv_rows + 1, (T,), generator=g).tolist()
    live = T - 1 if T > 1 else T
    reqs = T + 2
    req = (torch.randperm(reqs - 1, generator=g)[:T] + 1).tolist()   # table row 0 left for the padded request
    pool = max(10240, sum(kv[:live]) + live + 64)
    return dict(kv=kv, live=live, reqs=reqs, req=req, pool=pool, max_ctx=kv_rows + 8)


def random_weights(dev, seed: int, lay: dict, common: dict = None) -> dict:
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *shape: torch.randn(shape, generator=g, device=dev)
    b = lambda t: t.to(torch.bfloat16)
    table = torch.full((lay["reqs"], lay["max_ctx"]), -1, dtype=torch.int32, device=dev)
    perm = torch.randperm(lay["pool"] - 1, generator=g, device=dev).to(torch.int32) + 1   # (row 0: the padded token's)
    seq, ridx, oloc, at = [], [], [], 0
    for t in range(T):
        if t < lay["live"]:
            n = lay["kv"][t] + 1
            table[lay["req"][t], :n] = perm[at:at + n]
            seq.append(n)
            ridx.append(lay["req"][t])
            oloc.append(int(perm[at + n - 1]))
            at += n
        else:   # SGLang's padding of a captured graph
            seq.append(1)
            ridx.append(0)
            oloc.append(0)
    w = {
        "x": b(rn(T, H)),
        "wa": b(torch.cat([rn(NA, H) * H ** -0.5, torch.zeros(NA_PAD - NA, H, device=dev)])),
        "wq_norm": b(1 + 0.1 * rn(QL)), "wkv_norm": b(1 + 0.1 * rn(KV)),
        "wqb": b(rn(NQ, QL) * QL ** -0.5),
        "wkc": b(rn(HEADS * NOPE, KV) * NOPE ** -0.5),
        "wvc": b(rn(HEADS * KV, VD) * KV ** -0.5),
        "wg": b(rn(NG, H) * H ** -0.5),
        "wo": b(rn(H, NG) * NG ** -0.5),
        "cache": b(rn(lay["pool"], D)),
        "req_to_token": table,
        "seq_lens": torch.tensor(seq, dtype=torch.int64, device=dev),
        "req_idx": torch.tensor(ridx, dtype=torch.int64, device=dev),
        "out_loc": torch.tensor(oloc, dtype=torch.int64, device=dev),
        "bank": b(rn(T, NB_ALLOC, H)), "prefix": b(rn(T, H)), "q": b(rn(H) * H ** -0.5), "gain": b(1 + 0.1 * rn(H)),
        "ow": b(1 + 0.1 * rn(H)),
        "hidden": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
        "new_prefix": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
    }
    w["info"] = torch.tensor([NB, NB_ALLOC * H, H, lay["max_ctx"], 0, 0, 0, 0], dtype=torch.int32, device=dev)
    if common is not None:
        for k in w:
            if k not in SHARDED and k not in ("hidden", "new_prefix"):
                w[k] = common[k].to(dev).clone() if k == "cache" else common[k].to(dev)
    return w


def ref_front(w):
    latent = bf16(w["x"].double() @ w["wa"][:NA].double().T).double()
    q_a = rmsnorm(latent[:, :QL], w["wq_norm"])
    kv_new = torch.cat([rmsnorm(latent[:, QL:QL + KV], w["wkv_norm"]), latent[:, QL + KV:]], -1)
    gate = bf16(w["x"].double() @ w["wg"].double().T).double()
    return q_a, kv_new, gate


def ref_attn(q, w, cache, lay):
    """attn [live, 12 x 512] of the live requests from the kernel's q [T, NQ] and the cache (with this step's rows)."""
    qh = q.double().view(T, HEADS, NOPE + RP)
    wkc = w["wkc"].double().view(HEADS, NOPE, KV)
    q_abs = bf16(torch.einsum("thn,hnl->thl", qh[..., :NOPE], wkc)).double()
    out = []
    for t in range(lay["live"]):
        n = int(w["seq_lens"][t])
        rows = w["req_to_token"][int(w["req_idx"][t]), :n].long()
        K = cache[rows].double()
        s = (q_abs[t] @ K[:, :KV].T + qh[t, :, NOPE:] @ K[:, KV:].T) * SCALE   # [HEADS, n]
        e = torch.exp(s - s.amax(-1, keepdim=True))
        out.append(bf16((bf16(e).double() @ K[:, :KV]) / e.sum(-1, keepdim=True)).double().reshape(HEADS * KV))
    return torch.stack(out)


def check_all(ws, h, rep, cache0, lay):
    partials = []
    live = lay["live"]
    for gi, w in enumerate(ws):
        tag = f"GPU {gi}: "
        dev = w["x"].device
        q_a_ref, kv_ref, gate_ref = ref_front(w)
        q_a = read_buf(h, gi, "q_a", torch.bfloat16, (T, QL)).to(dev)
        rep.check(tag + "q_a", relrms(q_a, q_a_ref) < 2e-3, f"relRMS {relrms(q_a, q_a_ref):.2e}")
        rows = w["out_loc"][:live]
        rep.check(tag + "KV rows written", relrms(w["cache"][rows], kv_ref[:live]) < 2e-3,
                  f"relRMS {relrms(w['cache'][rows], kv_ref[:live]):.2e}")
        others = torch.ones(w["cache"].shape[0], dtype=torch.bool, device=dev)
        others[w["out_loc"]] = False
        rep.check(tag + "cache else unchanged", torch.equal(w["cache"][others].cpu(), cache0[others.cpu()]), "bits")
        q = read_buf(h, gi, "q_heads", torch.bfloat16, (T, NQ)).to(dev)
        q_ref = bf16(q_a.double() @ w["wqb"].double().T).double()
        rep.check(tag + "q (q_b)", relrms(q, q_ref) < 1e-3, f"relRMS {relrms(q, q_ref):.2e} (from the kernel's q_a)")
        attn = read_buf(h, gi, "attn", torch.bfloat16, (T, HEADS * KV)).to(dev)
        attn_ref = ref_attn(q, w, w["cache"], lay)
        rep.check(tag + "attn (live requests)", relrms(attn[:live], attn_ref) < 5e-3,
                  f"relRMS {relrms(attn[:live], attn_ref):.2e} (from the kernel's q and cache)")
        rep.check(tag + "attn finite", bool(torch.isfinite(attn.double()).all()), "all rows")
        o_in = read_buf(h, gi, "o_in", torch.bfloat16, (T, NG)).to(dev)
        v = bf16(torch.einsum("thl,hlv->thv", attn.double().view(T, HEADS, KV), w["wvc"].double().view(HEADS, KV, VD))).double()
        o_ref = bf16(v.reshape(T, NG) * bf16(torch.sigmoid(gate_ref)).double()).double()
        rep.check(tag + "o_in (w_vc, gate)", relrms(o_in, o_ref) < 5e-3, f"relRMS {relrms(o_in, o_ref):.2e} (from the kernel's attn)")
        partials.append(bf16(o_in.double() @ w["wo"].double().T).double())
    a = sum(p.to(ws[0]["x"].device) for p in partials)
    u_ref, hid_ref = ref_residual(a, ws[0])
    rep.check("u (new prefix)", relrms(ws[0]["new_prefix"], u_ref) < 2e-3, f"relRMS {relrms(ws[0]['new_prefix'], u_ref):.2e}")
    rep.check("hidden", relrms(ws[0]["hidden"], hid_ref) < 5e-3, f"relRMS {relrms(ws[0]['hidden'], hid_ref):.2e}")
    if len(ws) > 1:
        rep.check("hidden, u the same on every GPU", same_bits([w["hidden"] for w in ws]) and
                  same_bits([w["new_prefix"] for w in ws]), "bits")


def build_graph(mpk, w: dict, a, x_in=None, prefix_in=None, prefix_wait=None):
    """the MLA decode nodes; returns (hidden, new_prefix). x_in / prefix_in: graph tensors of the layer before in the same
    kernel (test_kda_moe_layer.py --order moe), else the inputs x / prefix; prefix_wait: attn_residual's (counter, need)"""
    from mirage.core import bfloat16, float32
    given = {"x": x_in, "prefix": prefix_in}
    att = {k: given[k] if given.get(k) is not None else mpk.attach_input(torch_tensor=w[k], name=k) for k in (
        "x", "wa_k", "wq_norm", "wkv_norm", "cache", "info", "out_loc", "wqb_k", "wkc_k", "req_to_token", "seq_lens", "req_idx",
        "wg_k", "wvc_k", "wo_k", "bank", "prefix", "q", "gain", "ow", "hidden", "new_prefix")}
    latent = mpk.new_tensor(dims=(T, NA_PAD), dtype=float32, name="latent")
    q_a = mpk.new_tensor(dims=(T, QL), dtype=bfloat16, name="q_a")
    qb = mpk.new_tensor(dims=(T, NQ), dtype=bfloat16, name="q_heads")
    gp = mpk.new_tensor(dims=(T, NG), dtype=float32, name="gate_parts")
    attn = mpk.new_tensor(dims=(T, HEADS * KV), dtype=bfloat16, name="attn")
    o_in = mpk.new_tensor(dims=(T, NG), dtype=bfloat16, name="o_in")
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="oproj_sent")
    mpk.gemm_tile_layer(input=att["x"], weight=att["wa_k"], output=latent, combine="slots")
    mpk.mla_norm_layer(latent=latent, w_q=att["wq_norm"], w_kv=att["wkv_norm"], kv_cache=att["cache"], out_loc=att["out_loc"],
                       q_a=q_a, eps_q=EPS, eps_kv=EPS, groups=int(os.environ.get("MPK_MLA_NORM_GROUPS", "1")))   # (tasks per token)
    mpk.gemm_tile_layer(input=q_a, weight=att["wqb_k"], output=qb, combine="local", input_polled=True)
    mpk.gemm_tile_layer(input=att["x"], weight=att["wg_k"], output=gp, combine="slots")
    query = mpk.new_tensor(dims=(T, HEADS * (KV + RP)), dtype=bfloat16, name="query")
    mpk.mla_absorb_layer(q=qb, w_kc=att["wkc_k"], output=query)
    mpk.mla_attn_decode_layer(query=query, norm_out=q_a, kv_cache=att["cache"], table=att["req_to_token"], info=att["info"],
                              seq_lens=att["seq_lens"], req_idx=att["req_idx"], out_loc=att["out_loc"], output=attn,
                              page_size=1, scale=SCALE, kv_splits=a.kv_splits)
    # MPK_MLA_OPROJ_POLLED=0: o_proj waits for mla_vgate's count and fills its activation by DMA (as KDA's o_proj at T > 16)
    polled = os.environ.get("MPK_MLA_OPROJ_POLLED", "1") == "1"
    mpk.mla_vgate_layer(attn=attn, gate_parts=gp, w_vc=att["wvc_k"], output=o_in, counted=not polled)
    rs = os.environ.get("MPK_AR_RS") == "1"   # (as test_kda_decode.py: o_proj push_rs + attn_residual RS)
    frs = rs and not polled and os.environ.get("MPK_OPROJ_FRS") == "1"   # (as test_kda_decode.py: K parts, combine finish_rs)
    mpk.gemm_tile_layer(input=o_in, weight=att["wo_k"], output=sent,
                        combine="finish_rs" if frs else "push_rs" if rs else "push_all", input_polled=polled)
    mpk.attn_residual_layer(input=sent, bank=att["bank"], prefix=att["prefix"], q=att["q"], gain=att["gain"], ow=att["ow"],
                            output=att["hidden"], u_output=att["new_prefix"], info=att["info"], eps=EPS, prefix_wait=prefix_wait,
                            rs=rs, chunks=int(os.environ.get("MPK_AR_CHUNKS", "0")))   # (MPK_AR_CHUNKS 4: 1792-column chunks, A/B)
    return att["hidden"], att["new_prefix"]


def main():
    global T
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=int(os.environ.get("MPK_TOKENS", "8")), help="requests (batch size)")
    p.add_argument("--gpus", type=int, default=8, choices=(1, 2, 4, 8))
    p.add_argument("--kv-rows", type=int, default=300, help="the most context rows of a request before this step")
    p.add_argument("--kv-splits", type=int, default=1, help="mla_attn_decode's KV splits per request (grid y)")
    p.add_argument("--out", default="mla_decode")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--wait", type=float, default=60)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--flush-mb", type=int, default=512)
    p.add_argument("--e2e-reps", type=int, default=20)
    p.add_argument("--gemm-shuffled", action="store_true")
    p.add_argument("--grids", default=None)
    p.add_argument("--codegen", action="store_true")
    a = p.parse_args()
    T = a.tokens
    os.environ["MPK_TOKENS"] = str(T)
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
    from mirage.mpk import compiler
    from mirage.mpk.static_harness import Harness
    from mirage.mpk.static_megakernel import StaticMegakernel
    from plan_one_gpu import simple_plan
    n = a.gpus
    lay = layout(a.seed, a.kv_rows)
    print(f"T = {T}: context rows {lay['kv'][:lay['live']]} (+ the step's), {T - lay['live']} padded", flush=True)
    if a.codegen:
        ws = [random_weights(torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda", 0), a.seed, lay)]
    else:
        ws = [random_weights(torch.device("cuda", 0), a.seed, lay)]
        ws += [random_weights(torch.device("cuda", g), a.seed + g, lay, common=ws[0]) for g in range(1, n)]
    cache0 = ws[0]["cache"].cpu().clone()
    torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    prep_weights(ws, a.gemm_shuffled)
    if a.gemm_shuffled:
        mpk.gemm_weight_layout = "shuffled"
    build_graph(mpk, ws[0], a)
    plan = simple_plan(mpk, json.loads(a.grids) if a.grids else None)
    print("plan grids:", dict(plan["grids"]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    compiler.write_plan_file(os.path.join(a.out, f"plan_{n}gpu.json"), plan)
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
    print("checks of launch 1:")
    check_all(ws, h, rep, cache0, lay)
    for k in outs:
        rep.check(f"launch 1 == launch 0: {k}", all(same_bits([r0[k], r1[k]]) for r0, r1 in zip(runs[0], runs[1])), "bits")
    print("ALL CHECKS PASSED" if not rep.failed else f"FAILED: {rep.failed}", flush=True)
    if a.e2e_reps and not rep.failed:
        e2e, spans = [], []
        for i in range(a.e2e_reps):
            h.launch(a.flush_mb << 20, timeout_s=a.wait)
            e2e.append(max(sk.elapsed_us()))
            spans.append(h.span_us())
        for g in range(n):
            torch.cuda.synchronize(g)
        e2e.sort()
        spans.sort()
        print(f"LATENCY (T = {T}, {a.e2e_reps} launches, {n} GPUs, context <= {a.kv_rows} rows, "
              f"{'timing build' if a.profile else 'normal build'}, {a.flush_mb} MB flushed before each): e2e median "
              f"{e2e[len(e2e) // 2]:.1f} us (min {e2e[0]:.1f}, max {e2e[-1]:.1f}); span median {spans[len(spans) // 2]:.1f} us",
              flush=True)


if __name__ == "__main__":
    main()
