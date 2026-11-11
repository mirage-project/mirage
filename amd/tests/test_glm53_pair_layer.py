"""One GLM-5.3 layer as ONE kernel on AMD GPUs (TP 8, plain decode): the attention sublayer then its MLP (MoE, or the dense MLP
of layers 0 .. 2), the graph and the weight loaders of the server (amd/serving/glm53/glm53_layers.py). Checks that the outputs are
finite, equal on every GPU and bit-identical over relaunches (the sublayers' numerics: test_glm53_attn_layer.py /
test_glm53_moe_layer.py, the same task graphs); then the e2e latency. --codegen --serving: the layer.cu of a rank library
(amd/tools/gen_rank_lib.py --kind g53full | g53shared | g53dense).

    python amd/tests/test_glm53_pair_layer.py --tokens 8 [--layer 6] [--ctx 200] [--codegen [--serving]] [--profile]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "serving", "glm53"))
import glm53_layers as gl  # noqa: E402
from test_glm_mhc import cpu_standins  # noqa: E402


def standin_weights(full, mlp, n):
    """zero tensors of the weights' shapes and dtypes (codegen: no checkpoint read)"""
    NH, IR, F = gl.NHT // n, gl.I_MOE // n, gl.I_DENSE // n
    b, f, u8 = torch.bfloat16, torch.float32, torch.uint8
    wd = u8 if gl.FP8 else b   # (the gemms' weights: e4m3 bytes + fp32 scales *_s, glm53_layers.FP8)
    s = {"w_qkva": ((gl.NA, gl.H), wd), "q_norm": ((gl.QL,), b), "kv_norm": ((gl.KVL,), b), "w_qb": ((NH * (gl.NOPE + gl.RP), gl.QL), wd),
         "w_kc": ((NH * gl.KVL, gl.NOPE), b), "w_vc": ((NH * gl.DV, gl.KVL), b), "w_o": ((gl.H, NH * gl.DV), wd), "post_norm": ((gl.H,), b),
         "next_norm": ((gl.H,), b)}
    sc = lambda nm: ((s[nm][0][0] // 128, s[nm][0][1] // 128), f)
    if full:
        s.update({"w_k": ((gl.ID, gl.H), wd), "k_nw": ((gl.ID,), f), "k_nb": ((gl.ID,), f)})
    if mlp == "moe":
        NB = gl.E + 1
        s.update({"router": ((gl.E, gl.H), b), "bias": ((gl.E,), f), "w13": ((NB * 2 * IR, gl.H), u8),
                  "w13_sf": ((NB * (IR // 128) * (gl.H // 128) * 2,), f), "w2": ((NB * gl.H, IR), u8),
                  "w2_sf": ((NB * (gl.H // 128) * (IR // 128),), f)})
    else:
        s.update({"w_gu": ((2 * F, gl.H), wd), "w_dn": ((gl.H, F), wd)})
    if gl.FP8:
        s.update({nm + "_s": sc(nm) for nm in ("w_qkva", "w_qb", "w_o", "w_k", "w_gu", "w_dn") if nm in s})
    return {k: torch.zeros(*sh, dtype=dt) for k, (sh, dt) in s.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--model-dir", default="/dev/shm/glm53")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--ctx", type=int, default=200)
    p.add_argument("--flush-mb", type=int, default=512, help="MB written before each timed launch (L2 / MALL emptied); 0: warm")
    p.add_argument("--splits", type=int, default=0, help="sparse-attention splits (0: 8 at T <= 8, else 4)")
    p.add_argument("--gemm-cap", type=int, default=256, help="wide gemm grids up to this many tasks (0: the defaults)")
    p.add_argument("--store-kp", type=int, default=4)
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--out", default="glm53_pair")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--split-loop", type=int, default=1, help="1: one task loop per sublayer (MPK_SPLIT_LOOP at the MLP's first node)")
    p.add_argument("--serving", action="store_true")
    p.add_argument("--codegen", action="store_true")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    T, n = a.tokens, a.gpus
    os.environ["MPK_TOKENS"] = str(T)
    a.flag = gl.apply_settings(a.flag)   # (GLM-5.3's build settings: glm53_layers.G53_FLAGS / G53_ENV)
    if a.codegen:
        cpu_standins()
    from mirage.core import bfloat16, float32, int32, int64, uint8  # noqa: F401
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import StaticMegakernel
    from plan_one_gpu import simple_plan
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    eps = cfg["rms_norm_eps"]
    full = cfg["indexer_types"][a.layer] == "full"
    mlp = "dense" if a.layer < cfg["first_k_dense_replace"] else "moe"
    kind = "g53dense" if mlp == "dense" else ("g53full" if full else "g53shared")
    print(f"layer {a.layer}: indexer {'full' if full else 'shared'}, MLP {mlp}: kind {kind}", flush=True)
    ck = gl.Ckpt(a.model_dir)
    # run tensors: requests t of ctx + 3 t earlier tokens, rows a random permutation (row 0: SGLang's pad slot)
    gen = torch.Generator().manual_seed(31)
    seq = torch.tensor([a.ctx + 3 * t + 1 for t in range(T)], dtype=torch.int32)
    assert int(seq.max()) <= gl.K_TOP
    W2T = gl.K_TOP if a.serving else ((int(seq.max()) + 63) // 64) * 64 + 64   # (the server's table: [T][2048], sglang_glm53.py)
    NROW = 64 * (T * W2T // 64 + 2)
    r2t = (torch.randperm(NROW - 64, generator=gen)[:T * W2T] + 64).view(T, W2T).to(torch.int32)
    pos = (seq - 1).long()
    out_loc = r2t[torch.arange(T), pos].long()
    cache0 = (torch.randn(NROW, gl.KVL + gl.RP, generator=gen) * 2).to(torch.float8_e4m3fn).view(torch.uint8)
    inv = 1.0 / (8000000.0 ** (torch.arange(0, gl.RP, 2, dtype=torch.float32) / gl.RP))
    fr = torch.einsum("i,j->ij", torch.arange(int(seq.max()) + 8, dtype=torch.float32), inv)
    cos, sin = gl.bf(fr.cos()), gl.bf(fr.sin())
    rows_in = torch.full((T, gl.K_TOP), -1, dtype=torch.int32)
    for t in range(T):
        rows_in[t, :int(seq[t])] = r2t[t, :int(seq[t])]
    x = gl.bf(torch.randn(T, gl.H, generator=gen))
    res = gl.bf(torch.randn(T, gl.H, generator=gen) * 4)
    ws = []
    for g in range(n):
        dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
        if a.codegen:
            w = standin_weights(full, mlp, n)
        else:
            w = gl.attn_weights(ck, cfg, a.layer, g, n, dev)
            w.update(gl.moe_weights(ck, cfg, a.layer, g, n, dev) if mlp == "moe" else gl.dense_weights(ck, cfg, a.layer, g, n, dev))
        d = {k: v.to(dev) for k, v in w.items()}
        d.update({"x_attn": x.to(dev), "res_in": res.to(dev), "cache": cache0.clone().to(dev), "out_loc": out_loc.to(dev),
                  "positions": pos.to(dev), "seq_lens": seq.to(dev), "cos": cos.to(dev), "sin": sin.to(dev),
                  "res_out": torch.zeros(T, gl.H, dtype=torch.bfloat16, device=dev),
                  "res_mid": torch.zeros(T, gl.H, dtype=torch.bfloat16, device=dev),
                  "x_next": torch.zeros(T, gl.H, dtype=torch.bfloat16, device=dev)})
        if full:
            d.update({"icache": torch.zeros(NROW // 64, 64 * (gl.ID + 4), dtype=torch.uint8, device=dev), "r2t": r2t.to(dev),
                      "req_idx": torch.arange(T, dtype=torch.int64, device=dev),
                      "rows_g": torch.zeros(T, gl.K_TOP, dtype=torch.int32, device=dev)})
        else:
            d["rows"] = rows_in.to(dev)
        ws.append(d)
        if g == 0 and not a.codegen:
            print(f"GPU 0 weights loaded ({sum(v.numel() * v.element_size() for v in w.values()) / 2**20:.0f} MiB)", flush=True)
    if not a.codegen:
        torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    at = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ws[0]}
    gl.build_layer(mpk, at, T, n, full, mlp, eps, splits=a.splits or (8 if T <= 8 else 4), store_kp=a.store_kp)
    nodes = compiler.graph_nodes(mpk)
    if a.split_loop and "MPK_SPLIT_LOOP" not in os.environ:   # (one task loop per sublayer, as GLM-5.3-Flash's pairs: glmpair2 -6..-17 us)
        # (the attention's last node: its all-reduce, which does the add + RMSNorm)
        ar = [q.graph_idx for q in nodes if q.name == "allreduce_send"][0]
        os.environ["MPK_SPLIT_LOOP"] = str(min(q.graph_idx for q in nodes if q.graph_idx > ar))
        print(f"MPK_SPLIT_LOOP={os.environ['MPK_SPLIT_LOOP']} (the MLP part's first node)", flush=True)
    from glm_plan import wide_gemm_grids
    grids = wide_gemm_grids(mpk, a.gemm_cap) if a.gemm_cap else {}
    plan = simple_plan(mpk, grids)
    print("nodes:", [(q.graph_idx, q.name, q.grid) for q in nodes], flush=True)
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
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag, profile=a.profile)
    h = Harness(sk)
    failed = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            failed.append(name)
    first = None
    for launch in range(a.launches):
        for g in range(n):
            ws[g]["cache"].copy_(cache0.to(ws[g]["cache"].device))
        h.launch(512 << 20, timeout_s=60)
        for g in range(n):
            torch.cuda.synchronize(g)
        G0 = {k: ws[0][k].cpu().clone() for k in ("res_out", "x_next")}
        rs = gl.rs_on(T, n)   # (RSN: residual_out only on each token's owner - its rows put together; x_next on every GPU)
        if rs:
            G0["res_out"] = gl.owner_rows([ws[g]["res_out"].cpu() for g in range(n)], T, n)
        check(f"launch {launch}: outputs finite", all(bool(torch.isfinite(v.float()).all()) for v in G0.values()))
        check(f"launch {launch}: every GPU's outputs equal", all(torch.equal(ws[g][k].cpu(), G0[k]) for g in range(1, n) for k in G0
                                                                  if not (rs and k == "res_out")))
        if first is None:
            first = G0
        else:
            check(f"launch {launch}: bit-identical to launch 0", all(torch.equal(G0[k], first[k]) for k in G0))
    if not failed and a.reps:
        us = []
        for _ in range(a.reps):
            h.launch(a.flush_mb << 20, timeout_s=60)
            us.append(max(h.sk.elapsed_us()))
        us.sort()
        print(f"LATENCY e2e median {us[len(us) // 2]:.1f} us (min {us[0]:.1f}, max {us[-1]:.1f}; {a.reps} launches, L2 "
              f"emptied, slowest GPU)" + (f" [flush {a.flush_mb} MB]" if a.flush_mb != 512 else ""), flush=True)
        if a.profile:
            from test_glm_moe_layer import node_times
            node_times(h, paths, compiler.graph_nodes(mpk))
    print("CHECKS " + ("PASS" if not failed else f"FAILED {failed}"), flush=True)


if __name__ == "__main__":
    main()
