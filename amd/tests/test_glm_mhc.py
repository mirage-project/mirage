"""GLM-5.3-Flash's attention-sublayer boundary on AMD GPUs (TP4): o_proj as a push_all gemm_tile (each GPU's heads -> a
bf16 partial of all H rows to every GPU) -> hc_post_mix (the GPUs' partials added, the attention's mHC post R' = post out +
comb^T R, the FFN pre's partial mixes) -> hc_pre (x_ffn = RMSNorm(sum pre_i R'_i) gamma, h_post / h_res of the FFN).
The mHC parameters (hc_ffn_fn / base / scale, post_attention_layernorm) of layer --layer from the checkpoint; o, w_o, R and
the attention's h_post / h_res random. Checked per launch against upstream's references (demo/glm53_flash/dsa_fused_demo.py
hc_post_ref / hc_pre_ref: SGLang's formulas): the all-reduced out, R', x_ffn, h_post, h_res; every GPU's outputs equal;
launch k bit-identical to launch 0.

    python amd/tests/test_glm_mhc.py --tokens 8 [--gpus 4] [--model-dir /dev/shm/glm53f --layer 4] [--codegen]
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

H, HC, NH, HD = 4096, 4, 64, 128   # hidden, streams, KDA heads, head dim


def bf(x):
    return x.to(torch.bfloat16)


def relrms(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def load_mhc(model_dir: str, layer: int, which: str = "ffn") -> dict:
    """hc_<which>_fn [24, HC H] (as fp32), base [24], scale [3], and the sublayer's input norm weight [H] bf16"""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    p = f"model.language_model.layers.{layer}."
    norm = "post_attention_layernorm.weight" if which == "ffn" else "input_layernorm.weight"
    names = {"fn": f"hc_{which}_fn", "base": f"hc_{which}_base", "scale": f"hc_{which}_scale", "gamma": norm}
    out = {}
    for k, n in names.items():
        with safe_open(os.path.join(model_dir, idx[p + n]), "pt") as f:
            out[k] = f.get_tensor(p + n)
    out["fn"] = out["fn"].float()
    return out


def hc_pre_ref(R, fn, base, scale, gamma, rms_eps, hc_eps, norm_eps, iters, post_mult=2.0):
    """upstream dsa_fused_demo.hc_pre_ref (sglang_norm): SGLang's _mhc_pre_torch + its fused norm"""
    n = R.shape[0]
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
    return bf(li * torch.rsqrt(s.square().mean(-1, keepdim=True) + norm_eps) * gamma.float()), post, comb.reshape(n, HC * HC)


def hc_post_ref(out, R, h_post, h_res):
    """upstream dsa_fused_demo.hc_post_ref: SGLang's _mhc_post_torch"""
    n = R.shape[0]
    comb = h_res.view(n, HC, HC)
    mixed = (comb.unsqueeze(-1) * R.float().unsqueeze(2)).sum(1)
    return bf(h_post.view(n, HC, 1) * out.float().unsqueeze(1) + mixed)


def cpu_standins():
    """--codegen without a GPU: torch tensors asked for on a GPU are made on the CPU, an MI355X's properties stand in"""
    for fn_ in ("empty", "zeros", "ones", "full", "randn", "rand"):
        def on_cpu(*args, _orig=getattr(torch, fn_), **kw):
            if str(kw.get("device", "")).startswith("cuda"):
                kw["device"] = "cpu"
            return _orig(*args, **kw)
        setattr(torch, fn_, on_cpu)

    class Props:
        major, minor, multi_processor_count, gcnArchName = 9, 5, 256, "gfx950"
        name, total_memory = "no GPU (codegen check)", 288 << 30
    torch.cuda.get_device_properties = lambda *_: Props
    torch.cuda.current_device = lambda: 0
    torch.cuda.device_count = lambda: 8
    torch.cuda.set_device = lambda *_: None
    torch.cuda.synchronize = lambda *_: None
    _gen = torch.Generator
    torch.Generator = lambda device=None: _gen()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--model-dir", default="/dev/shm/glm53f")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--o-grid", default="32,1,1", help="o_proj's grid (x: 4096 / rows per task)")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--out", default="glm_mhc")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--reps", type=int, default=20, help="timed launches after the checks (L2 emptied before each)")
    p.add_argument("--codegen", action="store_true", help="no GPU: layer.cu + hipcc --save-temps only")
    a = p.parse_args()
    T, n = a.tokens, a.gpus
    os.environ["MPK_TOKENS"] = str(T)
    if a.codegen:
        cpu_standins()
    from mirage.core import float32
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import HC_MIX_STRIDE, StaticMegakernel
    from plan_one_gpu import simple_plan
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    rms_eps, hc_eps, iters = cfg["rms_norm_eps"], cfg["hc_eps"], cfg["hc_sinkhorn_iters"]
    m = load_mhc(a.model_dir, a.layer, "ffn")
    gen = torch.Generator().manual_seed(5)
    R = bf(torch.randn(T, HC, H, generator=gen))
    o = [bf(torch.randn(T, NH // n * HD, generator=gen)) for _ in range(n)]
    w_o = bf(torch.randn(H, NH * HD, generator=gen) * (NH * HD) ** -0.5)
    hp = (torch.rand(T, HC, generator=gen) * 2).float()
    hr = torch.rand(T, HC * HC, generator=gen).float()
    hc_small = torch.zeros(32, dtype=torch.float32)
    hc_small[:HC * (HC + 2)], hc_small[HC * (HC + 2):HC * (HC + 2) + 3] = m["base"], m["scale"]
    ws = []
    for g in range(n):
        dev = torch.device("cpu") if a.codegen else torch.device("cuda", g)
        cols = slice(g * NH // n * HD, (g + 1) * NH // n * HD)
        ws.append({"o": o[g].to(dev), "w_o": w_o[:, cols].contiguous().to(dev), "R_in": R.reshape(T, HC * H).to(dev),
                   "h_post_in": hp.to(dev), "h_res_in": hr.to(dev), "hc_ffn_fn": m["fn"].to(dev),
                   "hc_ffn_small": hc_small.to(dev), "ffn_norm": m["gamma"].to(dev),
                   "R_out": torch.zeros(T, HC * H, dtype=torch.bfloat16, device=dev),
                   "x_ffn": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
                   "h_post_out": torch.zeros(T, HC, dtype=torch.float32, device=dev),
                   "h_res_out": torch.zeros(T, HC * HC, dtype=torch.float32, device=dev)})
    torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    at = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ws[0]}
    from mirage.core import bfloat16
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="o_sent")
    mpk.gemm_tile_layer(input=at["o"], weight=at["w_o"], output=sent, combine="push_all")
    mix = mpk.new_tensor(dims=(T, HC_MIX_STRIDE), dtype=float32, name="ffn_hc_mix")
    mpk.hc_post_mix_layer(sent=sent, residual=at["R_in"], coef=(at["h_post_in"], at["h_res_in"]), fn=at["hc_ffn_fn"],
                          output=at["R_out"], mix=mix)
    mpk.hc_pre_layer(mix=mix, residual=at["R_out"], hc_small=at["hc_ffn_small"], gamma=at["ffn_norm"], output=at["x_ffn"],
                     rms_eps=rms_eps, hc_eps=hc_eps, norm_eps=rms_eps, sinkhorn_iters=iters,
                     export=(at["h_post_out"], at["h_res_out"]))
    nodes = compiler.graph_nodes(mpk)
    plan = simple_plan(mpk, {nodes[0].graph_idx: tuple(int(v) for v in a.o_grid.split(","))})
    print("nodes:", [(x.graph_idx, x.name, x.grid) for x in nodes], "plan grids:", dict(plan["grids"]), flush=True)
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
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag)
    h = Harness(sk)
    # references (CPU, fp32 as SGLang)
    parts = [bf(o[g].float() @ w_o[:, g * NH // n * HD:(g + 1) * NH // n * HD].float().t()) for g in range(n)]
    acc = torch.zeros(T, H)
    for q in parts:
        acc = acc + q.float()
    out = bf(acc)
    R2 = hc_post_ref(out, R, hp, hr)
    x_ref, post_ref, comb_ref = hc_pre_ref(R2, m["fn"], m["base"], m["scale"], m["gamma"], rms_eps, hc_eps, rms_eps, iters)
    failed = []

    def check(name, ok, detail):
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            failed.append(name)
    first = None
    for launch in range(a.launches):
        h.launch(512 << 20, timeout_s=60)
        for g in range(n):
            torch.cuda.synchronize(g)
        got = [{k: ws[g][k].cpu().clone() for k in ("R_out", "x_ffn", "h_post_out", "h_res_out")} for g in range(n)]
        G0 = got[0]
        R2k = G0["R_out"].view(T, HC, H)
        check(f"launch {launch}: R' vs SGLang's mhc_post", relrms(R2k, R2) < 1e-2,
              f"relRMS {relrms(R2k, R2):.2e}, bit-equal {(R2k == R2).float().mean().item():.4f}")
        # x / post / comb from the kernel's own R' (the step alone)
        xk, pk, ck = hc_pre_ref(R2k, m["fn"], m["base"], m["scale"], m["gamma"], rms_eps, hc_eps, rms_eps, iters)
        check(f"launch {launch}: x_ffn (from the kernel's R')", relrms(G0["x_ffn"], xk) < 1e-2,
              f"relRMS {relrms(G0['x_ffn'], xk):.2e} (vs the full reference {relrms(G0['x_ffn'], x_ref):.2e})")
        check(f"launch {launch}: h_post", relrms(G0["h_post_out"], pk) < 1e-4, f"relRMS {relrms(G0['h_post_out'], pk):.2e}")
        check(f"launch {launch}: h_res", relrms(G0["h_res_out"], ck) < 1e-4, f"relRMS {relrms(G0['h_res_out'], ck):.2e}")
        same = all(torch.equal(got[g][k], G0[k]) for g in range(1, n) for k in G0)
        check(f"launch {launch}: every GPU's outputs equal", same, "")
        if first is None:
            first = G0
        else:
            check(f"launch {launch}: bit-identical to launch 0", all(torch.equal(G0[k], first[k]) for k in G0), "")
    if not failed and a.reps:
        us = []
        for _ in range(a.reps):
            h.launch(512 << 20, timeout_s=60)
            us.append(max(h.sk.elapsed_us()))
        us.sort()
        print(f"LATENCY e2e median {us[len(us) // 2]:.1f} us (min {us[0]:.1f}, max {us[-1]:.1f}; {a.reps} launches, L2 "
              f"emptied, slowest GPU)", flush=True)
    print("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}", flush=True)


if __name__ == "__main__":
    main()
