"""One gemm_tile node on AMD GPUs with T tokens (the B columns: 16 per MFMA tile), checked per token against float64: the K
parts of combine "slots" (summed in part order), or the rows of combine "store" whose activation another node writes
(situ_and_mul-like producer: a gemm "store" from a polled-free input is enough here: the input is a graph tensor).

    python amd/tests/test_gemm_tokens.py --tokens 16 [--n 896 --k 7168 --grid 7,14,1] [--combine slots|store] [--flag ...]
    python amd/tests/test_gemm_tokens.py --tokens 32 --chain   (the MoE's shared expert alone: gate_up "add" 12 x 8 tasks ->
                                                               situ_and_mul 12 -> down "store" 56; S per token vs the
                                                               kernel's own h_s)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_layer_one_gpu import Report, relrms  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--n", type=int, default=896)
    p.add_argument("--k", type=int, default=7168)
    p.add_argument("--grid", default="7,14,1")
    p.add_argument("--combine", default="slots", choices=("slots", "store"))
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--gemm-shuffled", action="store_true")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--out", default="gemm_tokens")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--chain", action="store_true")
    p.add_argument("--codegen", action="store_true", help="no GPU: layer.cu + hipcc --save-temps only")
    a = p.parse_args()
    if a.codegen and not torch.cuda.is_available():   # (as test_kda_decode.py: the tensors on the CPU, an MI355X's props)
        for fn in ("empty", "zeros", "ones", "full", "randn"):
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
        torch.cuda.synchronize = lambda *_: None
        _gen = torch.Generator
        torch.Generator = lambda device=None: _gen()
    if a.chain:
        return chain(a)
    T, N, K, n = a.tokens, a.n, a.k, a.gpus
    os.environ["MPK_TOKENS"] = str(T)
    from mirage.core import float32
    from mirage.mpk import compiler
    from mirage.mpk.static_harness import Harness
    from mirage.mpk.static_megakernel import StaticMegakernel, shuffle_gemm_weight
    from plan_one_gpu import simple_plan
    ws = []
    for g in range(n):
        dev = torch.device("cuda", g)
        gen = torch.Generator(device=dev).manual_seed(1)
        x = torch.randn(T, K, generator=gen, device=dev).to(torch.bfloat16)
        w = (torch.randn(N, K, generator=gen, device=dev) * K ** -0.5).to(torch.bfloat16)
        ws.append({"x": x, "w": w, "w_k": shuffle_gemm_weight(w) if a.gemm_shuffled else w,
                   "y": torch.zeros(T, N, dtype=torch.float32, device=dev)})
    torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    if a.gemm_shuffled:
        mpk.gemm_weight_layout = "shuffled"
    att = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ("x", "w_k", "y")}
    out = att["y"] if a.combine == "store" else mpk.new_tensor(dims=(T, N), dtype=float32, name="parts")
    mpk.gemm_tile_layer(input=att["x"], weight=att["w_k"], output=out, combine=a.combine)
    gidx = compiler.graph_nodes(mpk)[0].graph_idx
    plan = simple_plan(mpk, {gidx: tuple(int(v) for v in a.grid.split(","))})
    print("plan grids:", dict(plan["grids"]), flush=True)
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
        print(f"hipcc exit {r.returncode}", r.stderr[-2000:] if r.returncode else "", flush=True)
        return
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag)
    h = Harness(sk)
    rep = Report()
    ref = ws[0]["x"].double() @ ws[0]["w"].double().T
    for launch in range(a.launches):
        h.launch(512 << 20, timeout_s=60)
        for g in range(n):
            torch.cuda.synchronize(g)
        if a.combine == "store":
            got = ws[0]["y"].double()
        else:
            raw = np.frombuffer(h.sk.read(0, "parts"), dtype=np.float32).copy()
            parts = torch.from_numpy(raw).view(-1, T, N).double().to(ref.device)
            got = parts.sum(0)
        bad = [t for t in range(T) if not relrms(got[t], ref[t]) < 1e-5]
        if os.environ.get("MPK_DBG_ACT") == "1":   # (gemm_tile STATIC_GEMM_DBG_ACT: LDS activation pieces != their source)
            st = h._array(0, "stamps", np.int64).reshape(h.layout["num_sms"], h.layout["num_stamps"])[0]
            print(f"  launch {launch}: LDS activation mismatches after the fill {st[8]} (last token {st[9] - 1}), after the "
                  f"MFMAs {st[10]} (last token {st[11] - 1})", flush=True)
        rep.check(f"launch {launch}: every token's row", not bad,
                  f"relRMS {relrms(got, ref):.2e}; tokens off: {bad} (per token: "
                  f"{[round(relrms(got[t], ref[t]), 4) for t in bad][:16]})")
    print("ALL CHECKS PASSED" if not rep.failed else f"FAILED: {rep.failed}", flush=True)


def chain(a):
    T, n, H, SHR = a.tokens, a.gpus, 7168, 768
    os.environ["MPK_TOKENS"] = str(T)
    from mirage.core import bfloat16, int64
    from mirage.mpk import compiler
    from mirage.mpk.static_harness import Harness
    from mirage.mpk.static_megakernel import StaticMegakernel, shuffle_gemm_weight
    from plan_one_gpu import simple_plan
    ws = []
    for g in range(n):
        dev = torch.device("cuda", g)
        gen = torch.Generator(device=dev).manual_seed(1)
        x = torch.randn(T, H, generator=gen, device=dev).to(torch.bfloat16)
        wsgu = (torch.randn(2 * SHR, H, generator=gen, device=dev) * H ** -0.5).to(torch.bfloat16)
        wsd = (torch.randn(H, SHR, generator=gen, device=dev) * SHR ** -0.5).to(torch.bfloat16)
        ws.append({"x": x, "wsgu": wsgu, "wsd": wsd, "wsgu_k": shuffle_gemm_weight(wsgu) if a.gemm_shuffled else wsgu,
                   "wsd_k": shuffle_gemm_weight(wsd) if a.gemm_shuffled else wsd,
                   "S": torch.zeros(T, H, dtype=torch.float32, device=dev)})
    torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=n)
    if a.gemm_shuffled:
        mpk.gemm_weight_layout = "shuffled"
    att = {k: mpk.attach_input(torch_tensor=ws[0][k], name=k) for k in ("x", "wsgu_k", "wsd_k", "S")}
    sgu = mpk.new_tensor(dims=(T, 2 * SHR), dtype=int64, name="sgu")
    hs = mpk.new_tensor(dims=(T, SHR), dtype=bfloat16, name="hs")
    mpk.gemm_tile_layer(input=att["x"], weight=att["wsgu_k"], output=sgu, combine="add")
    mpk.situ_and_mul_layer(input=sgu, output=hs)
    mpk.gemm_tile_layer(input=hs, weight=att["wsd_k"], output=att["S"], combine="store")
    gn = [x.graph_idx for x in compiler.graph_nodes(mpk)]
    plan = simple_plan(mpk, {gn[0]: (12, 8, 1), gn[1]: (12, 1, 1), gn[2]: (56, 1, 1)})
    print("plan grids:", dict(plan["grids"]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(a.out, "build"), extra_flags=a.flag)
    h = Harness(sk)
    rep = Report()
    for launch in range(a.launches):
        h.launch(512 << 20, timeout_s=60)
        for g in range(n):
            torch.cuda.synchronize(g)
        hs_k = torch.from_numpy(np.frombuffer(h.sk.read(0, "hs"), dtype=np.uint16).copy()).view(torch.bfloat16)[:T * SHR]
        hs_k = hs_k.view(T, SHR).to(ws[0]["x"].device).double()
        ref = hs_k @ ws[0]["wsd"].double().T
        got = ws[0]["S"].double()
        bad = [t for t in range(T) if not relrms(got[t], ref[t]) < 1e-5]
        rep.check(f"launch {launch}: S per token (from the kernel's h_s)", not bad,
                  f"relRMS {relrms(got, ref):.2e}; tokens off: {bad}")
    print("ALL CHECKS PASSED" if not rep.failed else f"FAILED: {rep.failed}", flush=True)


if __name__ == "__main__":
    main()
