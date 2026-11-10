"""A GLM-5.3-Flash per-rank server library (amd/serving/glm/build_glm_libs.sh) in 4 processes, one GPU each (as SGLang runs
it), without SGLang: this rank's real weights of --layer from the hook's own loaders (amd/serving/glm/sglang_glm.py), random
run tensors in the server's shapes, --launches launches; checks: no fault, the outputs finite, the mHC outputs equal on every
rank; then e2e per launch (events, L2 not flushed).

  python -m torch.distributed.run --nproc-per-node 4 amd/tests/test_glm_rank_lib.py --kind gmoe --T 64 --layer 3
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "amd", "serving", "glm"))
os.environ.setdefault("MPK_GLM", "")
import sglang_glm as sg  # noqa: E402
from rank_lib import RankKernel  # noqa: E402

H, HC = 4096, 4
_S_EXTRA = []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=("gkda", "gdsa", "gmoe", "gdense"))
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--dlib", default="/mpk-amd/glm/dlib")
    p.add_argument("--model", default="/dev/shm/glm53f")
    p.add_argument("--launches", type=int, default=5)
    p.add_argument("--layers", default="", help="gmoe: instances of these layers (a range a-b or a list), run in turn, --launches rounds")
    p.add_argument("--lib", default="", help="the library (default: <dlib>/<kind>_t<T>/lib.so)")
    p.add_argument("--inputs", default="", help="a dump of the server's inputs (sglang_glm MPK_GLM_DUMP: r<rank>_t<T>_l<layer>_B.pt "
                   "with {rank} in the path) replacing the random ones")
    p.add_argument("--b2b", type=int, default=0, help="then this many launches back to back (no barrier, no sync between them; "
                   "an event pair per launch): the server's pattern (per-launch times printed as 5 %% / median / 95 %%)")
    p.add_argument("--graph", type=int, default=0, help="then a HIP graph of 10 launches (each instance in turn) replayed this "
                   "many times, an event pair per replay (the server's pattern: its decode step is a graph)")
    p.add_argument("--nan-x", action="store_true", help="NaN inputs (the graph capture's warm-up with uninitialised states)")
    a = p.parse_args()
    dist.init_process_group("nccl")
    rank, n = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    T = a.T
    sk = sg._Slices(a.model)
    k = RankKernel(a.lib or os.path.join(a.dlib, f"{a.kind}_t{T}", "lib.so"), dist.group.WORLD, dev)
    if a.kind == "gkda":
        w, hd = sg._kda_weights(a.model, a.layer, rank, n, dev)
        w.update(sg._mhc(sk, a.layer, "ffn", dev)[0])
    elif a.kind == "gdsa":
        w, nh = sg._dsa_weights(sk, a.layer, rank, n, dev)
        w.update(sg._mhc(sk, a.layer, "ffn", dev)[0])
    elif a.kind == "gmoe":
        w = sg._moe_weights(sk, a.layer, rank, n, dev)
        w.update(sg._mhc(sk, a.layer + 1, "attn", dev)[0])
        if a.layers:   # (the server's pattern: one library, an instance per layer, called in turn)
            lo_hi = a.layers.split("-")
            lst = list(range(int(lo_hi[0]), int(lo_hi[1]) + 1)) if len(lo_hi) == 2 else [int(v) for v in a.layers.split(",")]
            extra = []
            for L in lst:
                wl = sg._moe_weights(sk, L, rank, n, dev)
                wl.update(sg._mhc(sk, L + 1, "attn", dev)[0])
                extra.append((L, wl))
            _S_EXTRA[:] = extra
    else:
        w, F = sg._dense_weights(sk, a.layer, rank, n, dev)
        w.update(sg._mhc(sk, a.layer + 1, "attn", dev)[0])
    layer = k.create_layer(w)
    insts = [(L, k.create_layer(wl)) for L, wl in _S_EXTRA] or [(a.layer, layer)]
    g = torch.Generator(device=dev).manual_seed(5)   # (the same on every rank: the mHC inputs are replicated)
    rnd = lambda *sh, dt=torch.bfloat16: torch.randn(*sh, generator=g, device=dev).to(dt)
    io = {"R_in": rnd(T, HC * H), "h_post_in": torch.rand(T, HC, generator=g, device=dev) * 2,
          "h_res_in": torch.rand(T, HC * HC, generator=g, device=dev), "R_out": rnd(T, HC * H) * 0,
          "h_post_out": torch.zeros(T, HC, device=dev), "h_res_out": torch.zeros(T, HC * HC, device=dev),
          "x_attn": rnd(T, H), "x_ffn": rnd(T, H)}
    if a.inputs:
        d = torch.load(a.inputs.format(rank=rank))
        for kk, v in d.items():
            io[kk] = v.to(dev).contiguous()
        print(f"rank {rank}: inputs from {a.inputs.format(rank=rank)}: " + ", ".join(
            f"{kk} finite {bool(torch.isfinite(v.float()).all())} absmax {v.float().abs().max().item():.3g}" for kk, v in d.items()), flush=True)
    if a.nan_x:
        io["x_attn"].fill_(float("nan"))
        io["x_ffn"].fill_(float("nan"))
    slots = 3 * T + 2
    if a.kind == "gkda":
        io.update(state_pool=torch.randn(slots, 16, 128, 128, generator=g, device=dev) * 0.05,
                  conv_pool=rnd(slots, 3, 3 * hd), slots=torch.randperm(slots, generator=g, device=dev)[:T].int(),
                  kda_y=rnd(T, hd))
    elif a.kind == "gdsa":
        maxctx, reqs = sg.MAXCTX, T + 2
        rows = 1 + reqs * maxctx
        seq = torch.randint(64, 400, (T,), generator=g, device=dev).int()
        r2t = (torch.randperm(rows - 1, generator=g, device=dev)[:T * maxctx] + 1).view(T, maxctx).int()
        io.update(cache=(torch.randn(rows, 1, 512, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8),
                  out_loc=r2t[torch.arange(T, device=dev), (seq - 1).long()].long(), seq_lens=seq,
                  req_slots=torch.arange(T, device=dev).long(), pooled_bt=(r2t[:, ::256] // 256).contiguous(),
                  tail_k=rnd(reqs, 4, 128), tail_score=rnd(reqs, 4, 128),
                  pcache=torch.zeros(rows // 256 + 2, 8448, dtype=torch.uint8, device=dev), r2t=r2t,
                  req_idx=torch.arange(T, device=dev).long(), q_c=rnd(T, 1536), scores=torch.zeros(T, maxctx // 4, device=dev),
                  idx_q=rnd(T, 4096), pool_idx=torch.zeros(T, 512, dtype=torch.int32, device=dev), q_mla=rnd(T, nh * 512),
                  attn=rnd(T, nh * 512), v_out=rnd(T, nh * 256))
    elif a.kind == "gmoe":
        io.update(pairs=torch.zeros(T, 9, dtype=torch.int64, device=dev), R_moe=torch.zeros(T * 9, H, device=dev))
    else:
        io.update(h_mlp=rnd(T, F))
    st = torch.cuda.current_stream()
    us = []
    for i in range(a.launches):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        dist.barrier()
        e0.record()
        for L, inst in insts:
            inst.run(io, st)
            if a.layers:
                torch.cuda.synchronize()
                if rank == 0 and i == 0:
                    print(f"  layer {L} instance ran", flush=True)
        e1.record()
        torch.cuda.synchronize()
        us.append(e0.elapsed_time(e1) * 1e3)
        if rank == 0:
            print(f"launch {i} done ({us[-1]:.1f} us)", flush=True)
    if a.b2b:
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.b2b)]
        dist.barrier()
        torch.cuda.synchronize()
        for e0, e1 in evs:
            e0.record()
            for L, inst in insts:
                inst.run(io, st)
            e1.record()
        torch.cuda.synchronize()
        raw = [e0.elapsed_time(e1) * 1e3 for e0, e1 in evs]
        if a.b2b >= 2000 and rank == 0:   # (a drift over seconds: power / clocks)
            wn = a.b2b // 10
            print("rank 0 back to back, median per tenth: " + " ".join(
                f"{sorted(raw[i * wn:(i + 1) * wn])[wn // 2]:.1f}" for i in range(10)), flush=True)
        bu = sorted(raw)
        tot = evs[0][0].elapsed_time(evs[-1][1]) * 1e3
        print(f"rank {rank} back to back: {a.b2b} launches x {len(insts)} instances, per launch 5 % / median / 95 % "
              f"{bu[len(bu) // 20]:.1f} / {bu[len(bu) // 2]:.1f} / {bu[len(bu) * 19 // 20]:.1f} us; whole {tot:.0f} us "
              f"({tot / a.b2b:.1f} per launch)", flush=True)
    if a.graph:
        gs = torch.cuda.Stream()
        gs.wait_stream(torch.cuda.current_stream())
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.stream(gs):
            for L, inst in insts:   # (a warm-up run on the capture stream)
                inst.run(io, gs)
        torch.cuda.synchronize()
        with torch.cuda.graph(gr, stream=gs):
            for _ in range(10):
                for L, inst in insts:
                    inst.run(io, gs)
        torch.cuda.synchronize()
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.graph)]
        dist.barrier()
        for e0, e1 in evs:
            e0.record()
            gr.replay()
            e1.record()
        torch.cuda.synchronize()
        gu = sorted(e0.elapsed_time(e1) * 1e3 / (10 * len(insts)) for e0, e1 in evs)
        print(f"rank {rank} graph: {a.graph} replays of 10 x {len(insts)} launches, per launch 5 % / median / 95 % "
              f"{gu[len(gu) // 20]:.1f} / {gu[len(gu) // 2]:.1f} / {gu[len(gu) * 19 // 20]:.1f} us", flush=True)
    outs = {kk: io[kk] for kk in ("R_out", "h_post_out", "h_res_out", "x_ffn" if a.kind in ("gkda", "gdsa") else "x_attn")}
    fin = all(bool(torch.isfinite(v.float()).all()) for v in outs.values())
    gathered = {}
    for kk, v in outs.items():
        lst = [torch.zeros_like(v) for _ in range(n)]
        dist.all_gather(lst, v.contiguous())
        gathered[kk] = all(torch.equal(lst[0], x) for x in lst)
    us.sort()
    print(f"rank {rank} {a.kind} T {T} layer {a.layer}: outputs finite {fin}, equal on every rank {gathered}, e2e median "
          f"{us[len(us) // 2]:.1f} us (min {us[0]:.1f}; {a.launches} launches, eager, L2 not flushed)", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
