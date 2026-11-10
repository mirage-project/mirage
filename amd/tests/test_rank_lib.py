"""A layer's per-rank server library (amd/tools/gen_rank_lib.py) in 8 processes, one GPU each (as SGLang runs it): the same
random weights and inputs as the single-process harness test (test_kda_layer.py / test_mla_layer.py, same seed), its outputs
compared bit for bit with that test's launch 1 (--save-outs), then timed: per launch an L2 flush, the ranks lined up on the GPU
(mpk_test_align), e2e = events right before and after the kernel (eager), and the same inside a HIP graph.

  python amd/tests/test_kda_layer.py --grids G --save-outs kda_ref.pt ...           (the harness, same plan)
  MK_BACKEND=hip python amd/tests/test_kda_layer.py --codegen --grids G --out kcg    (the layer.cu of that plan)
  python amd/tools/gen_rank_lib.py --layer-cu kcg/layer.cu --kind kda --out rank_kda
  python -m torch.distributed.run --nproc-per-node 8 amd/tests/test_rank_lib.py --kind kda --lib rank_kda/lib.so --ref kda_ref.pt [--kv-rows N (mla)] [--gemm-shuffled]
(the harness, the codegen and this test with the same --gemm-shuffled choice)
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "amd", "serving"))
from rank_lib import RankKernel  # noqa: E402

GEMM_W = {"kda": ("wproj", "oproj"), "mla": ("wa", "wqb", "wg", "wo"),
          "moe": ("router_weight", "latent_down_weight", "latent_up_weight", "shared_gate_up_weight", "shared_down_weight")}
OUTS = {"kda": ("hidden", "new_prefix", "ring_k", "ring_v", "ring_g", "ring_beta", "icw"), "mla": ("hidden", "new_prefix", "cache"),
        "moe": ("moe_out",)}
# the MoE layer ending with the next layer's aggregation 1 (test_layer_one_gpu.py --agg NB,WRITE,2)
GEMM_W["moeagg"], OUTS["moeagg"] = GEMM_W["moe"], ("moe_out", "agg_out")
# the attention part and its MoE in one kernel (test_kda_moe_layer.py --attn kda / mla): both weight sets, both outputs
for _a in ("kda", "mla"):
    GEMM_W[_a + "moe"] = GEMM_W[_a] + GEMM_W["moe"]
    OUTS[_a + "moe"] = OUTS[_a] + OUTS["moe"]
    GEMM_W[_a + "moeagg"] = GEMM_W[_a] + GEMM_W["moe"]   # (test_kda_moe_layer.py --agg NB,WRITE,2)
    OUTS[_a + "moeagg"] = OUTS[_a] + OUTS["moeagg"]
# a MoE layer + aggregation 1, then the next layer's attention part (test_kda_moe_layer.py --order moe)
for _a in ("kda", "mla"):
    GEMM_W["moeagg" + _a], OUTS["moeagg" + _a] = GEMM_W[_a] + GEMM_W["moe"], OUTS[_a] + OUTS["moeagg"] + ("agg_prefix", "agg_bare") + \
        (("agg_mix",) if os.environ.get("MPK_AGG_MIX") == "1" else ())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=("kda", "mla", "moe", "moeagg", "kdamoe", "mlamoe", "kdamoeagg", "mlamoeagg",
                                                       "moeaggkda", "moeaggmla"))
    p.add_argument("--token-share", type=float, default=0.6, help="moe: as test_layer_one_gpu.py")
    p.add_argument("--lib", required=True)
    p.add_argument("--ref", default=None, help="the harness test's --save-outs file")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--kv-rows", type=int, default=37)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--flush-mb", type=int, default=512)
    p.add_argument("--gemm-shuffled", action="store_true", help="as the harness run: the GEMM weights in gemm_tile's shuffled layout")
    p.add_argument("--chain", type=int, default=0, help="also a HIP graph of N launches back to back (no flush or line-up between "
                   "them, as the server's forward): time per launch = graph time / N")
    a = p.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=dev)
    group = dist.group.WORLD
    attn = a.kind[6:] if a.kind.startswith("moeagg") and len(a.kind) > 6 else a.kind[:3] if a.kind not in ("moe", "moeagg") else None
    has_moe = "moe" in a.kind
    if has_moe:   # (test_layer_one_gpu.py: the GEMM weights replaced by their shuffled copies in place)
        import test_layer_one_gpu as tm
        if a.kind not in ("moe", "moeagg"):
            os.environ["MPK_RMSNORM_FOLD"] = "1"
        d = tm.load_demo()
        wm0 = tm.random_weights(d, dev, a.seed)
        wm = wm0 if rank == 0 else tm.random_weights(d, dev, a.seed + rank, common=wm0)
    if a.kind in ("moe", "moeagg"):
        w = wm
    elif attn == "kda":
        import test_kda_layer as t
        w0 = t.random_weights(dev, a.seed)
        w = w0 if rank == 0 else t.random_weights(dev, a.seed + rank, common=w0)
    else:
        import test_mla_layer as t
        w0 = t.random_weights(dev, a.seed, a.kv_rows)
        w = w0 if rank == 0 else t.random_weights(dev, a.seed + rank, a.kv_rows, common=w0)
        w["wkc_k"] = w["wkc"].view(t.HEADS, t.NOPE, t.KV).transpose(1, 2).contiguous().view(t.HEADS * t.KV, t.NOPE)
        w["wvc_k"] = w["wvc"].view(t.HEADS, t.KV, t.VD).transpose(1, 2).contiguous().view(t.HEADS * t.VD, t.KV)
    if attn is not None and has_moe:
        w = {**w, **wm}
        if a.kind.startswith("moeagg"):   # (the next attention's prefix: moe_agg1's output in that kernel)
            w["agg_prefix"] = torch.empty(8, 7168, dtype=torch.bfloat16, device=dev)
            w["agg_bare"] = torch.empty(8, 7168, dtype=torch.bfloat16, device=dev)
            if os.environ.get("MPK_AGG_MIX") == "1":
                w["agg_mix"] = torch.empty(8, 7168, dtype=torch.bfloat16, device=dev)
    if a.gemm_shuffled:
        from mirage.mpk.static_megakernel import shuffle_gemm_weight
    for name in GEMM_W[a.kind]:
        key = name if name in GEMM_W["moe"] else name + "_k"
        w[key] = shuffle_gemm_weight(w[name]) if a.gemm_shuffled else w[name]
    torch.cuda.synchronize()
    k = RankKernel(a.lib, group, dev)
    layer = k.create_layer({n: w[n] for n in k.weight_names})
    # every rank's layer set up (its exchange region and counters initialised) before any rank's first launch: a rank that
    # launched early could otherwise write into a peer's exchange region before that peer's own set-up cleared it (the
    # check then hung at its first launch, 3 of 5 runs on 10-15)
    torch.cuda.synchronize()
    dist.barrier()
    io = {n: w[n] for n in k.run_names}
    ok = True

    def bits_vs_ref(tag):
        nonlocal ok
        if not a.ref:
            return
        ref = torch.load(a.ref)[rank]
        bad = [n for n in OUTS[a.kind] if not torch.equal(w[n].cpu().view(torch.uint8), ref[n].view(torch.uint8))]
        flags = [torch.zeros(1, device=dev)]
        flags[0][0] = len(bad)
        dist.all_reduce(flags[0])
        if rank == 0 or bad:
            print(f"rank {rank}: {tag}: outputs vs the harness's launch 1: "
                  f"{'bit-identical' if not bad else 'DIFFER: ' + ','.join(bad)} (ranks with a difference: {int(flags[0].item())})",
                  flush=True)
        ok = ok and flags[0].item() == 0

    for i in range(3):
        layer.run(io)
        torch.cuda.synchronize()
        bits_vs_ref(f"eager launch {i}")
    flush = torch.empty(a.flush_mb << 20, dtype=torch.uint8, device=dev)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    e2e = []
    for i in range(a.reps):
        flush.fill_(i & 0x7F)
        k.test_align()
        ev[0].record()
        layer.run(io)
        ev[1].record()
        torch.cuda.synchronize()
        e2e.append(ev[0].elapsed_time(ev[1]) * 1e3)
    bits_vs_ref("after the eager timed launches")
    # a HIP graph of one launch (the flush and the line-up outside it)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(st):
        layer.run(io, st)   # warm-up on the capture stream
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=st):
        layer.run(io, st)
    torch.cuda.synchronize()
    ge2e = []
    for i in range(a.reps):
        flush.fill_(i & 0x7F)
        k.test_align()
        ev[0].record()
        g.replay()
        ev[1].record()
        torch.cuda.synchronize()
        ge2e.append(ev[0].elapsed_time(ev[1]) * 1e3)
    bits_vs_ref("after the graph replays")
    per = 0.0
    if a.chain > 0:   # (outputs not checked after: each launch's state advances)
        gc = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gc, stream=st):
            for _ in range(a.chain):
                layer.run(io, st)
        torch.cuda.synchronize()
        ce = []
        for i in range(max(5, a.reps // 4)):
            flush.fill_(i & 0x7F)
            k.test_align()
            ev[0].record()
            gc.replay()
            ev[1].record()
            torch.cuda.synchronize()
            ce.append(ev[0].elapsed_time(ev[1]) * 1e3 / a.chain)
        per = sorted(ce)[len(ce) // 2]
    res = torch.tensor([sorted(e2e)[len(e2e) // 2], sorted(ge2e)[len(ge2e) // 2], layer.span_us(), per], device=dev)
    allr = [torch.zeros_like(res) for _ in range(world)]
    dist.all_gather(allr, res)
    if rank == 0:
        m = torch.stack(allr).cpu()
        print(f"LATENCY ({a.kind}, {world} processes, {a.reps} launches each, {a.flush_mb} MB flushed and ranks lined up before "
              f"each): e2e median per rank eager {[round(x, 1) for x in m[:, 0].tolist()]} us, HIP graph "
              f"{[round(x, 1) for x in m[:, 1].tolist()]} us; max over ranks eager {m[:, 0].max():.1f}, graph {m[:, 1].max():.1f}; "
              f"last span per rank {[round(x, 1) for x in m[:, 2].tolist()]}", flush=True)
        if a.chain > 0:
            print(f"CHAIN ({a.chain} launches in one graph): per launch per rank {[round(x, 1) for x in m[:, 3].tolist()]} us", flush=True)
        print("RANK LIB " + ("PASSED" if ok else "FAILED"), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
