"""Microbenchmark of one sm90_moe_ring node on Hopper at GPT-OSS's MoE shapes (decode, T = 8 tokens, 4 routed experts,
H = I = 2880, MXFP4 weights with fp32 scales). It measures per-task time, so a change to the ring can be timed and checked
for exact output equality before the full model runs.

    python moe_ring_bench.py --glu 1 --place spread --sms 48 [--xflags "-DFOO"] [--save out.pt] [--ref base.pt]

--glu 1: W13 (SwiGLU fused, 128-row tasks); 0: W2 (64-row tasks). --place one: --tasks tasks back to back on SM 0 (the rest
on the other SMs); spread: every task round-robin over --sms SMs (48 ~ the layer's second W13 wave, 132 the first). The L2
is flushed before each timed launch. Prints the median per-task time; --save / --ref compare the output bit for bit.
"""
import argparse, os, statistics, tempfile
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel, sm90_flags, sm90_moe_ring_pack
from mirage.mpk.static_harness import Harness

p = argparse.ArgumentParser()
p.add_argument("--glu", type=int, default=1)
p.add_argument("--place", choices=["one", "spread"], default="one")
p.add_argument("--tasks", type=int, default=8, help="place one: tasks on SM 0")
p.add_argument("--sms", type=int, default=132, help="place spread: SMs used")
p.add_argument("--xflags", default="", help="extra nvcc flags (A/B switches)")
p.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "moe_ring_bench"))
p.add_argument("--save", default=None, help="save the output tensor here")
p.add_argument("--ref", default=None, help="compare the output with this saved tensor")
args = p.parse_args()

T, A, E, H, I = 8, 4, 4, 2880, 2880
glu = args.glu == 1
R = 128 if glu else 64
N, K = (2 * I, H) if glu else (H, I)
torch.manual_seed(0)
packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
scales = torch.randint(118, 124, (E, N, K // 32), dtype=torch.uint8, device="cuda")
inp = (torch.randn(T, K, device="cuda") * 0.1).half() if glu else (torch.randn(A, T, K, device="cuda") * 0.1).half()
routing = torch.zeros(E, T, dtype=torch.int32, device="cuda")
routing[:, 0] = torch.arange(1, E + 1, dtype=torch.int32)   # token 0 to every expert
out = (torch.zeros(A, T, N // 2, dtype=torch.float16, device="cuda") if glu
       else torch.zeros(T, A, N, dtype=torch.bfloat16, device="cuda"))
bufs = {"inp": inp, "w": sm90_moe_ring_pack(packed, scales, R, glu=glu), "routing": routing,
        "active": torch.tensor(list(range(E)) + [E], dtype=torch.int32, device="cuda"), "out": out,
        "bias": torch.zeros(E, N, dtype=torch.bfloat16, device="cuda")}
mpk = StaticMegakernel(num_gpus=1)
D = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
mpk.sm90_moe_ring_layer(D["inp"], D["w"], D["routing"], D["active"], D["out"], bias=D["bias"], glu=glu,
                        rows_per_task=R, fp32_scales=True)
nsm = mpk._num_sms()
node = next(iter(compiler.graph_nodes(mpk)))
tasks = [(a, y, 0) for y in range(N // R) for a in range(A)]
lists = [[] for _ in range(nsm)]
if args.place == "one":
    for t in tasks[:args.tasks]:
        lists[0].append((node.graph_idx, t))
    for i, t in enumerate(tasks[args.tasks:]):
        lists[1 + i % (nsm - 1)].append((node.graph_idx, t))
else:
    for i, t in enumerate(tasks):
        lists[i % args.sms].append((node.graph_idx, t))
od = os.path.join(args.out, f"glu{int(glu)}_{args.place}_{args.tasks}_{args.sms}_{abs(hash(args.xflags)) % 10**8}")
paths, _ = compiler.compile_plan(mpk, {"grids": {node.graph_idx: (A, N // R, 1)}, "lists": [lists]}, od + "/schedules")
sk = mpk.compile_static(paths, gpu_tensors=[bufs], out_dir=od + "/build", extra_flags=sm90_flags() + args.xflags.split(),
                        profile=True)
h = Harness(sk)
h.launch(timeout_s=20)
durs, spans = [], []
for _ in range(10):
    h.launch(flush_bytes=256 << 20, timeout_s=20)
    spans.append(h.span_us())
    tt = h.task_times()[0]
    ent = tt[0] if args.place == "one" else [e for sm in tt for e in sm]
    durs += [(e[1] - e[0]) / 1e3 for e in ent if e[0]]
kb = R * ((K + 127) // 128) * 68 / 1e3   # a task's weight blocks (with their scales)
d, sp = statistics.median(durs), statistics.median(spans)
print(f"glu={int(glu)} place={args.place} tasks={args.tasks} sms={args.sms} {args.xflags}: task median {d:.2f} us "
      f"({kb:.0f} KB, {kb / d:.1f} GB/s a task); span {sp:.1f} us ({kb * len(tasks) / sp / 1e3:.2f} TB/s)")
if args.save:
    torch.save(out.cpu(), args.save)
if args.ref:
    print("output equal to the reference:", torch.equal(out.cpu(), torch.load(args.ref)))
sk.finalize()
