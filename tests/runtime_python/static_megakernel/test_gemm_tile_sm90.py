"""One static-megakernel GEMM layer on Hopper: out[8, N] = x[8, K] . W[N, K]^T (fp32), checked against torch and timed.
usage: test_gemm_tile_sm90.py N K [rows_per_task 128|64] [reps]"""
import os, statistics, sys
import torch
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel
from mirage.mpk.static_harness import Harness

N, K = int(sys.argv[1]), int(sys.argv[2])
ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 128
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
T = 8
torch.manual_seed(0)
x = (torch.randn(T, K, device="cuda") * 0.1).bfloat16()
w = (torch.randn(N, K, device="cuda") * 0.05).bfloat16()
out = torch.zeros(T, N, dtype=torch.float32, device="cuda")

mpk = StaticMegakernel(num_gpus=1)
mpk.gemm_tile_layer(input=mpk.attach_input(torch_tensor=x, name="x"),
                    weight=mpk.attach_input(torch_tensor=w, name="w"),
                    output=mpk.attach_input(torch_tensor=out, name="out"), combine="store")
node = next(iter(mpk._static_nodes))
grid = (N // ROWS, 1, 1)
nsm = mpk._num_sms()
lists = [[] for _ in range(nsm)]
for i in range(grid[0]):
    lists[i % nsm].append((node, (i, 0, 0)))
plan = {"grids": {node: grid}, "lists": [lists]}
out_dir = os.path.join(os.environ.get("OUT", "build_static"), f"gemm_{N}x{K}_{ROWS}")
paths, info = compiler.compile_plan(mpk, plan, os.path.join(out_dir, "schedules"))
sk = mpk.compile_static(paths, gpu_tensors=[{"x": x, "w": w, "out": out}], out_dir=os.path.join(out_dir, "build"))
h = Harness(sk)
h.launch(flush_bytes=0, timeout_s=10)
ref = x.float() @ w.float().T
err = (out - ref).abs().max().item()
rel = ((out - ref).norm() / ref.norm()).item()
print(f"CHECK N={N} K={K} rows={ROWS}: max abs err {err:.3e}, rel {rel:.3e}")
spans = []
for _ in range(REPS):
    h.launch(flush_bytes=256 << 20, timeout_s=10)
    spans.append(h.span_us())
bytes_w = N * K * 2
med = statistics.median(spans)
print(f"TIME N={N} K={K} rows={ROWS}: span median {med:.1f} us min {min(spans):.1f} us "
      f"-> weight stream {bytes_w / med / 1e3:.0f} GB/s")
sk.finalize()
