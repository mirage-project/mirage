"""fast_argmax_test.py: fast_argmax(x, -1) == torch.argmax(x, -1), bit for bit, on random and adversarial rows (ties across
chunks, -inf rows, +inf, NaN, -0.0 / 0.0, softcap-saturated logits, odd widths, strided rows), then both timed on the
sampler's [8, 129280]. One GPU (CUDA_VISIBLE_DEVICES). Exit 1 on any mismatch."""
import sys

import torch

import time

import fast_argmax as fa
from fast_argmax import fast_argmax

dev = torch.device("cuda", 0)
g = torch.Generator(device=dev).manual_seed(1)
fails, cases = 0, 0


def check(x, what):
    global fails, cases
    cases += 1
    a, b = torch.argmax(x, -1), fast_argmax(x, -1)
    if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
        fails += 1
        bad = (a != b).nonzero().flatten()[:4].tolist() if a.shape == b.shape else "shape"
        print(f"MISMATCH {what}: shape {tuple(x.shape)} rows {bad}: torch {a.flatten()[:4].tolist()} fast {b.flatten()[:4].tolist()}")


V = 129280


def suite():
    for seed in range(20):
        x = torch.randn(8, V, device=dev, generator=g)
        check(x, f"randn seed {seed}")
        cap = 30.0
        y = cap * torch.tanh(x * 40 / cap)            # softcap-saturated: many values equal to the cap after rounding
        check(y, f"softcap-saturated seed {seed}")
    for shape in [(1, V), (8, 32320), (8, V + 1), (64, V), (8, 4096), (8, 1), (8, 4097), (3, 8191), (8, 8193)]:
        check(torch.randn(*shape, device=dev, generator=g), f"randn {shape}")
    x = torch.randn(8, V, device=dev, generator=g)
    m = x.max().item() + 1.0
    for r, cols in enumerate([[5, 4096 + 5], [4095, 4096], [100, 70000, 129279], [0, 129279], [129279], [8191, 8192, 8193],
                              [64000, 63999], [3, 3]]):
        x[r, cols] = m
    check(x, "ties across chunks")
    x = torch.full((8, V), float("-inf"), device=dev)
    check(x, "all -inf")
    x = torch.randn(8, V, device=dev, generator=g)
    x[0, [7, 9000]] = float("inf")
    x[1, 130] = float("inf")
    x[2, :] = float("-inf")
    x[2, 77777] = -5.0
    check(x, "+inf / -inf")
    x = torch.randn(8, V, device=dev, generator=g)
    x[0, 12345] = float("nan")
    x[1, [100000, 50]] = float("nan")
    x[2, [4096, 4095]] = float("nan")
    x[3, 129279] = float("nan")
    x[4, 0] = float("nan")
    check(x, "NaN")
    x = torch.full((8, V), -1.0, device=dev)
    x[:, 10] = -0.0
    x[:, 9] = 0.0
    x[1, 9] = -0.0
    x[1, 10] = 0.0
    check(x, "-0.0 / 0.0")
    big = torch.randn(8, V + 512, device=dev, generator=g)
    check(big[:, 3:V + 3], "strided rows (a slice)")
    x = torch.randn(8, V, device=dev, generator=g).to(torch.bfloat16)
    check(x, "bf16 (falls back to torch.argmax)")



for mode in ("row", "two"):
    fa.MODE[0] = mode
    suite()
    print(f"mode {mode}: {cases} cases so far, {fails} mismatches", flush=True)
x = torch.randn(8, V, device=dev, generator=g)
from torch.profiler import ProfilerActivity, profile
for name, f in (("torch.argmax", lambda t: torch.argmax(t, -1)), ("fast row", None), ("fast two", None)):
    if name.startswith("fast"):
        fa.MODE[0] = name.split()[1]
        f = lambda t: fast_argmax(t, -1)
    for _ in range(30):
        f(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        f(x)
    host = (time.perf_counter() - t0) / 200 * 1e6
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(50):
            f(x)
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
    gpu = sum(e.device_time for e in ev) / 50
    print(f"{name}: GPU {gpu:.1f} us a call ({len(ev) // 50} kernels), host {host:.1f} us a call (enqueue, no sync), {torch.cuda.get_device_name(0)}", flush=True)
sys.exit(1 if fails else 0)
