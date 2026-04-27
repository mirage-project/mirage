"""
Profile true GPU kernel time for linear_nvfp4_sm100_no_quantization vs torch._scaled_mm,
using nsys to strip host-side overhead (cudaMalloc, memcpy, synchronize, chunking loop).

Usage:
    python profile/profile_linear_1d2d_nvfp4_nsys.py [--m-values 1,8,32,128] [--n-values 1024] [--k-values 1024,4096]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import _runtime_path  # noqa: F401
import torch

DEVICE = "cuda"
DTYPE_OUT = torch.float32
DEFAULT_M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_N_VALUES = [128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 7168]
DEFAULT_K_VALUES = [256, 512, 768, 1024, 1536, 2048, 4096, 7168]
SUPPORTED_N_VALUES = set(DEFAULT_N_VALUES)
SUPPORTED_K_VALUES = set(DEFAULT_K_VALUES)
SMALL_M_MAX = 128
LARGE_SHAPE = (4096, 128, 768)

CUSTOM_KERNEL_RE   = re.compile(r"linear_nvfp4_swapAB_sm100_wrapper|linear_nvfp4_1d2d_sm100_wrapper")
BASELINE_KERNEL_RE = re.compile(r"nvjet_sm100")

NSYS_TMPDIR = os.path.expanduser("~/nsys_tmp")
os.makedirs(NSYS_TMPDIR, exist_ok=True)

WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "_nsys_worker.py")


def parse_int_list(value: str) -> list[int]:
    return [int(p.strip()) for p in value.split(",") if p.strip()]


def supported_shapes(m_values, n_values, k_values):
    for m in m_values:
        if 1 <= m <= SMALL_M_MAX:
            for n in n_values:
                for k in k_values:
                    if n in SUPPORTED_N_VALUES and k in SUPPORTED_K_VALUES:
                        yield m, n, k
        elif m == LARGE_SHAPE[0] and LARGE_SHAPE[1] in n_values and LARGE_SHAPE[2] in k_values:
            yield LARGE_SHAPE


def write_worker(warmup: int, reps: int) -> None:
    """Write the worker script that nsys will actually profile."""
    src = f"""\
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import _runtime_path  # noqa: F401
import torch
import runtime_kernel_blackwell_linear_nvfp4 as rt
from nvfp4_util import (
    make_random_nvfp4_tensors, interleave_sf_tensor,
    _to_blocked_sf, _pad_rows_uint8, encode_ue4m3,
)

UE4M3_ONE = encode_ue4m3(1.0)
M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
WARMUP, REPS = {warmup}, {reps}

x, weight, x_scale, weight_scale = make_random_nvfp4_tensors(M, N, K)
x_scale_il    = interleave_sf_tensor(x_scale)
weight_scale_il = interleave_sf_tensor(weight_scale)
output = torch.empty((M, N), device="cuda", dtype=torch.float32)

padded_rows = ((M + 127) // 128) * 128
x_pad  = _pad_rows_uint8(x,       padded_rows, fill_value=0)          if padded_rows != M else x
xs_pad = _pad_rows_uint8(x_scale, padded_rows, fill_value=UE4M3_ONE)  if padded_rows != M else x_scale
weight_fp4_t        = weight.view(torch.float4_e2m1fn_x2).transpose(0, 1)
x_fp4               = x_pad.view(torch.float4_e2m1fn_x2)
blocked_weight_scale = _to_blocked_sf(weight_scale)
blocked_x_scale      = _to_blocked_sf(xs_pad)

# Warmup both kernels
for _ in range(WARMUP):
    rt.linear_nvfp4_sm100_no_quantization(x, x_scale_il, weight, weight_scale_il, None, output)
    torch._scaled_mm(x_fp4, weight_fp4_t, blocked_x_scale, blocked_weight_scale,
                     bias=None, out_dtype=torch.float32)
torch.cuda.synchronize()

# Benchmark region captured by nsys
torch.cuda.nvtx.range_push("custom")
for _ in range(REPS):
    rt.linear_nvfp4_sm100_no_quantization(x, x_scale_il, weight, weight_scale_il, None, output)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("baseline")
for _ in range(REPS):
    torch._scaled_mm(x_fp4, weight_fp4_t, blocked_x_scale, blocked_weight_scale,
                     bias=None, out_dtype=torch.float32)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
"""
    with open(WORKER_SCRIPT, "w") as f:
        f.write(src)


def nsys_kernel_avg_ns(m: int, n: int, k: int) -> tuple[float | None, float | None]:
    """Run nsys on the worker for one shape, return (custom_avg_ns, baseline_avg_ns)."""
    rep_path = os.path.join(NSYS_TMPDIR, f"bench_{m}_{n}_{k}")
    python = sys.executable

    nsys_cmd = [
        "nsys", "profile",
        "--trace=cuda",
        f"--output={rep_path}",
        "--force-overwrite=true",
        python, WORKER_SCRIPT, str(m), str(n), str(k),
    ]
    env = os.environ.copy()
    env["TMPDIR"] = NSYS_TMPDIR

    result = subprocess.run(nsys_cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  nsys profile failed for M={m} N={n} K={k}:\n{result.stderr[:500]}", file=sys.stderr)
        return None, None

    stats_cmd = [
        "nsys", "stats",
        "--report", "cuda_gpu_kern_sum",
        "--force-export=true",
        f"{rep_path}.nsys-rep",
    ]
    result = subprocess.run(stats_cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  nsys stats failed for M={m} N={n} K={k}:\n{result.stderr[:500]}", file=sys.stderr)
        return None, None

    custom_ns_total, custom_count = 0.0, 0
    baseline_ns_total, baseline_count = 0.0, 0

    for line in result.stdout.splitlines():
        cols = line.split()
        # Table rows have at least: Time(%) TotalTime(ns) Instances Avg Med Min Max StdDev Name
        if len(cols) < 8:
            continue
        try:
            total_ns = float(cols[1])
            instances = int(cols[2])
        except ValueError:
            continue
        name = " ".join(cols[8:])
        if CUSTOM_KERNEL_RE.search(name):
            custom_ns_total += total_ns
            custom_count    += instances
        elif BASELINE_KERNEL_RE.search(name):
            baseline_ns_total += total_ns
            baseline_count    += instances

    custom_avg   = custom_ns_total   / custom_count   if custom_count   else None
    baseline_avg = baseline_ns_total / baseline_count if baseline_count else None
    return custom_avg, baseline_avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values",  type=parse_int_list, default=DEFAULT_M_VALUES)
    parser.add_argument("--n-values",  type=parse_int_list, default=DEFAULT_N_VALUES)
    parser.add_argument("--k-values",  type=parse_int_list, default=DEFAULT_K_VALUES)
    parser.add_argument("--warmup",    type=int, default=5)
    parser.add_argument("--reps",      type=int, default=10)
    args = parser.parse_args()

    shapes = list(dict.fromkeys(supported_shapes(args.m_values, args.n_values, args.k_values)))
    write_worker(args.warmup, args.reps)

    print(f"kernel_time_nvfp4 | shapes={len(shapes)} | warmup={args.warmup} | reps={args.reps} | units=us (GPU kernel only)")
    print(f"{'M':>5} {'N':>5} {'K':>5} | {'path':6} | {'custom_us':>10} | {'baseline_us':>11} | {'speedup':>7} | {'note'}")
    print("-" * 90)

    for m, n, k in shapes:
        path = "swapAB" if m <= SMALL_M_MAX else "1d2d"
        custom_ns, baseline_ns = nsys_kernel_avg_ns(m, n, k)

        if custom_ns is None or baseline_ns is None:
            print(f"M={m:4d} N={n:4d} K={k:4d} | {path:6} | FAILED")
            continue

        # For swapAB, the wrapper is called ceil(M/8) times per Python call.
        # nsys sees each chunk launch as a separate kernel instance.
        # custom_ns is already the per-launch average; total compute = custom_ns * chunks.
        chunks = (m + 7) // 8 if m <= SMALL_M_MAX else 1
        custom_total_us  = custom_ns * chunks / 1000.0
        baseline_us      = baseline_ns / 1000.0
        speedup          = baseline_us / custom_total_us

        note = f"({chunks} x {custom_ns/1000:.1f} us)" if chunks > 1 else ""
        print(
            f"M={m:4d} N={n:4d} K={k:4d} | {path:6} | "
            f"{custom_total_us:10.2f} | {baseline_us:11.2f} | {speedup:7.2f}x | {note}"
        )


if __name__ == "__main__":
    main()
