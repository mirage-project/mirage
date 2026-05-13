"""Per-layer kernel audit runner.

Drives the existing standalone tests under tests/runtime_python/blackwell/
in sequence. For each kernel:
  - Measures μs/call (median over `reps` runs)
  - Compares output to a PyTorch reference (cos sim)
  - Flags >3 μs gap to vLLM/cuBLAS target
  - Reports both perf and correctness in one table

Usage:
    cd /home/muhengl/mirage
    python scratch/per_layer_audit_runner.py [--layer NAME] [--reps N]

If --layer is omitted, runs all known layers.

Output: prints a markdown table to stdout AND appends to
scratch/per_layer_audit_results.md.

To add a new layer: append to LAYERS list with (name, build_fn, run_fn,
ref_fn, mpk_target_us, vllm_target_us, notes).
"""

import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path("/home/muhengl/mirage")
BENCH_ROOT = REPO / "tests" / "runtime_python" / "blackwell"
RESULTS_FILE = REPO / "scratch" / "per_layer_audit_results.md"
WARMUP = 16
REPS_DEFAULT = 200


# ============================================================
# Helper: build + import the wrapper .so for a given test dir
# ============================================================

def ensure_wrapper(test_dir: Path):
    """Build the wrapper if not already built; return the imported module."""
    sys.path.insert(0, str(test_dir))
    so_glob = list(test_dir.glob("runtime_kernel_*.cpython-*-linux-gnu.so"))
    if not so_glob:
        print(f"  building wrapper in {test_dir.name}...")
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(test_dir), stdout=subprocess.DEVNULL)
        so_glob = list(test_dir.glob("runtime_kernel_*.cpython-*-linux-gnu.so"))
    so_path = so_glob[0]
    mod_name = so_path.stem.split(".cpython")[0]
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def time_kernel_us(fn, warmup=WARMUP, reps=REPS_DEFAULT):
    """Median μs/call over `reps` runs using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]
    times_us.sort()
    return times_us[len(times_us) // 2]


def cos_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) /
            (a_f.norm() * b_f.norm() + 1e-12)).item()


# ============================================================
# Layer benches — one function per layer
# ============================================================

def bench_fp8_dense_smallm_qkv_a(reps):
    """qkv_a fused: M=128, N=2176, K=7168, num_workers=128."""
    test_dir = BENCH_ROOT / "sm100_fp8_gemm_dense"
    mod = ensure_wrapper(test_dir)
    sys.path.insert(0, str(BENCH_ROOT / "common"))
    from sm100_fp8_scale_layout import FP8_MAX  # noqa: E402

    M, N, K = 128, 2176, 7168
    g = torch.Generator(device="cuda").manual_seed(42)
    a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    b_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)

    def quantize_act(x):
        # Per-row, 1x128 group scale.
        x_f = x.float()
        groups = x_f.view(x.shape[0], x.shape[1] // 128, 128).abs().amax(dim=-1)
        scale = torch.clamp(groups / FP8_MAX, min=1e-10)
        scale_exp = scale.repeat_interleave(128, dim=1)
        q = (x_f / scale_exp).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q, scale

    def quantize_weight(w):
        # 128x128 block scale → sb shape (N/128, K/128).
        N, K = w.shape
        w_f = w.float().view(N // 128, 128, K // 128, 128)
        amax = w_f.abs().amax(dim=(1, 3))  # (N/128, K/128)
        scale = torch.clamp(amax / FP8_MAX, min=1e-10)
        scale_exp = scale.unsqueeze(1).unsqueeze(-1).expand(-1, 128, -1, 128).reshape(N, K)
        q = (w.float() / scale_exp).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q, scale

    a_fp8, sa = quantize_act(a_bf16)
    b_fp8, sb = quantize_weight(b_bf16)
    output = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)

    def run():
        mod.fp8_gemm_dense_smallm_multi_cta_launch(a_fp8, b_fp8, sa, sb, output, 128)

    # Correctness
    run()
    torch.cuda.synchronize()
    # Reference: dequant + matmul
    a_dq = a_fp8.float() * sa.repeat_interleave(128, dim=1)
    # b_dq: (N, K), sb is (N/128, K/128). Expand sb to (N, K).
    N_full, K_full = b_fp8.shape
    sb_exp = sb.unsqueeze(1).unsqueeze(-1).expand(-1, 128, -1, 128).reshape(N_full, K_full)
    b_dq = b_fp8.float() * sb_exp
    ref = (a_dq @ b_dq.t()).to(torch.bfloat16)
    cos = cos_sim(output, ref)

    us = time_kernel_us(run, reps=reps)
    return us, cos, "fp8_gemm_dense_smallm (qkv_a fused: M=128 N=2176 K=7168)"


def bench_fp8_dense_smallm_q_a(reps):
    """Baseline q_a: M=128, N=1536, K=7168, num_workers=128."""
    test_dir = BENCH_ROOT / "sm100_fp8_gemm_dense"
    mod = ensure_wrapper(test_dir)
    sys.path.insert(0, str(BENCH_ROOT / "common"))
    from sm100_fp8_scale_layout import FP8_MAX  # noqa: E402

    M, N, K = 128, 1536, 7168
    g = torch.Generator(device="cuda").manual_seed(42)
    a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    b_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)

    def quantize_act(x):
        # Per-row, 1x128 group scale.
        x_f = x.float()
        groups = x_f.view(x.shape[0], x.shape[1] // 128, 128).abs().amax(dim=-1)
        scale = torch.clamp(groups / FP8_MAX, min=1e-10)
        scale_exp = scale.repeat_interleave(128, dim=1)
        q = (x_f / scale_exp).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q, scale

    def quantize_weight(w):
        # 128x128 block scale → sb shape (N/128, K/128).
        N, K = w.shape
        w_f = w.float().view(N // 128, 128, K // 128, 128)
        amax = w_f.abs().amax(dim=(1, 3))  # (N/128, K/128)
        scale = torch.clamp(amax / FP8_MAX, min=1e-10)
        scale_exp = scale.unsqueeze(1).unsqueeze(-1).expand(-1, 128, -1, 128).reshape(N, K)
        q = (w.float() / scale_exp).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q, scale

    a_fp8, sa = quantize_act(a_bf16)
    b_fp8, sb = quantize_weight(b_bf16)
    output = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)

    def run():
        mod.fp8_gemm_dense_smallm_multi_cta_launch(a_fp8, b_fp8, sa, sb, output, 128)

    run()
    torch.cuda.synchronize()
    a_dq = a_fp8.float() * sa.repeat_interleave(128, dim=1)
    # b_dq: (N, K), sb is (N/128, K/128). Expand sb to (N, K).
    N_full, K_full = b_fp8.shape
    sb_exp = sb.unsqueeze(1).unsqueeze(-1).expand(-1, 128, -1, 128).reshape(N_full, K_full)
    b_dq = b_fp8.float() * sb_exp
    ref = (a_dq @ b_dq.t()).to(torch.bfloat16)
    cos = cos_sim(output, ref)
    us = time_kernel_us(run, reps=reps)
    return us, cos, "fp8_gemm_dense_smallm (baseline q_a: M=128 N=1536 K=7168)"


def bench_quantize_fp8_slice(reps):
    """Quantize a (128,1536) slice from a (128,2176) BF16 buffer with OUTPUT_STRIDE=1536."""
    test_dir = BENCH_ROOT / "sm100_fp8_gemm_dense"
    mod = ensure_wrapper(test_dir)
    g = torch.Generator(device="cuda").manual_seed(42)
    a_bf16 = torch.randn(128, 2176, device="cuda", dtype=torch.bfloat16, generator=g)
    out_fp8 = torch.zeros(128, 1536, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.zeros(128, 12, device="cuda", dtype=torch.float32)

    def run():
        mod.quantize_fp8_slice_launch(a_bf16, out_fp8, scale)

    run()
    torch.cuda.synchronize()
    # Reference (Python)
    a_slice = a_bf16[:, :1536].float()
    groups = a_slice.view(128, 12, 128).abs().amax(dim=-1)
    ref_scale = torch.clamp(groups / 448.0, min=1e-10)
    cos = cos_sim(scale, ref_scale)
    us = time_kernel_us(run, reps=reps)
    return us, cos, "quantize_fp8 slice (BATCH=128 HIDDEN=1536 GLOBAL=2176)"


def bench_splitk_swapAB_o_proj(reps):
    """o_proj decode: BATCH=16, OUT_PER_TASK=128, K=16384, SPLIT_K=8 (k_per_task=2048)."""
    test_dir = BENCH_ROOT / "sm100_linear_splitk_fp8_swapAB"
    mod = ensure_wrapper(test_dir)
    sys.path.insert(0, str(BENCH_ROOT / "common"))
    from sm100_fp8_scale_layout import quantize_to_fp8_packed_ue8m0  # noqa: E402

    B, OUT, K_per_task, SPLIT_K = 16, 128, 2048, 8
    K = K_per_task * SPLIT_K  # = 16384
    g = torch.Generator(device="cuda").manual_seed(42)
    a_bf16 = (torch.randn(B, K, dtype=torch.bfloat16, device="cuda",
                          generator=g) * 0.1).contiguous()
    w_bf16 = (torch.randn(OUT, K, dtype=torch.bfloat16, device="cuda",
                          generator=g) / (K ** 0.5)).contiguous()
    a_fp8, a_scale = quantize_to_fp8_packed_ue8m0(a_bf16)
    w_fp8, w_scale = quantize_to_fp8_packed_ue8m0(w_bf16)
    a_fp8 = a_fp8.contiguous(); a_scale = a_scale.contiguous()
    w_fp8 = w_fp8.contiguous(); w_scale = w_scale.contiguous()
    output = torch.zeros(B, OUT, dtype=torch.bfloat16, device="cuda")

    # Wrapper signature may vary — try the common one.
    if not hasattr(mod, "linear_splitk_fp8_swapAB_launch"):
        return float("nan"), 0.0, "(wrapper entry not found — skip)"

    def run():
        # Note: actual function signature TBD from wrapper inspection.
        # This is a placeholder; actual call needs the wrapper API.
        raise RuntimeError("SPLITK wrapper integration pending — see bench_linear_splitk_fp8_swapAB.py for the real call signature.")

    return float("nan"), 0.0, "(integration pending)"


# Registry of benches
LAYERS = [
    ("fp8_dense_smallm_q_a_baseline_n1536", bench_fp8_dense_smallm_q_a, None, 10),
    ("fp8_dense_smallm_qkv_a_fused_n2176", bench_fp8_dense_smallm_qkv_a, None, 13),
    ("quantize_fp8_slice_n1536_from_2176", bench_quantize_fp8_slice, None, None),
    ("splitk_swapAB_o_proj_decode (PLACEHOLDER)", bench_splitk_swapAB_o_proj, None, 15),
]


def append_to_results(rows):
    """Append a markdown table block to scratch/per_layer_audit_results.md."""
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n## Run {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n| layer | μs/call | cos | vLLM target μs | gap μs | shape |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, us, cos, vllm_us, label in rows:
            gap = "" if vllm_us is None else f"{us - vllm_us:+.2f}"
            vllm_str = "" if vllm_us is None else f"{vllm_us:.1f}"
            f.write(f"| {name} | {us:.2f} | {cos:.4f} | {vllm_str} | {gap} | {label} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default=None,
                    help="Run only one layer by substring match")
    ap.add_argument("--reps", type=int, default=REPS_DEFAULT)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(1)

    rows = []
    for name, bench_fn, _, vllm_us in LAYERS:
        if args.layer and args.layer not in name:
            continue
        print(f"\n=== {name} ===")
        try:
            us, cos, label = bench_fn(args.reps)
            print(f"  μs/call = {us:.2f}  cos = {cos:.4f}  shape = {label}")
            if vllm_us is not None:
                gap = us - vllm_us
                flag = " ⚠ >3μs gap to vLLM" if gap > 3.0 else ""
                print(f"  vs vLLM target {vllm_us:.1f} μs:  gap = {gap:+.2f}{flag}")
            rows.append((name, us, cos, vllm_us, label))
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append((name, -1.0, 0.0, vllm_us, f"ERROR: {e}"))

    print("\n=== Summary ===")
    print(f"{'layer':<45s} {'μs/call':>8s} {'cos':>6s} {'gap':>8s}")
    for name, us, cos, vllm_us, label in rows:
        gap = "" if vllm_us is None or us < 0 else f"{us - vllm_us:+.2f}"
        print(f"{name:<45s} {us:>8.2f} {cos:>6.4f} {gap:>8s}")

    append_to_results(rows)
    print(f"\nResults appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
