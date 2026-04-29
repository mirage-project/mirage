"""Latency benchmark for the split-K FP8 swap-AB Linear kernel.

The wrapper launches `split_k_factor` CTAs serially per call, so the
reported latency is roughly `split_k_factor` × per-CTA cost. In production
the MPK runtime launches them concurrently across SMs — comparing
split_k=1 vs higher values here shows the per-CTA K-walk reduction; the
real throughput improvement comes from the concurrent launch.

Run:
  python bench_linear_splitk_fp8_swapAB.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import runtime_kernel_blackwell_linear_splitk_fp8_swapAB as mod  # noqa: E402
from sm100_fp8_scale_layout import quantize_to_fp8_packed_ue8m0  # noqa: E402


# (label, B, OUT, K_per_task, split_k_factor)
SHAPES = [
    # o_proj-like (full_K=16384) at varying split factors
    ("o_proj split_k=1 (full K=16384)",  16, 128, 16384, 1),  # may not be in
                                                              # supported_shapes; SKIP if so
    ("o_proj split_k=2",                 16, 128, 8192,  2),
    ("o_proj split_k=4",                 16, 128, 4096,  4),
    # q_a-like (full_K=7168)
    ("q_a split_k=1 (full K=7168)",      16, 128, 7168,  1),
    ("q_a split_k=2",                    16, 128, 3584,  2),
    # down-like (full_K=4608) — full_K/512=9, valid splits ∈ {1, 3, 9}
    ("down split_k=1 (full K=4608)",     16, 128, 4608,  1),
    ("down split_k=3",                   16, 128, 1536,  3),
    # B=1 single-token decode
    ("o_proj split_k=4 @ B=1",            1, 128, 4096,  4),
]

WARMUP_REPS = 8
TIMED_REPS = 100


def _quantize(x_bf16):
    x_q, scales = quantize_to_fp8_packed_ue8m0(x_bf16)
    return x_q.contiguous(), scales.contiguous()


def bench(label, batch, out, k_per_task, split_k):
    full_k = k_per_task * split_k
    if (batch, out, k_per_task) not in set(tuple(s) for s in mod.supported_shapes()):
        return None

    torch.manual_seed(0)
    device = "cuda"
    input_bf16 = (torch.randn(batch, full_k, dtype=torch.bfloat16, device=device)
                  * 0.1).contiguous()
    weight_bf16 = (torch.randn(out, full_k, dtype=torch.bfloat16, device=device)
                   / (full_k ** 0.5)).contiguous()
    input_fp8, input_scale = _quantize(input_bf16)
    weight_fp8, weight_scale = _quantize(weight_bf16)
    output = torch.zeros(batch, out, dtype=torch.bfloat16, device=device)

    # Warmup
    mod.linear_splitk_fp8_swapAB_sm100(
        input_fp8, input_scale, weight_fp8, weight_scale, output,
        split_k, repeat=WARMUP_REPS)
    torch.cuda.synchronize()

    output.zero_()  # reset before timed loop

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    mod.linear_splitk_fp8_swapAB_sm100(
        input_fp8, input_scale, weight_fp8, weight_scale, output,
        split_k, repeat=TIMED_REPS)
    end.record()
    torch.cuda.synchronize()
    avg_us = (start.elapsed_time(end) / TIMED_REPS) * 1e3  # one full
                                                            # split_k_factor sweep
    per_cta_us = avg_us / split_k

    # 2 * B * N * full_K FLOPs total; split_k CTAs share the K work.
    flops = 2.0 * batch * out * full_k
    tflops_total = flops / (avg_us * 1e-6) / 1e12
    return avg_us, per_cta_us, tflops_total


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Reps:   warmup={WARMUP_REPS}, timed={TIMED_REPS}")
    print()
    header = (f"{'label':<38} {'B':>3} {'OUT':>4} {'K/t':>5} {'S':>3} "
              f"{'sweep µs':>10} {'per-CTA µs':>12} {'TFLOPS':>8}")
    print(header)
    print("-" * len(header))

    for label, b, n, k_per_task, s in SHAPES:
        result = bench(label, b, n, k_per_task, s)
        if result is None:
            print(f"{label:<38} {b:>3} {n:>4} {k_per_task:>5} {s:>3}   "
                  f"SKIP (shape not pre-instantiated)")
            continue
        avg_us, per_cta, tflops = result
        print(f"{label:<38} {b:>3} {n:>4} {k_per_task:>5} {s:>3} "
              f"{avg_us:>10.2f} {per_cta:>12.2f} {tflops:>8.2f}")


if __name__ == "__main__":
    main()
