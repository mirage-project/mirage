"""Latency benchmark for the MPK FP8 swap-AB Linear kernel at DeepSeek V3
representative shapes.

The kernel is a single CTA on one SM. Each (BATCH, OUTPUT, K) entry below
maps to a layer the DeepSeek V3 demo (TP4 + Blackwell) fires per token.
Run after building the wrapper:

    cd tests/runtime_python/blackwell/sm100_linear_fp8_mpk
    python setup.py build_ext --inplace
    python bench_linear_fp8_mpk.py

DeepSeek V3 K dimensions used here (decode regime, per the public config):
    K = 1536   q_b family input (q_lora_rank)
    K = 4608   down input (intermediate_size_per_tp = 18432/4)
    K = 7168   q_a / kv_a / o_proj input (hidden_size)
    K = 16384  o_proj raw width (num_heads * v_head_dim = 128 * 128)

Per-task OUTPUT must be a multiple of 128 (tcgen05 MMA_M=128 hard
requirement). Real-world per-task N is the full N divided by grid_dim.x in
the MPK Python layer; values here represent typical post-split tile sizes.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import runtime_kernel_blackwell_linear_fp8_mpk as mod  # noqa: E402
from sm100_fp8_scale_layout import quantize_to_fp8_packed_ue8m0  # noqa: E402


# DeepSeek V3 layer-flavored shapes. (label, BATCH, OUTPUT, K) — every triple
# must appear in `mod.supported_shapes()`. Add new entries to the wrapper's
# DISPATCH_FOR_BATCH macro and rebuild before extending this list.
SHAPES = [
    # (label, BATCH, OUTPUT_PER_TASK, K)
    ("q_a (hidden→q_lora)",              1, 128, 7168),
    ("q_a (hidden→q_lora)",              4, 128, 7168),
    ("q_a (hidden→q_lora)",             16, 128, 7168),
    ("q_b (q_lora→head_dim)",            1, 128, 1536),
    ("q_b (q_lora→head_dim)",            4, 128, 1536),
    ("q_b (q_lora→head_dim)",           16, 128, 1536),
    ("q_b wider tile",                   1, 256, 1536),
    ("q_b wider tile",                  16, 256, 1536),
    ("q_b widest tile",                  1, 512, 1536),
    ("q_b widest tile",                 16, 512, 1536),
    ("o_proj (heads→hidden)",            1, 128, 16384),
    ("o_proj (heads→hidden)",            4, 128, 16384),
    ("o_proj (heads→hidden)",           16, 128, 16384),
    ("down (intermediate→hidden)",       1, 128, 4608),
    ("down (intermediate→hidden)",       4, 128, 4608),
    ("down (intermediate→hidden)",      16, 128, 4608),
    ("down wider tile",                  1, 256, 4608),
    ("down wider tile",                 16, 256, 4608),
]

WARMUP_REPS = 16
TIMED_REPS = 200


def quantize(x_bf16):
    x_q, scales = quantize_to_fp8_packed_ue8m0(x_bf16)
    return x_q.contiguous(), scales.contiguous()


def bench_one(label, batch, output_size, k, device):
    torch.manual_seed(0)
    input_bf16 = (torch.randn(batch, k, dtype=torch.bfloat16, device=device)
                  * 0.1).contiguous()
    weight_bf16 = (torch.randn(output_size, k, dtype=torch.bfloat16,
                               device=device) / (k ** 0.5)).contiguous()
    input_fp8, input_scale = quantize(input_bf16)
    weight_fp8, weight_scale = quantize(weight_bf16)
    output = torch.zeros(batch, output_size, dtype=torch.bfloat16,
                         device=device)

    # Quick correctness probe (optional; skip if you only want timing).
    mod.linear_fp8_mpk_sm100(input_fp8, input_scale, weight_fp8, weight_scale,
                             output)
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError(f"non-finite output for {label} "
                           f"B={batch} N={output_size} K={k}")

    # Warmup with repeat=WARMUP_REPS (single host call, descriptors built once).
    mod.linear_fp8_mpk_sm100(input_fp8, input_scale, weight_fp8, weight_scale,
                             output, repeat=WARMUP_REPS)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # Single host call running TIMED_REPS kernel launches back-to-back —
    # avoids cudaMalloc/cudaMemcpy/launch-overhead bias dominating timing.
    mod.linear_fp8_mpk_sm100(input_fp8, input_scale, weight_fp8, weight_scale,
                             output, repeat=TIMED_REPS)
    end.record()
    torch.cuda.synchronize()
    avg_us = (start.elapsed_time(end) / TIMED_REPS) * 1e3  # ms→µs

    # 1 CTA computes batch * output_size * k MACs = 2 * B * N * K FLOPs.
    flops = 2.0 * batch * output_size * k
    tflops = flops / (avg_us * 1e-6) / 1e12
    return avg_us, tflops


def main():
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    supported = set(tuple(s) for s in mod.supported_shapes())
    print(f"Wrapper has {len(supported)} pre-instantiated shapes; "
          f"benchmarking {len(SHAPES)} entries.")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Reps:   warmup={WARMUP_REPS}, timed={TIMED_REPS}")
    print()
    header = (f"{'label':<32} {'B':>3} {'N':>4} {'K':>6} "
              f"{'avg µs':>10} {'TFLOPS':>8}")
    print(header)
    print("-" * len(header))

    for label, batch, n, k in SHAPES:
        if (batch, n, k) not in supported:
            print(f"{label:<32} {batch:>3} {n:>4} {k:>6}   "
                  f"SKIP (not pre-instantiated; rebuild wrapper)")
            continue
        try:
            avg_us, tflops = bench_one(label, batch, n, k, device)
            print(f"{label:<32} {batch:>3} {n:>4} {k:>6} "
                  f"{avg_us:>10.2f} {tflops:>8.2f}")
        except Exception as e:
            print(f"{label:<32} {batch:>3} {n:>4} {k:>6}   ERROR: {e}")


if __name__ == "__main__":
    main()
