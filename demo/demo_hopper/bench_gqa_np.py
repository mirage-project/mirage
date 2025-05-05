#!/usr/bin/env python3
"""
Benchmark comparing PyTorch vs Mirage execution of a specific custom graph.
The graph structure and EXACT DIMENSIONS are taken from a working Mirage example.
The 'PyTorch' side uses standard PyTorch ops; the '--fp8' flag enables TE's FP8 context,
but standard PyTorch ops may not be accelerated by it.
"""

import torch
import argparse
import math
# No longer using TE's DotProductAttention directly for the benchmarked computation
from transformer_engine.common.recipe import DelayedScaling, Format
import transformer_engine.pytorch as te
import mirage as mi # Assuming mirage is installed and importable

# --- Helper Function for Mirage Benchmark (Unchanged) ---
def measure_mirage_throughput(optimized_graph, inputs, seq_len, batch_size, num_iters=100):
    # Warm-up phase
    for _ in range(10):
        optimized_graph(inputs=inputs)
    torch.cuda.synchronize()
    # Timed benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        optimized_graph(inputs=inputs)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / num_iters
    # Use the actual batch_size passed (which will be total_bs)
    tokens_per_s = seq_len * batch_size / (avg_ms / 1000.0)
    return avg_ms, tokens_per_s

# --- Helper Function for PyTorch Benchmark (Unchanged) ---
def measure_pytorch_throughput(pytorch_func, inputs, seq_len, batch_size, num_iters=100, fp8_autocast_kwargs=None):
    # Warm-up phase
    if fp8_autocast_kwargs:
        with te.fp8_autocast(**fp8_autocast_kwargs):
            for _ in range(10):
                _ = pytorch_func(*inputs) # Call the function
    else:
        # Use standard FP16 autocast for PyTorch baseline
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(10):
                 _ = pytorch_func(*inputs)
    torch.cuda.synchronize()

    # Timed benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if fp8_autocast_kwargs:
        with te.fp8_autocast(**fp8_autocast_kwargs):
            for _ in range(num_iters):
                 _ = pytorch_func(*inputs)
    else:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(num_iters):
                 _ = pytorch_func(*inputs)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / num_iters
    # Use the actual batch_size passed (which will be total_bs)
    tokens_per_s = seq_len * batch_size / (avg_ms / 1000.0)
    return avg_ms, tokens_per_s

# --- The specific computation graph from the working Mirage example ---
def mirage_custom_computation(Q, K, V):
    """
    Implements the graph: O = normalize(exp(Q @ K)) @ V
    using PyTorch operations.
    Shapes: Q(B, S, D1), K(B, D1, D2), V(B, D2, D1) -> O(B, S, D1)
    """
    A = torch.matmul(Q, K) # (B, S, D2)
    E = torch.exp(A)       # (B, S, D2)
    # Sum reduction along the last dimension (D2)
    S = torch.sum(E, dim=-1, keepdim=True) # (B, S, 1)
    # Add small epsilon for numerical stability before division
    D = torch.div(E, S + 1e-6) # (B, S, D2)
    O = torch.matmul(D, V) # (B, S, D1)
    return O

# ========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch vs Mirage Benchmark for Specific Custom Graph")
    parser.add_argument('--bs', type=int, default=1,
                        help='Base batch size (actual batch size used is 2*bs)')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to benchmark')
    parser.add_argument('--fp8', action='store_true',
                        help='Enable TE FP8 context for PyTorch version (may not accelerate std ops)')
    # REMOVED --seqlen, --dim1, --dim2 as fixed dimensions are used

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    torch.backends.cudnn.benchmark = True

    # --- Dimensions ---
    # Use EXACT dimensions from the working Mirage example
    base_bs = args.bs
    total_bs = 2 * base_bs # Use 2*bs like the working example
    fixed_s = 256
    fixed_d1 = 64
    fixed_d2 = 4096

    device = torch.device('cuda')
    dtype_pytorch = torch.float16 # PyTorch baseline runs in FP16
    dtype_mirage = mi.float16

    print("--- Configuration ---")
    print(f"Benchmarking Custom Graph: O = normalize(exp(Q @ K)) @ V")
    print(f"Using FIXED Dimensions from working Mirage example:")
    print(f"  Batch Size (B): {total_bs} (2 * {base_bs})")
    print(f"  Sequence Length (S): {fixed_s}")
    print(f"  Dimension 1 (D1): {fixed_d1}")
    print(f"  Dimension 2 (D2): {fixed_d2}")
    print(f"PyTorch FP8 Context Enabled: {args.fp8}")
    print("--------------------")

    # --- FP8 Setup for PyTorch Context ---
    fp8_kwargs = None
    if args.fp8:
        print("Note: Enabling FP8 context for PyTorch benchmark.")
        print("      Standard PyTorch ops inside context may not use FP8.")
        recipe = DelayedScaling(
            fp8_format=Format.E4M3,
            amax_history_len=16,
            amax_compute_algo="max"
        )
        fp8_kwargs = {'enabled': True, 'fp8_recipe': recipe}
    else:
         print("Note: Running PyTorch benchmark in FP16.")


    # ========================================================================
    # PyTorch Benchmark (Implementing the Mirage Custom Computation w/ Fixed Dims)
    # ========================================================================
    print("\n== PyTorch Custom Graph Benchmark ==")

    # Create PyTorch tensors with the fixed 3D shapes
    # Shapes: Q(total_bs, S, D1), K(total_bs, D1, D2), V(total_bs, D2, D1)
    Q_pt = torch.randn(total_bs, fixed_s, fixed_d1, device=device, dtype=dtype_pytorch)
    K_pt = torch.randn(total_bs, fixed_d1, fixed_d2, device=device, dtype=dtype_pytorch)
    V_pt = torch.randn(total_bs, fixed_d2, fixed_d1, device=device, dtype=dtype_pytorch)
    pytorch_inputs = [Q_pt, K_pt, V_pt]

    # Measure throughput
    lat_pt, tp_pt = measure_pytorch_throughput(
        mirage_custom_computation,
        pytorch_inputs,
        fixed_s,      # Use fixed sequence length
        total_bs,     # Use total batch size
        num_iters=args.iters,
        fp8_autocast_kwargs=fp8_kwargs
    )
    print(f"PyTorch Custom Graph Avg latency: {lat_pt:.3f} ms | Throughput: {tp_pt:.1f} tokens/s")


    # ========================================================================
    # Mirage Benchmark (Using the working custom graph definition w/ Fixed Dims)
    # ========================================================================
    print("\n== Mirage Custom Graph Benchmark ==")
    print("Note: Building Mirage graph using exact structure and dimensions from working example.")

    # --- Build Mirage Graph (matches working example exactly) ---
    graph = mi.new_kernel_graph()
    # Use the fixed dimensions
    Q = graph.new_input(dims=(total_bs, fixed_s, fixed_d1), dtype=dtype_mirage)
    K = graph.new_input(dims=(total_bs, fixed_d1, fixed_d2), dtype=dtype_mirage)
    V = graph.new_input(dims=(total_bs, fixed_d2, fixed_d1), dtype=dtype_mirage)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, dim=2) # Reduce along the D2 dimension (axis 2)
    D = graph.div(E, S) # Rely on broadcasting
    O = graph.matmul(D, V)
    graph.mark_output(O)
    # --- End Mirage Graph ---

    # --- Superoptimize the Graph ---
    print("Starting Mirage superoptimization (this may take time)...")
    optimized_graph = None
    try:
        optimized_graph = graph.superoptimize(config="attention")
        if optimized_graph is None:
            print("Mirage ERROR: Did not discover a valid muGraph. Cannot run Mirage benchmark.")
        else:
            print("Mirage superoptimization successful.")
    except Exception as e:
        print(f"Mirage ERROR: Superoptimization failed: {e}")
        optimized_graph = None

    # --- Mirage Benchmark ---
    lat_mi, tp_mi = -1.0, -1.0
    if optimized_graph:
        # Prepare runtime inputs matching graph input definitions (FP16)
        Q_t_mirage = torch.randn(total_bs, fixed_s, fixed_d1, device=device, dtype=torch.float16)
        K_t_mirage = torch.randn(total_bs, fixed_d1, fixed_d2, device=device, dtype=torch.float16)
        V_t_mirage = torch.randn(total_bs, fixed_d2, fixed_d1, device=device, dtype=torch.float16)
        inputs_mirage = [Q_t_mirage, K_t_mirage, V_t_mirage]

        try:
            lat_mi, tp_mi = measure_mirage_throughput(
                optimized_graph,
                inputs_mirage,
                fixed_s,      # Use fixed sequence length
                total_bs,     # Use total batch size
                num_iters=args.iters
            )
            print(f"Mirage Custom Graph Avg latency: {lat_mi:.3f} ms | Throughput: {tp_mi:.1f} tokens/s")
        except Exception as e:
            print(f"Mirage ERROR: Runtime execution failed: {e}")
            lat_mi, tp_mi = -1.0, -1.0
    else:
        print("Skipping Mirage runtime benchmark due to optimization failure.")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n== Summary ==")
    pytorch_precision = "FP16 (FP8 Context)" if args.fp8 else "FP16"
    print(f"PyTorch ({pytorch_precision}): {tp_pt:.1f} tokens/s ({lat_pt:.3f} ms)")
    if optimized_graph and lat_mi > 0:
        print(f"Mirage (FP16): {tp_mi:.1f} tokens/s ({lat_mi:.3f} ms)")
    elif not optimized_graph:
         print(f"Mirage (FP16): Failed (Optimization)")
    else:
         print(f"Mirage (FP16): Failed (Runtime)")
    print("\nNote: Both benchmarks execute the same custom computation graph with fixed dimensions.")




# --- Configuration ---
# Benchmarking Custom Graph: O = normalize(exp(Q @ K)) @ V
# Using FIXED Dimensions from working Mirage example:
#   Batch Size (B): 2 (2 * 1)
#   Sequence Length (S): 256
#   Dimension 1 (D1): 64
#   Dimension 2 (D2): 4096
# PyTorch FP8 Context Enabled: False
# --------------------
# Note: Running PyTorch benchmark in FP16.

# == PyTorch Custom Graph Benchmark ==
# PyTorch Custom Graph Avg latency: 0.107 ms | Throughput: 4788354.8 tokens/s

# == Mirage Custom Graph Benchmark ==
# Note: Building Mirage graph using exact structure and dimensions from working example.
# Starting Mirage superoptimization (this may take time)...
# ========== Search Configuration ==========
#   max num threadblock graph op: 9
#   max num kernel_graph op: 7
#   max num threadblock graphs: 1
#   max num threadblock graph inputs: 3
#   max num threadblock graph outputs: 2
#   search_thread: 8
#   imaps to explore:
#   imap combs to explore:
#   omaps to explore:
#   grid dims to explore:
#   block dims to explore:
#   fmaps to explore:
#   franges to explore:4 16 64
# [Search] States: 901, Random tests: 1, Valid mugraphs: 0, Time: 4.938996
# [Search] First step finished. Time elapsed: 4.939450sec
# [Search] States: 580401, Random tests: 1572, Valid mugraphs: 7, Time: 366.251295
# [Search] Second step finished. Time elapsed: 366.268407sec
# [Search] Total states explored: 580423
# [Search] Random tests performed: 1572
# [Serach] Valid kernel graphs explored: 7
# Transpiling discovered 7 muGraphs ...
# Warning: planned smem_size(147584) exceeds MAX_SMEM_SIZE(98304)
# required shared memory size 147584 exceed max shared memory size of current gpu arch 101376
# muGraph 0: profiled performance (ms) = 0.038133758544921875
# muGraph 1: profiled performance (ms) = 0.03801702499389648
# muGraph 2: profiled performance (ms) = 0.05713292694091797
# muGraph 3: profiled performance (ms) = 0.4138067321777344
# muGraph 4: profiled performance (ms) = 0.4166144104003906
# muGraph 5: skipping since its shared memory usage exceed limit
# muGraph 6: profiled performance (ms) = 0.04983814239501953
# Mirage superoptimization successful.
# Mirage Custom Graph Avg latency: 0.038 ms | Throughput: 13376472.5 tokens/s

# == Summary ==
# PyTorch (FP16): 4788354.8 tokens/s (0.107 ms)
# Mirage (FP16): 13376472.5 tokens/s (0.038 ms)

# Note: Both benchmarks execute the same custom computation graph with fixed dimensions.

