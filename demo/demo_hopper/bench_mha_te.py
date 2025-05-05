#!/usr/bin/env python3
"""
Benchmark comparing NVIDIA TE's DotProductAttention (MHA) vs. Mirage superoptimized MHA.
This script generates random inputs, runs TE's MHA and an equivalent Mirage graph, and reports throughput.
"""

import torch
import argparse
import math
from transformer_engine.pytorch import DotProductAttention
from transformer_engine.common.recipe import DelayedScaling, Format
import transformer_engine.pytorch as te
import mirage as mi


def measure_throughput(func, inputs, num_iters=100, fp8_autocast_kwargs=None):
    """
    Measures average latency (ms) of calling func(*inputs) over num_iters on CUDA.
    Works with both nn.Modules and standalone Python functions.
    """
    with torch.no_grad():
        # Warm-up
        if fp8_autocast_kwargs:
            with te.fp8_autocast(**fp8_autocast_kwargs):
                for _ in range(10):
                    func(*inputs)
        else:
            for _ in range(10):
                func(*inputs)
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if fp8_autocast_kwargs:
            with te.fp8_autocast(**fp8_autocast_kwargs):
                for _ in range(num_iters):
                    func(*inputs)
        else:
            for _ in range(num_iters):
                func(*inputs)
        end.record()
        torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return total_ms / num_iters


def measure_mirage_throughput(optimized_graph, inputs, seq_len, batch_size, num_iters=100):
    # Warm-up
    for _ in range(10):
        optimized_graph(inputs=inputs)
    torch.cuda.synchronize()
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        optimized_graph(inputs=inputs)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms / num_iters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare TE MHA vs Mirage MHA")
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    parser.add_argument('--embed', type=int, default=512, help='Embedding dimension (heads*head_dim)')
    parser.add_argument('--iters', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--fp8', action='store_true', help='Enable TE FP8')
    args = parser.parse_args()

    # Validate
    if args.embed % args.heads != 0:
        raise ValueError(f"embed ({args.embed}) must be divisible by heads ({args.heads})")

    # Setup
    batch_size = args.bs
    seq_len = args.seq_len
    num_heads = args.heads
    head_dim = args.embed // num_heads
    total_tokens = seq_len * batch_size
    device = torch.device('cuda')
    dtype = torch.float16

    # FP8 recipe for TE
    fp8_kwargs = None
    if args.fp8:
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        fp8_kwargs = {'enabled': True, 'fp8_recipe': recipe}

    print(f"Config: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}, TotalTok={total_tokens}")

    # ----------------------------
    # TE MHA Benchmark
    # ----------------------------
    print("\n== TE Multi-Head Attention ==")
    # Generate Q,K,V of shape [S, B, H, D]
    Q_te = torch.randn(seq_len, batch_size, num_heads, head_dim, device=device, dtype=dtype)
    K_te = torch.randn_like(Q_te)
    V_te = torch.randn_like(Q_te)
    # TE attention: emulate standard MHA via GQA with groups=num_heads
    te_attn_module = DotProductAttention(
        num_attention_heads=num_heads,
        kv_channels=head_dim,
        num_gqa_groups=num_heads
    ).to(device=device, dtype=dtype)

    def te_forward(q, k, v):
        return te_attn_module(query_layer=q, key_layer=k, value_layer=v)

    # Measure TE
    te_latency = measure_throughput(te_forward, [Q_te, K_te, V_te], num_iters=args.iters, fp8_autocast_kwargs=fp8_kwargs)
    te_throughput = total_tokens / (te_latency / 1000.0)
    print(f"TE MHA Avg latency: {te_latency:.3f} ms | Throughput: {te_throughput:.1f} tokens/s")

    # ----------------------------
    # Mirage MHA Benchmark
    # ----------------------------
    print("\n== Mirage Multi-Head Attention ==")
    # Host-side flattening for batched GEMM
    Q_flat = Q_te.permute(1,2,0,3).reshape(batch_size * num_heads, seq_len, head_dim)
    K_flat_T = K_te.permute(1,2,3,0).reshape(batch_size * num_heads, head_dim, seq_len)
    V_flat = V_te.permute(1,2,0,3).reshape(batch_size * num_heads, seq_len, head_dim)

    # Build Mirage MHA graph
    graph = mi.new_kernel_graph()
    Qg = graph.new_input(dims=(batch_size * num_heads, seq_len, head_dim), dtype=mi.float16)
    Kt_T = graph.new_input(dims=(batch_size * num_heads, head_dim, seq_len), dtype=mi.float16)
    Vg = graph.new_input(dims=(batch_size * num_heads, seq_len, head_dim), dtype=mi.float16)
    A = graph.matmul(Qg, Kt_T)
    E = graph.exp(A)
    S = graph.reduction(E, 2)     # sum over seq dim
    D = graph.div(E, S)
    O = graph.matmul(D, Vg)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="attention")
    if optimized_graph is None:
        raise RuntimeError("Mirage superoptimize failed for MHA graph.")

    # Measure Mirage
    mi_latency = measure_mirage_throughput(optimized_graph, [Q_flat, K_flat_T, V_flat], seq_len, batch_size, num_iters=args.iters)
    mi_throughput = total_tokens / (mi_latency / 1000.0)
    print(f"Mirage MHA Avg latency: {mi_latency:.3f} ms | Throughput: {mi_throughput:.1f} tokens/s")

    # ----------------------------
    # Summary
    # ----------------------------
    print("\n== Summary ==")
    print(f"TE MHA:     {te_throughput:.1f} tokens/s")
    print(f"Mirage MHA: {mi_throughput:.1f} tokens/s")


# RESULTS FROM G6E.XLARGE

# Config: B=1, S=128, H=8, D=64, TotalTok=128

# == TE Multi-Head Attention ==
# TE MHA Avg latency: 0.159 ms | Throughput: 803135.4 tokens/s

# == Mirage Multi-Head Attention ==
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
# [Search] States: 1001, Random tests: 4, Valid mugraphs: 0, Time: 4.836948
# [Search] First step finished. Time elapsed: 4.837241sec
# [Search] States: 494301, Random tests: 1732, Valid mugraphs: 5, Time: 374.949753
# [Search] Second step finished. Time elapsed: 374.957317sec
# [Search] Total states explored: 494321
# [Search] Random tests performed: 1732
# [Serach] Valid kernel graphs explored: 5
# Transpiling discovered 5 muGraphs ...
# muGraph 0: profiled performance (ms) = 0.03311513519287109
# muGraph 1: profiled performance (ms) = 0.03174195289611816
# muGraph 2: profiled performance (ms) = 0.031735807418823245
# muGraph 3: profiled performance (ms) = 0.0317706241607666
# muGraph 4: profiled performance (ms) = 0.02877952003479004
# Mirage MHA Avg latency: 0.030 ms | Throughput: 4288164.6 tokens/s

# == Summary ==
# TE MHA:     803135.4 tokens/s
# Mirage MHA: 4288164.6 tokens/s



# FP8 RUN
# Config: B=1, S=128, H=8, D=64, TotalTok=128

# == TE Multi-Head Attention ==
# TE MHA Avg latency: 0.167 ms | Throughput: 767814.7 tokens/s

# == Mirage Multi-Head Attention ==
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
# [Search] States: 1001, Random tests: 4, Valid mugraphs: 0, Time: 4.822042
# [Search] First step finished. Time elapsed: 4.822338sec
# [Search] States: 494301, Random tests: 1732, Valid mugraphs: 5, Time: 372.858656
# [Search] Second step finished. Time elapsed: 372.866003sec
# [Search] Total states explored: 494321
# [Search] Random tests performed: 1732
# [Serach] Valid kernel graphs explored: 5
# Transpiling discovered 5 muGraphs ...
# muGraph 0: profiled performance (ms) = 0.03126681518554687
# muGraph 1: profiled performance (ms) = 0.031094783782958983
# muGraph 2: profiled performance (ms) = 0.03166720008850098
# muGraph 3: profiled performance (ms) = 0.031059072494506835
# muGraph 4: profiled performance (ms) = 0.028919807434082033
# Mirage MHA Avg latency: 0.030 ms | Throughput: 4273367.3 tokens/s

# == Summary ==
# TE MHA:     767814.7 tokens/s
# Mirage MHA: 4273367.3 tokens/s

