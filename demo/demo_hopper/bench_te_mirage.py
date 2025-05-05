#!/usr/bin/env python3
"""
Benchmark comparing FP16/FP8 GQA throughput of NVIDIA TE vs. Mirage using matching input dimensions.
"""
import torch
import argparse
import transformer_engine.pytorch as te
from transformer_engine.pytorch import MultiheadAttention
from transformer_engine.common.recipe import DelayedScaling, Format
import mirage as mi


def measure_throughput(module, x, attention_mask=None, num_iters=100, fp8_autocast_kwargs=None):
    module.eval()
    with torch.no_grad():
        # Warm-up phase
        if fp8_autocast_kwargs:
            with te.fp8_autocast(**fp8_autocast_kwargs):
                for _ in range(10):
                    module(x, attention_mask)
        else:
            for _ in range(10):
                module(x, attention_mask)
        torch.cuda.synchronize()

        # Timed benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if fp8_autocast_kwargs:
            with te.fp8_autocast(**fp8_autocast_kwargs):
                for _ in range(num_iters):
                    module(x, attention_mask)
        else:
            for _ in range(num_iters):
                module(x, attention_mask)
        end.record()
        torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / num_iters
    seq_len, batch_size, hidden_size = x.shape
    tokens_per_s = seq_len * batch_size / (avg_ms / 1000.0)
    return avg_ms, tokens_per_s


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
    tokens_per_s = seq_len * batch_size / (avg_ms / 1000.0)
    return avg_ms, tokens_per_s


def main():
    parser = argparse.ArgumentParser(description="GQA benchmark: TE vs Mirage dims match")
    parser.add_argument('--bs', type=int, default=1,
                        help='Base batch size for Mirage (Mirage uses 2*bs)')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to benchmark')
    parser.add_argument('--fp8', action='store_true', help='Enable FP8 for TE')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    torch.backends.cudnn.benchmark = True

    # Mirage input dimensions (known to work)
    bs = args.bs
    total_bs = bs * 2      # Mirage uses 2*batch_size
    seq_len = 256          # sequence length
    embed = 64             # embedding dimension
    hid = 4096             # hidden dimension

    device = torch.device('cuda')
    dtype = torch.float16

    # FP8 recipe for TE
    fp8_kwargs = None
    if args.fp8:
        recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="max"
        )
        fp8_kwargs = {'enabled': True, 'fp8_recipe': recipe}

    # TE GQA (2-head attention on embed dims to mirror Mirage)
    num_heads = 2  # match Mirage's group count
    print("== TE GQA Benchmark ==")
    x_te = torch.rand(seq_len, total_bs, embed, device=device, dtype=dtype)
    te_attn = MultiheadAttention(embed, num_heads).to(device=device, dtype=dtype)
    lat_te, tp_te = measure_throughput(
        te_attn, x_te, None,
        num_iters=args.iters,
        fp8_autocast_kwargs=fp8_kwargs
    )
    print(f"TE GQA Avg latency: {lat_te:.3f} ms | Throughput: {tp_te:.1f} tokens/s")

    # Mirage GQA Benchmark
    print("\n== Mirage GQA Benchmark ==")
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(total_bs, seq_len, embed), dtype=mi.float16)
    K = graph.new_input(dims=(total_bs, embed, hid), dtype=mi.float16)
    V = graph.new_input(dims=(total_bs, hid, embed), dtype=mi.float16)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="attention", save_codes=True)
    if optimized_graph is None:
        raise RuntimeError(
            "Mirage did not discover a valid muGraph for these dimensions."
        )
    Q_t = torch.randn(total_bs, seq_len, embed, device=device, dtype=dtype)
    K_t = torch.randn(total_bs, embed, hid, device=device, dtype=dtype)
    V_t = torch.randn(total_bs, hid, embed, device=device, dtype=dtype)
    lat_mi, tp_mi = measure_mirage_throughput(
        optimized_graph,
        [Q_t, K_t, V_t],
        seq_len,
        total_bs,
        num_iters=args.iters
    )
    print(f"Mirage GQA Avg latency: {lat_mi:.3f} ms | Throughput: {tp_mi:.1f} tokens/s")

    # Summary
    print("\n== Summary ==")
    print(f"TE GQA: {tp_te:.1f} tokens/s vs Mirage GQA: {tp_mi:.1f} tokens/s")

if __name__ == '__main__':
    main()

