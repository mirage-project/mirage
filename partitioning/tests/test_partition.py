#!/usr/bin/env python3
"""
Test HybridModel: Mirage-optimized kernels + PyTorch fallback
+ Benchmark: average latency & speedup vs. PyTorch
"""
import argparse
import cProfile, pstats
import time
import torch
import sys
from pathlib import Path
torch.set_num_threads(1)        # intra-op CPU threads
torch.set_num_interop_threads(1)  # inter-op CPU threads

import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime import HybridModel, partition_graph_with_dp
from testing_models import TestMLP, TestTransformer

# for test_mlp
BATCH_SIZE_MLP = 32
INPUT_DIM_MLP = 1024

# for test_transformer
BATCH_SIZE_TRANSFORMER = 1
SEQ_LEN = 1024
VOCAB_SIZE = 16384

NUM_BENCH_ITERS = 100
NUM_BENCH_WARMUP = 10

IGNORE_OPS = set()
UNSUPPORTED_OPS = {"ReduceSum", "Constant", "Identity", "Unsqueeze", "Abs", "Gemm", "Expand", "Gather", "Reshape", "Transpose", "Cast", "CastLike", "Tanh"}

model_name_to_class = {
    "test-mlp": TestMLP,
    "test-transformer": TestTransformer,
}

def _get_input_tensor(model_name) -> torch.Tensor:
    if model_name == "test-mlp":
        return torch.randn(BATCH_SIZE_MLP, INPUT_DIM_MLP, device=torch.device("cuda"), dtype=torch.float16)
    elif model_name == "test-transformer":
        # Return int64 token IDs (correct for transformers)
        return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE_TRANSFORMER, SEQ_LEN), device=torch.device("cuda"), dtype=torch.int64)

def _benchmark_model(model: nn.Module | HybridModel,
                     x: torch.Tensor,
                     iters: int = 100,
                     warmup: int = 10,
                     profile_name: str = "benchmark") -> float:
    """
    Measure average forward-pass latency (milliseconds) for `model(x)`.

    - Uses CUDA events when CUDA is available (most accurate on GPU).
    - Falls back to time.perf_counter on CPU.
    - Puts model in eval/no-grad mode and runs warmups before timing.
    """
    assert x.is_cuda and torch.cuda.is_available(), "Input tensor must be on CUDA device for benchmarking."
    if isinstance(model, nn.Module):
        model.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(max(0, warmup)):
            _ = model(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for it in range(iters):
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        # elapsed_time returns milliseconds
        total_ms = start.elapsed_time(end)
        avg_ms = total_ms / max(1, iters)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_stack=True,
                        profile_memory=True) as prof:
            with record_function(profile_name):
                _ = model(x)
        print(f"\n=== Profiling Result for '{profile_name}' ===")
        ps = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        print(ps)
        print("=== End of Profiling Result ===\n")

    return avg_ms

def test_hybrid_model(dry_run: bool = True, cost_model: str = "gnn-xgboost", max_nodes_per_partition: int = 4, max_mirage_ops: int = 9, test_model_name: str = "test-transformer"):
    assert torch.cuda.is_available(), "HybridModel test requires a CUDA-capable GPU."
    # Slightly improve backend perf variability on GPU
    torch.backends.cudnn.benchmark = True

    model_class = model_name_to_class[test_model_name]
    model = model_class().to(torch.device("cuda"))
    dummy_input = _get_input_tensor(test_model_name)

    print(f"\n{'='*60}")
    print(f"Testing HybridModel (dry_run={dry_run}, cost_model='{cost_model}')")
    print(f"{'='*60}\n")

    hybrid_model = partition_graph_with_dp(
        model,
        dummy_input,
        IGNORE_OPS=IGNORE_OPS,
        UNSUPPORTED_OPS=UNSUPPORTED_OPS,
        cost_model=cost_model,
        max_nodes_per_partition=max_nodes_per_partition,
        max_mirage_ops=max_mirage_ops,
        dry_run=dry_run
    )

    # Test execution and numerical correctness
    print(f"\n{'='*60}")
    print("Testing Execution & Numerical Correctness")
    print(f"{'='*60}\n")

    test_input = _get_input_tensor(test_model_name)

    model = model.to(torch.device("cuda"))
    try:
        # Run both models
        with torch.no_grad():
            original_output = model(test_input)
        hybrid_output = hybrid_model(test_input, debug=False)

        # Compare results
        max_diff = torch.max(torch.abs(hybrid_output - original_output)).item()
        mean_diff = torch.mean(torch.abs(hybrid_output - original_output)).item()
        rel_error = mean_diff / (torch.mean(torch.abs(original_output)).item() + 1e-8)

        print(f"✓ Execution successful!")
        print(f"  Input shape:  {test_input.shape}")
        print(f"  Output shape: {hybrid_output.shape}")
        print(f"\n📊 Numerical Correctness:")
        print(f"  Max absolute diff:  {max_diff:.2e}")
        print(f"  Mean absolute diff: {mean_diff:.2e}")
        print(f"  Relative error:     {rel_error:.2e}")

        tol = 0.15
        if max_diff < tol:
            print(f"\n  ✅ PASSED: Outputs match within tolerance (tol={tol})!")
        else:
            print(f"\n  ❌ FAILED: Differences too large")
    except Exception as e:
        print(f"✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return  # Skip benchmarking if execution failed

    # ---------------------------------------------------------------------
    # Benchmark: average latency & speedup
    # ---------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Benchmark: Average Latency & Speedup")
    print(f"{'='*60}\n")

    bench_input = _get_input_tensor(test_model_name)

    iters = NUM_BENCH_ITERS
    warmup = NUM_BENCH_WARMUP

    try:
        pytorch_ms = _benchmark_model(model, bench_input, iters=iters, warmup=warmup, profile_name="pytorch")
        hybrid_ms = _benchmark_model(hybrid_model, bench_input, iters=iters, warmup=warmup, profile_name="hybrid")

        speedup = pytorch_ms / hybrid_ms if hybrid_ms > 0 else float("inf")

        print(f"Device: CUDA (fixed)")
        print(f"Input:  {tuple(bench_input.shape)}")
        print(f"Iterations: {iters} (warmup: {warmup})\n")

        print(f"PyTorch avg latency: {pytorch_ms:.3f} ms")
        print(f"Hybrid   avg latency: {hybrid_ms:.3f} ms")

        if speedup >= 1.0:
            print(f"\n🚀 Speedup (Hybrid over PyTorch): {speedup:.2f}× faster")
        else:
            print(f"\n🐢 Slowdown (Hybrid vs PyTorch): {1/speedup:.2f}× slower")

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HybridModel and benchmark speedup.")
    parser.add_argument(
        "--cost-model",
        choices=["gnn-xgboost", "dnn-abacus"],
        required=True,
        help="Which cost model to use for partitioning."
    )
    parser.add_argument(
        "--test-model-name",
        type=str,
        required=True,
        choices=["test-mlp", "test-transformer"],
        help="Name of the model to use for testing: 'test-mlp' or 'test-transformer'."
    )
    parser.add_argument(
        "--max-nodes-per-partition",
        type=int,
        default=4,
        help="Maximum number of nodes per partition for in-context partitioning (if cost_model='in_ctx')."
    )
    parser.add_argument(
        "--max-mirage-ops",
        type=int,
        default=9,
        help="Maximum number of kernel ops for Mirage superoptimization."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do a dry run (skip compiling/allocating heavy kernels if your partitioner supports it)."
    )
    args = parser.parse_args()

    test_hybrid_model(dry_run=args.dry_run, cost_model=args.cost_model, test_model_name=args.test_model_name,
                      max_nodes_per_partition=args.max_nodes_per_partition, max_mirage_ops=args.max_mirage_ops)
