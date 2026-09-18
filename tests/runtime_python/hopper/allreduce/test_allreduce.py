#!/usr/bin/env python3
"""
Unit test for nvshmem_tile_allreduce device function.

This test validates the NVSHMEM tile allreduce operation by:
1. Initializing MPI and NVSHMEM environments
2. Creating test tensors across multiple GPUs
3. Running the allreduce kernel
4. Comparing results against PyTorch reference
5. Benchmarking performance

Usage:
    # Run with mpirun for multi-GPU testing
    mpirun -np 2 python test_allreduce.py
    
    # Or with torchrun
    torchrun --nproc_per_node=2 test_allreduce.py
"""

import os
import sys
import torch
import numpy as np

# Try to import NVSHMEM - will fail gracefully if not available
try:
    import runtime_kernel_hopper_allreduce as kernel
    NVSHMEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import runtime_kernel_hopper_allreduce: {e}")
    print("Make sure to build with USE_NVSHMEM=1 and install NVSHMEM libraries")
    NVSHMEM_AVAILABLE = False

# MPI initialization
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    print("Warning: mpi4py not available. Single-GPU testing only.")
    MPI_AVAILABLE = False

# ==============================================================================
# Test Configuration
# ==============================================================================

class TestConfig:
    """Configuration for allreduce tests - easy to modify for different scenarios"""
    
    # Data dimensions (batch_size, hidden_size)
    BATCH_SIZES = [1, 2, 4, 8]
    OUTPUT_SIZES = [64, 128, 256, 512]  # Must match template instantiations
    OUTPUT_STRIDES = [64, 128, 256, 512, 1024, 2048, 4096]  # Hidden dimensions
    
    # Test tolerance
    RTOL = 1e-2
    ATOL = 1e-2
    
    # Performance settings
    WARMUP_RUNS = 16
    BENCHMARK_RUNS = 1000
    
    # Logging
    VERBOSE = True


# ==============================================================================
# NVSHMEM Initialization Helpers
# ==============================================================================

def init_mpi():
    """Initialize NVSHMEM using MPI for multi-GPU setup"""
    if not MPI_AVAILABLE:
        raise RuntimeError("MPI is required for NVSHMEM initialization")
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # Set CUDA device based on local rank
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

# ==============================================================================
# Test Functions
# ==============================================================================

def test_allreduce_single_config(batch_size, output_size, output_stride, rank, world_size, nvshmem_teams_ptr):
    """Test allreduce for a single configuration"""
    
    if TestConfig.VERBOSE and rank == 0:
        print(f"\n{'='*70}")
        print(f"Testing: batch_size={batch_size}, output_size={output_size}, output_stride={output_stride}")
        print(f"{'='*70}")
    
    # Create input tensor - different values on each rank
    # Use a generator with rank-specific seed for reproducibility
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    
    # Input shape: (batch_size, output_stride)
    # But we only use the first output_size columns for the allreduce
    input_tensor = torch.randn(
        (batch_size, output_stride), 
        device="cuda", 
        dtype=torch.bfloat16,
        generator=g
    )
    
    # Output tensor
    output_tensor = torch.empty(
        (batch_size, output_stride), 
        device="cuda", 
        dtype=torch.bfloat16
    )
    
    # For verification: gather all inputs to compute expected sum
    if MPI_AVAILABLE:
        # Collect all inputs for reference computation
        all_inputs = [torch.zeros_like(input_tensor) for _ in range(world_size)]
        comm = MPI.COMM_WORLD
        
        # Use MPI to gather (for verification only)
        # Convert to float for MPI communication
        input_cpu = input_tensor.float().cpu().numpy()
        all_inputs_cpu = comm.allgather(input_cpu)
        
        # Compute expected result: sum across all ranks
        expected_sum = sum(torch.from_numpy(arr).cuda().bfloat16() for arr in all_inputs_cpu)
    else:
        # Single GPU - expected sum is just the input
        expected_sum = input_tensor
    
    # Run the allreduce kernel
    task_offset = 0  # For testing, use task 0
    
    try:
        runtime_kernel_hopper_allreduce.allreduce(
            input_tensor,
            output_tensor,
            nvshmem_teams_ptr,
            task_offset
        )
    except Exception as e:
        print(f"Error running allreduce kernel: {e}")
        return False
    
    # Verify results - only check the first output_size columns
    output_slice = output_tensor[:, :output_size]
    expected_slice = expected_sum[:, :output_size]
    
    try:
        torch.testing.assert_close(
            output_slice,
            expected_slice,
            rtol=TestConfig.RTOL,
            atol=TestConfig.ATOL,
        )
        if TestConfig.VERBOSE and rank == 0:
            print(f"✓ Correctness test PASSED")
        return True
    except AssertionError as e:
        print(f"✗ Correctness test FAILED on rank {rank}")
        print(f"  Error: {e}")
        print(f"  Max diff: {(output_slice.float() - expected_slice.float()).abs().max().item()}")
        return False


def benchmark_allreduce(batch_size, output_size, output_stride, rank, world_size, nvshmem_teams_ptr):
    """Benchmark allreduce performance"""
    
    if not (TestConfig.VERBOSE and rank == 0):
        return
    
    # Create input/output tensors
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    input_tensor = torch.randn(
        (batch_size, output_stride), 
        device="cuda", 
        dtype=torch.bfloat16,
        generator=g
    )
    output_tensor = torch.empty_like(input_tensor)
    task_offset = 0
    
    # Warmup
    for _ in range(TestConfig.WARMUP_RUNS):
        runtime_kernel_hopper_allreduce.allreduce(
            input_tensor, output_tensor, nvshmem_teams_ptr, task_offset
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    starter.record()
    for _ in range(TestConfig.BENCHMARK_RUNS):
        runtime_kernel_hopper_allreduce.allreduce(
            input_tensor, output_tensor, nvshmem_teams_ptr, task_offset
        )
    ender.record()
    
    torch.cuda.synchronize()
    total_time = starter.elapsed_time(ender)
    avg_time = total_time / TestConfig.BENCHMARK_RUNS
    
    # Calculate bandwidth
    data_size_bytes = batch_size * output_size * 2  # bfloat16 = 2 bytes
    bandwidth_gb_s = (data_size_bytes / (avg_time / 1000)) / (1024**3)
    
    print(f"\n{'='*70}")
    print(f"Performance Results:")
    print(f"  Average time: {avg_time:.6f} ms")
    print(f"  Throughput: {bandwidth_gb_s:.2f} GB/s")
    print(f"  Data size per GPU: {data_size_bytes / (1024**2):.2f} MB")
    print(f"{'='*70}")


# ==============================================================================
# Main Test Runner
# ==============================================================================

def main():
    """Main test function"""
    
    if not NVSHMEM_AVAILABLE:
        print("ERROR: NVSHMEM kernel not available. Build with USE_NVSHMEM=1")
        sys.exit(1)
    
    # Initialize MPI and NVSHMEM
    rank, world_size, local_rank = init_mpi()
    
    # Create NVSHMEM teams
    print(f"\n{'='*70}")
    print(f"NVSHMEM Tile Allreduce Unit Test")
    print(f"{'='*70}")
    print(f"World size: {world_size}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"{'='*70}\n")
    
    # Run tests for different configurations
    all_passed = True

    kernel.init_nvshmem()
    input = torch.full((8, 4096), rank + 1, device="cuda", dtype=torch.bfloat16)
    output = torch.ones_like(input)

    print(input)
    print(output)

    kernel.allreduce(input, output, 0)

    expected_value = input * world_size
    try:
      torch.testing.assert_close(
          output,
          expected_value,
          rtol=TestConfig.RTOL,
          atol=TestConfig.ATOL,
      )
    except Exception as e:
      print(f"Correctness test FAILED on rank {rank}")
      all_passed = False
      print(output)
    
    # for batch_size in TestConfig.BATCH_SIZES:
    #     for output_size in TestConfig.OUTPUT_SIZES:
    #         for output_stride in TestConfig.OUTPUT_STRIDES:
    #             # Skip if output_size > output_stride (invalid)
    #             if output_size > output_stride:
    #                 continue
                
    #             passed = test_allreduce_single_config(
    #                 batch_size, output_size, output_stride,
    #                 rank, world_size, nvshmem_teams_ptr
    #             )
                
    #             if not passed:
    #                 all_passed = False
                
    #             # Benchmark a few selected configurations
    #             if rank == 0 and batch_size == 8 and output_size == 64:
    #                 benchmark_allreduce(
    #                     batch_size, output_size, output_stride,
    #                     rank, world_size, nvshmem_teams_ptr
    #                 )
    
    # Summary
    if rank == 0:
        print(f"\n{'='*70}")
        if all_passed:
            print("✓ All tests PASSED!")
        else:
            print("✗ Some tests FAILED!")
        print(f"{'='*70}\n")
    
    # Clean up (if needed)
    # TODO: Add NVSHMEM cleanup/finalization
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
