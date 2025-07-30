import mirage as mi
import torch

types = torch.bfloat16


def test_multi_token_linear():
    """
    Test multi_token_linear by comparing with PyTorch's linear operation
    """
    print("Testing Multi-Token Linear")
    print(f"Using type: {types}")
    print("=" * 60)

    # Initialize
    mi.set_gpu_device_id(0)

    # Get proper configuration from GPU
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    print(f"GPU configuration: num_workers={num_workers}, num_schedulers={num_schedulers}")

    # Test configurations
    test_cases = [
        (4, 512, 384),   # (num_tokens, reduction_size, output_size)
        (8, 1024, 768),
        (16, 2048, 1536),
    ]
    
    for num_tokens, reduction_size, output_size in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing with num_tokens={num_tokens}, reduction_size={reduction_size}, output_size={output_size}")
        print(f"{'-'*40}")

        # Create input tensors
        input_data = torch.randn(1, num_tokens * reduction_size, dtype=types, device="cuda")
        weight = torch.randn(reduction_size, output_size, dtype=types, device="cuda")
        residual = torch.randn(1, num_tokens * output_size, dtype=types, device="cuda")
        
        # PyTorch reference - apply softmax first to match MPK pipeline
        print("\n1. PyTorch reference (with dummy softmax):")
        # Apply softmax to match dummy operation
        input_after_softmax = torch.nn.functional.softmax(input_data, dim=-1)
        # Now do linear operation
        input_reshaped = input_after_softmax.view(num_tokens, reduction_size)
        output_ref = torch.matmul(input_reshaped, weight) + residual.view(num_tokens, output_size)
        output_ref = output_ref.view(1, -1)
        print(f"   Output shape: {output_ref.shape}")

        # MPK multi-token linear
        print("\n2. MPK multi_token_linear:")
        
        # Create output buffer
        output_buffer = torch.zeros(1, num_tokens * output_size, dtype=types, device='cuda:0')

        # Create PersistentKernel
        mpk = mi.PersistentKernel(
            world_size=1,
            mpi_rank=0,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=512,
            eos_token_id=128001,
            meta_tensors=[torch.zeros(1, dtype=torch.int32, device="cuda"), 
                         torch.zeros(512, dtype=torch.int64, device="cuda")],
            profiler_tensor=None,
        )
        
        # Attach tensors
        input_tensor = mpk.attach_input(input_data, "input")
        weight_tensor = mpk.attach_input(weight, "weight")
        residual_tensor = mpk.attach_input(residual, "residual")
        output_tensor = mpk.attach_input(output_buffer, "output")
        
        # Create a dummy intermediate tensor for chaining operations
        dummy_output = mpk.new_tensor(
            dims=(1, num_tokens * reduction_size),
            dtype=mi.bfloat16,
            name="dummy_output",
            io_category="cuda_tensor"
        )
        
        # First operation: dummy softmax to satisfy single-block constraint
        mpk.softmax_layer(
            input=input_tensor,
            output=dummy_output,
            grid_dim=(1, 1, 1),
            block_dim=(512, 1, 1),
            temperature=1.0
        )
        
        # Multi-token linear operation - use dummy_output as input to chain operations
        mpk.multi_token_linear_layer(
            input=dummy_output,  # Use dummy output to chain operations
            weight=weight_tensor,
            residual=residual_tensor,
            output=output_tensor,
            grid_dim=(num_tokens, 1, 1),
            block_dim=(256, 1, 1),  # Match kernel's THREADS_PER_BLOCK
            max_tokens=64
        )
        
        # Compile and execute
        mpk.compile()
        mpk()
        
        print(f"   Output shape: {output_buffer.shape}")

        # Compare outputs
        print("\n3. Comparison:")
        max_diff = torch.max(torch.abs(output_buffer - output_ref)).item()
        avg_diff = torch.mean(torch.abs(output_buffer - output_ref)).item()
        rel_error = max_diff / torch.max(torch.abs(output_ref)).item()

        print(f"   Maximum difference: {max_diff}")
        print(f"   Average difference: {avg_diff}")
        print(f"   Relative error: {rel_error}")

        # Check if results match within tolerance
        tolerance = 1e-2  # BFloat16 precision
        if max_diff < tolerance:
            print(f"   ✓ PASSED")
        else:
            print(f"   ✗ FAILED")
            # Print a few values for debugging
            print("\n   First few values:")
            print(f"   MPK:       {output_buffer[0, :5]}")
            print(f"   Reference: {output_ref[0, :5]}")

    print(f"\n{'='*60}")
    print("All tests completed!")


if __name__ == "__main__":
    test_multi_token_linear()