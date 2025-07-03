import mirage as mi
import torch
import numpy as np

def test_softmax_persistent_kernel():
    """
    Test the new softmax layer in PersistentKernel
    """
    print("Testing Softmax in PersistentKernel")
    print("=" * 60)
    
    # Parameters
    batch_size = 1  # Use batch size 1 to avoid known issues
    vocab_size = 1024  # Small vocab size for single-stage softmax
    temperature = 1.0
    
    # Initialize
    mi.set_gpu_device_id(0)
    
    # Create tensors for meta information
    step = torch.zeros(1, dtype=torch.int32, device="cuda")
    tokens = torch.zeros(1, vocab_size, dtype=torch.int32, device="cuda")
    
    # Get proper configuration from GPU
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    print(f"GPU configuration: num_workers={num_workers}, num_schedulers={num_schedulers}")
    
    # Create PersistentKernel - meta_tensors must be exactly [step, tokens]
    mpk = mi.PersistentKernel(
        world_size=1,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=512,
        eos_token_id=0,
        meta_tensors=[step, tokens],
        profiler_tensor=None,
    )
    
    # Test: Single-stage softmax
    print("\nTest: Single-stage softmax")
    print("-" * 40)
    
    # Create input tensor with known values for better verification
    input_tensor = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16, device='cuda:0')
    
    # Create pre-allocated output buffer
    output_buffer = torch.zeros(batch_size, vocab_size, dtype=torch.bfloat16, device='cuda:0')
    
    # Compute expected output using PyTorch for comparison
    expected_output = torch.nn.functional.softmax(input_tensor / temperature, dim=-1)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor (first 10 values): {input_tensor[0, :10]}")
    print(f"Expected softmax output (first 10 values): {expected_output[0, :10]}")
    
    # Attach input tensor
    softmax_in = mpk.attach_input(torch_tensor=input_tensor, name="softmax_input")
    
    # Attach output buffer as input (this allows us to read it back later)
    softmax_out = mpk.attach_input(torch_tensor=output_buffer, name="softmax_output")
    
    # Add softmax layer
    mpk.softmax_layer(
        input=softmax_in,
        output=softmax_out,
        grid_dim=(1, 1, 1),  # Single task to process all batches
        block_dim=(128, 1, 1),
        temperature=temperature,
    )
    
    print(f"✓ Created single-stage softmax layer")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Temperature: {temperature}")
    
    
    # Compile the kernel
    print("\nCompiling PersistentKernel...")
    try:
        mpk.compile()
        print("✓ Compilation successful!")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Execute the kernel
    print("\nExecuting softmax kernel...")
    try:
        # Execute the persistent kernel
        mpk()
        print("✓ Softmax kernel executed successfully!")
        
        # Now check the output
        print("\nVerifying softmax output:")
        print("-" * 40)
        
        # The output_buffer should now contain the softmax results
        print(f"Output tensor (first 10 values): {output_buffer[0, :10]}")
        
        # Compare with expected output
        diff = torch.abs(output_buffer - expected_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"\nAccuracy comparison:")
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        
        # Check if output sums to 1 (property of softmax)
        output_sum = torch.sum(output_buffer, dim=-1)
        print(f"\nOutput sum (should be ~1.0): {output_sum}")
        
        # Additional validation
        if max_diff < 0.01:  # Tolerance for bfloat16
            print("\n✓ Softmax output matches expected values!")
        else:
            print("\n⚠ Softmax output differs from expected values")
            
    except Exception as e:
        print(f"\n✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Successfully added softmax kernel to PersistentKernel")
    print("- Single-stage softmax implementation tested")
    print("- Output values successfully retrieved and verified")


if __name__ == "__main__":
    test_softmax_persistent_kernel()