import mirage as mi
import torch
import torch.nn.functional as F

types = torch.bfloat16


def test_multi_token_softmax_vs_pytorch():
    """
    Test multi_token_softmax by comparing with PyTorch's F.softmax
    Using LLaMA 3.1 8B settings
    """
    print("Testing Multi-Token Softmax vs PyTorch")
    print(f"Using type: {types}")
    print("=" * 60)

    # LLaMA 3.1 8B parameters
    vocab_size = 128256  # LLaMA 3.1 vocabulary size

    # Initialize
    mi.set_gpu_device_id(0)

    # Get proper configuration from GPU
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    print(f"GPU configuration: num_workers={num_workers}, num_schedulers={num_schedulers}")

    print(f"\nLLaMA 3.1 8B parameters:")
    print(f"  vocab_size: {vocab_size}")

    # Test different token counts and temperatures
    test_cases = [1, 4, 8, 16, 32]
    temperatures = [1.0, 0.5, 2.0]
    
    for temperature in temperatures:
        print(f"\n{'='*80}")
        print(f"Testing with temperature={temperature}")
        print(f"{'='*80}")
        
        for num_tokens in test_cases:
            print(f"\n{'-'*60}")
            print(f"Testing with {num_tokens} tokens")
            print(f"{'-'*40}")

            # Create random logits
            torch.manual_seed(42)  # For reproducibility
            logits = torch.randn(1, num_tokens * vocab_size, dtype=types, device="cuda")
            print(f"Logits shape: {logits.shape}")

            # PyTorch reference
            print("\n1. PyTorch F.softmax:")
            # Apply temperature and reshape for per-token softmax
            logits_scaled = logits / temperature
            logits_reshaped = logits_scaled.view(num_tokens, vocab_size)
            pytorch_output = F.softmax(logits_reshaped, dim=-1)
            pytorch_flat = pytorch_output.view(1, -1)
            print(f"   Output shape: {pytorch_output.shape}")
            print(f"   Flattened shape: {pytorch_flat.shape}")
            
            # Verify probability sums
            sums = torch.sum(pytorch_output, dim=-1)

            print(f"   Max deviation from 1.0: {torch.max(torch.abs(sums - 1.0)).item():.6e}")

            # MPK multi-token softmax
            print("\n2. MPK multi_token_softmax:")
            
            # Create output buffer
            output_buffer = torch.zeros(1, num_tokens * vocab_size, dtype=types, device='cuda:0')

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
            logits_tensor = mpk.attach_input(logits, "logits")
            output_tensor = mpk.attach_input(output_buffer, "output")
            
            # Add multi-token softmax layer
            # Grid: one block per token
            # Block: 512 threads (good for vocab_size=128256)
            mpk.multi_token_softmax_layer(
                logits=logits_tensor,
                output=output_tensor,
                grid_dim=(num_tokens, 1, 1),
                block_dim=(512, 1, 1),
                vocab_size=vocab_size,
                max_tokens=64,
                temperature=temperature
            )
            
            # Compile and execute
            mpk.compile()
            mpk()
            
            print(f"   Output shape: {output_buffer.shape}")
            
            # Verify probability sums for MPK output
            mpk_reshaped = output_buffer.view(num_tokens, vocab_size)
            mpk_sums = torch.sum(mpk_reshaped, dim=-1)
            print(f"   Probability sums: {mpk_sums.cpu().tolist()[:5]}{'...' if num_tokens > 5 else ''}")
            print(f"   Max deviation from 1.0: {torch.max(torch.abs(mpk_sums - 1.0)).item():.6e}")

            # Compare outputs
            print("\n3. Comparison:")
            max_diff = torch.max(torch.abs(output_buffer - pytorch_flat)).item()
            avg_diff = torch.mean(torch.abs(output_buffer - pytorch_flat)).item()

            print(f"   Maximum difference: {max_diff}")
            print(f"   Average difference: {avg_diff}")

            # Check if outputs are close
            tolerance = 1e-3  # BFloat16 precision
            if max_diff < tolerance:
                print(f"   ✓ Test PASSED: Outputs match within tolerance ({tolerance})")
            else:
                print(f"   ✗ Test FAILED: Maximum difference {max_diff} exceeds tolerance {tolerance}")

                # Show some sample values for debugging
                print("\n   Sample values comparison (first 5 elements):")
                for i in range(min(5, output_buffer.shape[1])):
                    print(f"     Index {i}: MPK={output_buffer[0, i]:.6f}, "
                          f"PyTorch={pytorch_flat[0, i]:.6f}")

                # Also show per-token comparison
                print("\n   Per-token analysis:")
                for t in range(min(3, num_tokens)):
                    token_start = t * vocab_size
                    token_end = (t + 1) * vocab_size
                    token_max_diff = torch.max(
                        torch.abs(output_buffer[0, token_start:token_end] - 
                                 pytorch_flat[0, token_start:token_end])
                    ).item()
                    print(f"     Token {t}: max_diff={token_max_diff:.6f}")

    # Test edge cases
    print("\n" + "=" * 80)
    print("Testing edge cases")
    print("=" * 80)
    
    # Test with extreme values
    print("\n1. Testing with extreme logits:")
    num_tokens = 4
    extreme_logits = torch.tensor([1000.0, -1000.0, 0.0, 100.0] * vocab_size, 
                                  dtype=types, device="cuda").unsqueeze(0)
    extreme_logits = extreme_logits[:, :num_tokens * vocab_size]  # Trim to correct size
    
    # PyTorch reference
    extreme_scaled = extreme_logits / 1.0
    extreme_reshaped = extreme_scaled.view(num_tokens, vocab_size)
    pytorch_extreme = F.softmax(extreme_reshaped, dim=-1).view(1, -1)
    
    # MPK
    extreme_output = torch.zeros_like(extreme_logits)
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
    
    extreme_logits_tensor = mpk.attach_input(extreme_logits, "extreme_logits")
    extreme_output_tensor = mpk.attach_input(extreme_output, "extreme_output")
    
    mpk.multi_token_softmax_layer(
        logits=extreme_logits_tensor,
        output=extreme_output_tensor,
        grid_dim=(num_tokens, 1, 1),
        block_dim=(512, 1, 1),
        vocab_size=vocab_size,
        max_tokens=64,
        temperature=1.0
    )
    
    mpk.compile()
    mpk()
    
    extreme_diff = torch.max(torch.abs(extreme_output - pytorch_extreme)).item()
    print(f"   Maximum difference with extreme values: {extreme_diff}")
    if extreme_diff < 1e-3:
        print(f"   ✓ Extreme values test PASSED")
    else:
        print(f"   ✗ Extreme values test FAILED")

    print("\n" + "=" * 60)
    print("Multi-token softmax test completed!")

if __name__ == "__main__":
    test_multi_token_softmax_vs_pytorch()