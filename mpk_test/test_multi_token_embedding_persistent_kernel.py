import mirage as mi
import torch

types = torch.bfloat16


def test_multi_token_embedding_vs_pytorch():
    """
    Test multi_token_embedding by comparing with PyTorch's F.embedding
    Using LLaMA 3.1 8B settings
    """
    print("Testing Multi-Token Embedding vs PyTorch")
    print(f"Using type: {types}")
    print("=" * 60)

    # LLaMA 3.1 8B parameters
    vocab_size = 128256  # LLaMA 3.1 vocabulary size
    embedding_dim = 4096  # LLaMA 3.1 8B embedding dimension

    # Initialize
    mi.set_gpu_device_id(0)

    # Create embedding table
    embedding_table = torch.randn(vocab_size, embedding_dim, dtype=types, device="cuda")

    # Get proper configuration from GPU
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    print(f"GPU configuration: num_workers={num_workers}, num_schedulers={num_schedulers}")

    print(f"\nLLaMA 3.1 8B parameters:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  embedding_dim: {embedding_dim}")

    # Test different token counts
    test_cases = [1, 4, 8, 16, 32]
    
    for num_tokens in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing with {num_tokens} tokens")
        print(f"{'-'*40}")

        # Create random token IDs
        token_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64, device="cuda")
        print(f"Token IDs: {token_ids.cpu().tolist()[:5]}{'...' if num_tokens > 5 else ''}")

        # PyTorch reference
        print("\n1. PyTorch F.embedding:")
        pytorch_output = torch.nn.functional.embedding(token_ids, embedding_table)
        pytorch_flat = pytorch_output.view(1, -1)
        print(f"   Output shape: {pytorch_output.shape}")
        print(f"   Flattened shape: {pytorch_flat.shape}")

        # MPK multi-token embedding
        print("\n2. MPK multi_token_embedding:")
        
        # Create output buffer
        output_buffer = torch.zeros(1, num_tokens * embedding_dim, dtype=types, device='cuda:0')

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
        token_ids_tensor = mpk.attach_input(token_ids, "token_ids")
        embed_table_tensor = mpk.attach_input(embedding_table, "embed_table")
        output_tensor = mpk.attach_input(output_buffer, "output")
        
        # Add multi-token embedding layer
        mpk.multi_token_embed_layer(
            token_ids=token_ids_tensor,
            weight=embed_table_tensor,
            output=output_tensor,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            max_tokens=64  # Increase to handle up to 64 tokens
        )
        
        # Compile and execute
        mpk.compile()
        mpk()
        
        print(f"   Output shape: {output_buffer.shape}")

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
            mpk_reshaped = output_buffer.view(num_tokens, embedding_dim)
            for t in range(min(3, num_tokens)):
                token_start = t * embedding_dim
                token_end = (t + 1) * embedding_dim
                token_max_diff = torch.max(
                    torch.abs(output_buffer[0, token_start:token_end] - 
                             pytorch_flat[0, token_start:token_end])
                ).item()
                print(f"     Token {t} (ID={token_ids[t].item()}): max_diff={token_max_diff:.6f}")

    print("\n" + "=" * 60)
    print("Multi-token embedding test completed!")

if __name__ == "__main__":
    test_multi_token_embedding_vs_pytorch()