import mirage as mi
import torch

types = torch.bfloat16


def test_multi_token_embedding_persistent_kernel():
    """
    Test multi_token_embedding by comparing with single_token_embedding
    Using LLaMA 3.1 8B settings
    """
    print("Testing Multi-Token Embedding vs Single-Token Embedding")
    print(f"Using type: {types}")
    print("=" * 60)

    # LLaMA 3.1 8B parameters
    vocab_size = 128256  # LLaMA 3.1 vocabulary size
    embedding_dim = 4096  # LLaMA 3.1 8B embedding dimension
    num_tokens = 8  # Number of tokens to embed (typical for EAGLE tree)

    # Initialize
    mi.set_gpu_device_id(0)

    # Create embedding table
    embedding_table = torch.randn(vocab_size, embedding_dim, dtype=types, device="cuda")

    # Create random token IDs
    token_ids = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.int64, device="cuda")

    # Get proper configuration from GPU
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    print(f"GPU configuration: num_workers={num_workers}, num_schedulers={num_schedulers}")

    print(f"\nLLaMA 3.1 8B parameters:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  num_tokens: {num_tokens}")
    print(f"  token_ids: {token_ids.cpu().tolist()}")

    # Test 1: Single-token embedding (multiple calls)
    print("\nTest 1: Single-token embedding (MPK)")
    print("-" * 40)

    # Collect outputs from multiple single-token embeddings
    single_token_outputs = []
    
    for i in range(num_tokens):
        # Create tensors for meta information
        step = torch.tensor([i], dtype=torch.int32, device="cuda")
        tokens_buffer = torch.zeros(512, dtype=torch.int64, device="cuda")
        tokens_buffer[:num_tokens] = token_ids
        
        # Create output buffer for single token
        output_buffer_single = torch.zeros(1, embedding_dim, dtype=types, device='cuda:0')
        
        # Create PersistentKernel for single-token embedding
        mpk_single = mi.PersistentKernel(
            world_size=1,
            mpi_rank=0,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=512,
            eos_token_id=128001,
            meta_tensors=[step, tokens_buffer],
            profiler_tensor=None,
        )
        
        # Attach tensors
        input_placeholder = mpk_single.attach_input(
            torch_tensor=torch.zeros(1, dtype=torch.int64, device="cuda"), 
            name="input"
        )
        embed_table_tensor = mpk_single.attach_input(
            torch_tensor=embedding_table, 
            name="embed_table"
        )
        output_tensor = mpk_single.attach_input(
            torch_tensor=output_buffer_single, 
            name="output"
        )
        
        # Add embedding layer (single token)
        mpk_single.embed_layer(
            input=input_placeholder,
            weight=embed_table_tensor,
            output=output_tensor,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1)
        )
        
        # Compile and execute
        mpk_single.compile()
        mpk_single()
        
        single_token_outputs.append(output_buffer_single.clone())
    
    # Concatenate all single-token outputs
    single_concat = torch.cat(single_token_outputs, dim=1)
    print(f"Single-token outputs concatenated shape: {single_concat.shape}")

    # Test 2: Multi-token embedding
    print("\nTest 2: Multi-token embedding (MPK)")
    print("-" * 40)

    # Create output buffer for multi-token
    output_buffer_multi = torch.zeros(1, num_tokens * embedding_dim, dtype=types, device='cuda:0')

    # Create tensors for meta information
    step_multi = torch.zeros(1, dtype=torch.int32, device="cuda")
    tokens_buffer_multi = torch.zeros(512, dtype=torch.int64, device="cuda")

    # Create PersistentKernel for multi-token test
    mpk_multi = mi.PersistentKernel(
        world_size=1,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=512,
        eos_token_id=128001,
        meta_tensors=[step_multi, tokens_buffer_multi],
        profiler_tensor=None,
    )
    
    # Attach tensors
    token_ids_tensor = mpk_multi.attach_input(
        torch_tensor=token_ids, 
        name="token_ids"
    )
    embed_table_tensor_multi = mpk_multi.attach_input(
        torch_tensor=embedding_table, 
        name="embed_table"
    )
    output_tensor_multi = mpk_multi.attach_input(
        torch_tensor=output_buffer_multi, 
        name="output"
    )
    
    # Add multi-token embedding layer
    mpk_multi.multi_token_embed_layer(
        token_ids=token_ids_tensor,
        weight=embed_table_tensor_multi,
        output=output_tensor_multi,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        max_tokens=32
    )
    
    # Compile and execute
    print("Compiling multi-token embedding kernel...")
    mpk_multi.compile()
    print("Executing multi-token embedding kernel...")
    mpk_multi()
    
    print(f"Multi-token output shape: {output_buffer_multi.shape}")

    # Test 3: Compare outputs
    print("\nTest 3: Comparing single vs multi-token outputs")
    print("-" * 40)

    # Compare the outputs
    max_diff = torch.max(torch.abs(output_buffer_multi - single_concat)).item()
    avg_diff = torch.mean(torch.abs(output_buffer_multi - single_concat)).item()

    print(f"Maximum difference: {max_diff}")
    print(f"Average difference: {avg_diff}")

    # Check if outputs are close
    tolerance = 1e-3  # BFloat16 precision
    if max_diff < tolerance:
        print(f"✓ Test PASSED: Outputs match within tolerance ({tolerance})")
    else:
        print(f"✗ Test FAILED: Maximum difference {max_diff} exceeds tolerance {tolerance}")

        # Show some sample values for debugging
        print("\nSample values comparison (first 5 elements):")
        for i in range(min(5, output_buffer_multi.shape[1])):
            print(f"  Index {i}: Multi={output_buffer_multi[0, i]:.6f}, "
                  f"Single={single_concat[0, i]:.6f}")

    # Test 4: Test with different token counts
    print("\nTest 4: Testing with different token counts")
    print("-" * 40)

    for test_num_tokens in [1, 4, 8, 16]:
        print(f"  Testing with {test_num_tokens} tokens... ", end="")
        
        # Create test token IDs
        test_token_ids = torch.randint(0, vocab_size, (test_num_tokens,), 
                                      dtype=torch.int64, device="cuda")
        
        # Multi-token output
        test_output_multi = torch.zeros(1, test_num_tokens * embedding_dim, 
                                       dtype=types, device='cuda:0')
        
        # Create new MPK for this test
        test_mpk_multi = mi.PersistentKernel(
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
        
        # Attach and run multi-token
        test_token_ids_tensor = test_mpk_multi.attach_input(test_token_ids, "token_ids")
        test_embed_table = test_mpk_multi.attach_input(embedding_table, "embed_table")
        test_output_tensor = test_mpk_multi.attach_input(test_output_multi, "output")
        
        test_mpk_multi.multi_token_embed_layer(
            token_ids=test_token_ids_tensor,
            weight=test_embed_table,
            output=test_output_tensor,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            max_tokens=32
        )
        
        test_mpk_multi.compile()
        test_mpk_multi()
        
        # Single-token outputs
        test_single_outputs = []
        for i in range(test_num_tokens):
            test_step = torch.tensor([i], dtype=torch.int32, device="cuda")
            test_tokens_buffer = torch.zeros(512, dtype=torch.int64, device="cuda")
            test_tokens_buffer[:test_num_tokens] = test_token_ids
            
            test_output_single = torch.zeros(1, embedding_dim, dtype=types, device='cuda:0')
            
            test_mpk_single = mi.PersistentKernel(
                world_size=1,
                mpi_rank=0,
                num_workers=num_workers,
                num_local_schedulers=num_schedulers,
                num_remote_schedulers=0,
                max_seq_length=512,
                eos_token_id=128001,
                meta_tensors=[test_step, test_tokens_buffer],
                profiler_tensor=None,
            )
            
            test_input = test_mpk_single.attach_input(
                torch.zeros(1, dtype=torch.int64, device="cuda"), "input"
            )
            test_embed = test_mpk_single.attach_input(embedding_table, "embed_table")
            test_out = test_mpk_single.attach_input(test_output_single, "output")
            
            test_mpk_single.embed_layer(
                input=test_input,
                weight=test_embed,
                output=test_out,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1)
            )
            
            test_mpk_single.compile()
            test_mpk_single()
            
            test_single_outputs.append(test_output_single.clone())
        
        test_single_concat = torch.cat(test_single_outputs, dim=1)
        
        # Compare
        test_max_diff = torch.max(torch.abs(test_output_multi - test_single_concat)).item()
        status = "✓ PASS" if test_max_diff < tolerance else "✗ FAIL"
        print(f"max_diff={test_max_diff:.6f} {status}")

    print("\n" + "=" * 60)
    print("Multi-token embedding test completed!")


if __name__ == "__main__":
    test_multi_token_embedding_persistent_kernel()