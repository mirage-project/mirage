import torch
import torch.nn.functional as F
import runtime_kernel
import numpy as np
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
q_heads = 4
k_heads = 1
v_heads = 1
head_dim = 128
num_tokens = 3
max_seq_len = 512

device = "cuda"
dtype = torch.bfloat16


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def pytorch_multitoken_attention(qkv, k_cache, v_cache, seq_len):
    """
    PyTorch reference implementation.

    Args:
        qkv: [num_tokens * (q_heads + k_heads + v_heads), head_dim]
        k_cache: [max_seq_len, head_dim]
        v_cache: [max_seq_len, head_dim]
        seq_len: Position where new tokens will be written

    Returns:
        outputs: [num_tokens, q_heads, head_dim]
        updated_k_cache: [max_seq_len, head_dim]
        updated_v_cache: [max_seq_len, head_dim]
    """
    heads_per_token = q_heads + k_heads + v_heads
    outputs = []

    # Clone caches for updates
    updated_k_cache = k_cache.clone()
    updated_v_cache = v_cache.clone()

    # Process each token
    for token_idx in range(num_tokens):
        offset = token_idx * heads_per_token

        # Extract Q, K, V for current token
        q = qkv[offset : offset + q_heads].unsqueeze(1)  # [q_heads, 1, head_dim]
        k_new = qkv[offset + q_heads]  # [head_dim]
        v_new = qkv[offset + q_heads + k_heads]  # [head_dim]

        # Update cache with new K, V
        updated_k_cache[seq_len - 1 + token_idx] = k_new
        updated_v_cache[seq_len - 1 + token_idx] = v_new

        # Compute attention using cache (excluding new tokens)
        if seq_len > 1:
            # Use positions 0 to seq_len-2 for attention
            k_context = k_cache[: seq_len - 1].unsqueeze(0).expand(q_heads, -1, -1)
            v_context = v_cache[: seq_len - 1].unsqueeze(0).expand(q_heads, -1, -1)

            # Attention computation
            scores = torch.matmul(q, k_context.transpose(-2, -1)) / (head_dim**0.5)
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v_context).squeeze(1)
        else:
            # No context, output zeros
            output = torch.zeros((q_heads, head_dim), device=device, dtype=dtype)

        outputs.append(output)

    return torch.stack(outputs), updated_k_cache, updated_v_cache


def test_multitoken_kernel():
    """Test the multitoken kernel with random initialization."""

    print("Multi-token Decoding Kernel Test")
    print("=" * 60)

    # Test different sequence lengths
    test_cases = [
        (1, "No context"),
        (2, "Single context token"),
        (5, "Multiple context tokens"),
        (32, "Medium context"),
        (100, "Large context"),
    ]

    all_passed = True

    for seq_len, description in test_cases:
        print(f"\nTest Case: seq_len={seq_len} ({description})")
        print("-" * 40)
        torch.cuda.empty_cache()

        # 1. Random initialization
        # Initialize caches with random values
        k_cache_init = (
            torch.randn((max_seq_len, head_dim), device=device, dtype=dtype) * 0.1
        )
        v_cache_init = (
            torch.randn((max_seq_len, head_dim), device=device, dtype=dtype) * 0.1
        )

        # Zero out positions that will be written to
        for i in range(seq_len - 1, min(seq_len - 1 + num_tokens, max_seq_len)):
            k_cache_init[i] = 0
            v_cache_init[i] = 0

        # Generate random QKV for all tokens
        heads_per_token = q_heads + k_heads + v_heads
        qkv = (
            torch.randn(
                num_tokens * heads_per_token, head_dim, device=device, dtype=dtype
            )
            * 0.1
        )

        # 2. Calculate expected results using PyTorch
        # Measure memory before PyTorch execution
        mem_before_pytorch = get_memory_usage()

        # Measure PyTorch execution time
        start_time_pytorch = time.perf_counter()
        expected_outputs, expected_k_cache, expected_v_cache = (
            pytorch_multitoken_attention(
                qkv, k_cache_init.clone(), v_cache_init.clone(), seq_len
            )
        )
        torch.cuda.synchronize()
        end_time_pytorch = time.perf_counter()

        # Measure memory after PyTorch execution
        mem_after_pytorch = get_memory_usage()

        pytorch_time_ms = (end_time_pytorch - start_time_pytorch) * 1000
        pytorch_memory_used_mb = mem_after_pytorch - mem_before_pytorch

        # Print profiling results for PyTorch
        print(f"PyTorch Performance:")
        print(f"  Execution time: {pytorch_time_ms:.3f} ms")
        print(f"  Memory used: {pytorch_memory_used_mb:.2f} MB")

        # 3. Run the kernel with profiling
        k_cache_kernel = k_cache_init.clone()
        v_cache_kernel = v_cache_init.clone()
        kernel_outputs = torch.empty(
            (num_tokens * q_heads, head_dim), device=device, dtype=dtype
        )

        # Measure memory before kernel execution
        mem_before = get_memory_usage()

        # Measure kernel execution time
        start_time = time.perf_counter()
        runtime_kernel.single_batch_multitoken_decoding(
            qkv,
            k_cache_kernel,
            v_cache_kernel,
            kernel_outputs,
            seq_len,
            False,
            False,
            None,
            None,
            None,
            None,
            0.0,
            0.0,
        )
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # Measure memory after kernel execution
        mem_after = get_memory_usage()

        kernel_time_ms = (end_time - start_time) * 1000
        memory_used_mb = mem_after - mem_before

        kernel_outputs = kernel_outputs.view(num_tokens, q_heads, head_dim)

        # Print profiling results
        print(f"Kernel Performance:")
        print(f"  Execution time: {kernel_time_ms:.3f} ms")
        print(f"  Memory used: {memory_used_mb:.2f} MB")

        # 4. Compare results
        # Compare attention outputs
        output_diff = (expected_outputs - kernel_outputs).abs()
        max_output_diff = output_diff.max().item()
        mean_output_diff = output_diff.mean().item()

        print(f"Attention Output Comparison:")
        print(f"  Max difference: {max_output_diff:.6f}")
        print(f"  Mean difference: {mean_output_diff:.6f}")

        # Check each token's output
        token_passed = True
        for token_idx in range(num_tokens):
            token_diff = output_diff[token_idx].max().item()
            token_norm_expected = expected_outputs[token_idx].norm().item()
            token_norm_kernel = kernel_outputs[token_idx].norm().item()

            if token_diff > 0.01:
                print(
                    f"  Token {token_idx}: FAILED - diff={token_diff:.6f}, "
                    f"expected_norm={token_norm_expected:.3f}, "
                    f"kernel_norm={token_norm_kernel:.3f}"
                )
                token_passed = False
            else:
                print(f"  Token {token_idx}: PASSED - diff={token_diff:.6f}")

        # Compare K cache updates
        print(f"\nK Cache Updates:")
        k_cache_passed = True
        for token_idx in range(num_tokens):
            pos = seq_len - 1 + token_idx
            if pos < max_seq_len:
                k_diff = (
                    (expected_k_cache[pos] - k_cache_kernel[pos]).abs().max().item()
                )
                if k_diff > 1e-5:
                    print(
                        f"  Position {pos} (token {token_idx}): FAILED - diff={k_diff:.6f}"
                    )
                    k_cache_passed = False
                else:
                    print(f"  Position {pos} (token {token_idx}): PASSED")

        # Compare V cache updates
        print(f"\nV Cache Updates:")
        v_cache_passed = True
        for token_idx in range(num_tokens):
            pos = seq_len - 1 + token_idx
            if pos < max_seq_len:
                v_diff = (
                    (expected_v_cache[pos] - v_cache_kernel[pos]).abs().max().item()
                )
                if v_diff > 1e-5:
                    print(
                        f"  Position {pos} (token {token_idx}): FAILED - diff={v_diff:.6f}"
                    )
                    v_cache_passed = False
                else:
                    print(f"  Position {pos} (token {token_idx}): PASSED")

        # Overall test case result
        case_passed = token_passed and k_cache_passed and v_cache_passed
        if case_passed:
            print(f"\nResult: ✅ PASSED")
        else:
            print(f"\nResult: ❌ FAILED")
            all_passed = False

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    test_multitoken_kernel()
