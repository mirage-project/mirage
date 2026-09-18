"""
Test for MLA (Multi-head Latent Attention) kernel on Blackwell (SM100).

This test validates the Mirage MLA kernel against FlashInfer's reference implementation.
The MLA architecture is used in DeepSeek models and features:
- 128 heads (fixed)
- Compressed KV (latent dimension): 512
- Positional embeddings (rope dimension): 64
- Total head dimension: 576 = 512 + 64
"""

import math
import torch

torch.set_printoptions(sci_mode=False, profile="full")

# Import the MLA kernel extension
import runtime_kernel_blackwell_mla

# Try to import FlashInfer for reference comparison
try:
    import flashinfer
    import flashinfer.mla
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    print("Warning: FlashInfer not found. Skipping FlashInfer comparison tests.")


def attention_ref_pytorch(
    q_nope_pe: torch.Tensor,
    ckv_kpe_cache: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Reference implementation of MLA attention using PyTorch.

    Args:
        q_nope_pe: Query tensor [batch, num_heads, head_dim_total]
        ckv_kpe_cache: Paged KV cache [num_pages, page_size, head_dim_total]
        kv_lens: Sequence lengths [batch]
        page_table: Page indices [batch, max_pages]
        page_size: Size of each page
        softmax_scale: Softmax scale factor

    Returns:
        Output tensor [batch, num_heads, head_dim_latent]
    """
    batch_size, num_heads, head_dim_total = q_nope_pe.shape
    head_dim_latent = 512
    head_dim_rope = 64

    # Split query into latent and rope parts
    q_latent = q_nope_pe[:, :, :head_dim_latent]  # [batch, num_heads, 512]
    q_rope = q_nope_pe[:, :, head_dim_latent:]    # [batch, num_heads, 64]

    outputs = []

    for b in range(batch_size):
        kv_len = kv_lens[b].item()
        num_pages_for_batch = (kv_len + page_size - 1) // page_size

        # Gather KV cache for this batch using page table
        kv_pages = []
        for p in range(num_pages_for_batch):
            page_idx = page_table[b, p].item()
            kv_pages.append(ckv_kpe_cache[page_idx])

        if num_pages_for_batch > 0:
            # Stack pages: [num_pages, page_size, head_dim_total]
            kv_gathered = torch.stack(kv_pages, dim=0)
            # Reshape to [kv_len_padded, head_dim_total]
            kv_gathered = kv_gathered.view(-1, head_dim_total)[:kv_len]
        else:
            kv_gathered = torch.empty(0, head_dim_total, device=q_nope_pe.device, dtype=q_nope_pe.dtype)

        if kv_len == 0:
            # Handle empty sequence
            output_b = torch.zeros(num_heads, head_dim_latent, device=q_nope_pe.device, dtype=q_nope_pe.dtype)
        else:
            # Split KV into latent and rope parts
            # kv_gathered: [kv_len, head_dim_total]
            c_latent = kv_gathered[:, :head_dim_latent]  # [kv_len, 512]
            k_rope = kv_gathered[:, head_dim_latent:]    # [kv_len, 64]

            # Compute attention
            # Q: [num_heads, head_dim_total]
            # K: [kv_len, head_dim_total] (broadcast to all heads)
            # V: [kv_len, head_dim_latent]

            # Combine latent and rope for Q and K
            q_combined = torch.cat([q_latent[b], q_rope[b]], dim=-1)  # [num_heads, 576]
            k_combined = torch.cat([c_latent, k_rope], dim=-1)        # [kv_len, 576]

            # Attention scores: Q @ K^T
            # [num_heads, 576] @ [576, kv_len] = [num_heads, kv_len]
            attn_scores = torch.matmul(q_combined.float(), k_combined.float().T) * softmax_scale

            # Softmax
            attn_probs = torch.softmax(attn_scores, dim=-1)

            # Output: attn_probs @ V (using c_latent as V)
            # [num_heads, kv_len] @ [kv_len, 512] = [num_heads, 512]
            output_b = torch.matmul(attn_probs, c_latent.float()).to(q_nope_pe.dtype)

        outputs.append(output_b)

    return torch.stack(outputs, dim=0)


def attention_ref_flashinfer(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    softmax_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Reference implementation using FlashInfer's BatchMLAPagedAttentionWrapper.

    Args:
        q_nope: Query latent [batch, num_heads, head_dim_latent]
        q_pe: Query rope [batch, num_heads, head_dim_rope]
        ckv_cache: Paged KV latent cache [num_pages, page_size, head_dim_latent]
        kpe_cache: Paged KV rope cache [num_pages, page_size, head_dim_rope]
        kv_lens: Sequence lengths [batch]
        page_table: Page indices [batch, max_pages]
        page_size: Size of each page
        softmax_scale: Softmax scale factor
        device: CUDA device

    Returns:
        Output tensor [batch, num_heads, head_dim_latent]
    """
    if not HAS_FLASHINFER:
        raise RuntimeError("FlashInfer is not installed")

    batch_size = q_nope.size(0)
    num_heads = q_nope.size(1)
    head_dim_latent = q_nope.size(2)
    head_dim_rope = q_pe.size(2)
    pages_per_batch = page_table.size(1)

    # Create FlashInfer MLA wrapper
    # Allocate workspace buffer (128MB should be sufficient)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend="fa2"
    )

    # For decode, each query length is 1
    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * pages_per_batch
    )
    kv_indices = page_table.flatten().contiguous()

    # Plan the attention operation
    mla_wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_latent,
        head_dim_rope,
        page_size,
        False,  # causal=False for decode
        softmax_scale,
        q_nope.dtype,
        ckv_cache.dtype,
    )

    # Run attention
    output = mla_wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=False)

    return output


def test_mla_attention_vs_flashinfer(
    batch_size: int = 4,
    max_seq_len: int = 1024,
    page_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    """Test MLA attention kernel against FlashInfer reference."""
    if not HAS_FLASHINFER:
        print("Skipping FlashInfer comparison test - FlashInfer not installed")
        return

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_heads = 128
    head_dim_latent = 512
    head_dim_rope = 64
    head_dim_total = head_dim_latent + head_dim_rope

    # Calculate pages needed
    pages_per_batch = (max_seq_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_batch

    print(f"\n=== Testing MLA Attention vs FlashInfer ===")
    print(f"  batch_size={batch_size}, max_seq_len={max_seq_len}, page_size={page_size}")
    print(f"  num_heads={num_heads}, head_dim_total={head_dim_total}")
    print(f"  total_pages={total_pages}, pages_per_batch={pages_per_batch}")

    # Create test tensors
    # Query: [batch, num_heads, head_dim_total]
    q_nope_pe = torch.randn(
        batch_size, num_heads, head_dim_total, device=device, dtype=dtype
    )

    # Split query for FlashInfer format
    q_nope = q_nope_pe[:, :, :head_dim_latent].contiguous()
    q_pe = q_nope_pe[:, :, head_dim_latent:].contiguous()

    # KV cache: [num_pages, page_size, head_dim_total]
    ckv_kpe_cache = torch.randn(
        total_pages, page_size, head_dim_total, device=device, dtype=dtype
    )

    # Split KV cache for FlashInfer format
    ckv_cache = ckv_kpe_cache[:, :, :head_dim_latent].contiguous()
    kpe_cache = ckv_kpe_cache[:, :, head_dim_latent:].contiguous()

    # Sequence lengths: [batch]
    kv_lens = torch.full((batch_size,), max_seq_len, device=device, dtype=torch.int32)

    # Page table: [batch, max_pages]
    # Simple sequential page allocation
    page_table = torch.arange(
        0, total_pages, device=device, dtype=torch.int32
    ).view(batch_size, pages_per_batch)

    # Output: [batch, num_heads, head_dim_latent]
    output_mirage = torch.empty(
        batch_size, num_heads, head_dim_latent, device=device, dtype=dtype
    )

    # Workspace for Mirage
    workspace_size = runtime_kernel_blackwell_mla.get_workspace_size(batch_size, max_seq_len)
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)

    # Softmax scale (before matrix absorption: 1/sqrt(128+64) = 1/sqrt(192))
    # Note: FlashInfer uses the original head dimension before absorption
    softmax_scale = 1.0 / math.sqrt(128 + 64)

    print(f"  softmax_scale={softmax_scale:.6f}")
    print(f"  workspace_size={workspace_size} bytes")

    # Run FlashInfer reference
    print("\nRunning FlashInfer reference...")
    output_flashinfer = attention_ref_flashinfer(
        q_nope, q_pe, ckv_cache, kpe_cache,
        kv_lens, page_table, page_size, softmax_scale, device
    )

    # Run Mirage MLA kernel
    print("Running Mirage MLA kernel...")
    runtime_kernel_blackwell_mla.mla_attention(
        q_nope_pe,
        ckv_kpe_cache,
        kv_lens,
        page_table,
        output_mirage,
        workspace,
        softmax_scale,
    )

    # Compare results
    print("\nComparing Mirage vs FlashInfer...")
    try:
        torch.testing.assert_close(output_mirage, output_flashinfer, rtol=1e-2, atol=1e-2)
        print("Test PASSED! Mirage matches FlashInfer.")
    except AssertionError as e:
        print(f"Test FAILED: {e}")

        # Debug info
        diff = (output_mirage - output_flashinfer).abs()
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")
        print(f"  Mirage output range: [{output_mirage.min().item():.4f}, {output_mirage.max().item():.4f}]")
        print(f"  FlashInfer output range: [{output_flashinfer.min().item():.4f}, {output_flashinfer.max().item():.4f}]")
        raise

    # Benchmark comparison
    print("\n=== Benchmark Comparison ===")

    repetitions = 100

    # Warmup Mirage
    for _ in range(16):
        runtime_kernel_blackwell_mla.mla_attention(
            q_nope_pe, ckv_kpe_cache, kv_lens, page_table,
            output_mirage, workspace, softmax_scale,
        )

    # Benchmark Mirage
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(repetitions):
        runtime_kernel_blackwell_mla.mla_attention(
            q_nope_pe, ckv_kpe_cache, kv_lens, page_table,
            output_mirage, workspace, softmax_scale,
        )
    ender.record()
    torch.cuda.synchronize()
    mirage_time = starter.elapsed_time(ender) / repetitions
    print(f"Mirage MLA: {mirage_time:.4f} ms")

    # Warmup FlashInfer
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend="fa2"
    )
    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_per_batch
    kv_indices = page_table.flatten().contiguous()
    mla_wrapper.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, head_dim_latent, head_dim_rope,
        page_size, False, softmax_scale, q_nope.dtype, ckv_cache.dtype,
    )

    for _ in range(16):
        mla_wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=False)

    # Benchmark FlashInfer
    torch.cuda.synchronize()
    starter.record()
    for _ in range(repetitions):
        mla_wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=False)
    ender.record()
    torch.cuda.synchronize()
    flashinfer_time = starter.elapsed_time(ender) / repetitions
    print(f"FlashInfer MLA: {flashinfer_time:.4f} ms")

    speedup = flashinfer_time / mirage_time
    print(f"Speedup (Mirage vs FlashInfer): {speedup:.2f}x")


def test_mla_attention_vs_pytorch(
    batch_size: int = 4,
    max_seq_len: int = 1024,
    page_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    """Test MLA attention kernel against PyTorch reference."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_heads = 128
    head_dim_latent = 512
    head_dim_rope = 64
    head_dim_total = head_dim_latent + head_dim_rope

    # Calculate pages needed
    pages_per_batch = (max_seq_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_batch

    print(f"\n=== Testing MLA Attention vs PyTorch ===")
    print(f"  batch_size={batch_size}, max_seq_len={max_seq_len}, page_size={page_size}")
    print(f"  num_heads={num_heads}, head_dim_total={head_dim_total}")
    print(f"  total_pages={total_pages}, pages_per_batch={pages_per_batch}")

    # Create test tensors
    # Query: [batch, num_heads, head_dim_total]
    q_nope_pe = torch.randn(
        batch_size, num_heads, head_dim_total, device=device, dtype=dtype
    )

    # KV cache: [num_pages, page_size, head_dim_total]
    ckv_kpe_cache = torch.randn(
        total_pages, page_size, head_dim_total, device=device, dtype=dtype
    )

    # Sequence lengths: [batch]
    kv_lens = torch.full((batch_size,), max_seq_len, device=device, dtype=torch.int32)

    # Page table: [batch, max_pages]
    # Simple sequential page allocation
    page_table = torch.arange(
        0, total_pages, device=device, dtype=torch.int32
    ).view(batch_size, pages_per_batch)

    # Output: [batch, num_heads, head_dim_latent]
    output = torch.empty(
        batch_size, num_heads, head_dim_latent, device=device, dtype=dtype
    )

    # Workspace
    workspace_size = runtime_kernel_blackwell_mla.get_workspace_size(batch_size, max_seq_len)
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)

    # Softmax scale (before matrix absorption: 1/sqrt(128+64) = 1/sqrt(192))
    softmax_scale = 1.0 / math.sqrt(128 + 64)

    print(f"  softmax_scale={softmax_scale:.6f}")
    print(f"  workspace_size={workspace_size} bytes")

    # Run reference implementation
    print("\nRunning PyTorch reference implementation...")
    output_ref = attention_ref_pytorch(
        q_nope_pe, ckv_kpe_cache, kv_lens, page_table, page_size, softmax_scale
    )
    # print('torch result:', output_ref)
    

    # Run Mirage MLA kernel
    print("Running Mirage MLA kernel...")
    runtime_kernel_blackwell_mla.mla_attention(
        q_nope_pe,
        ckv_kpe_cache,
        kv_lens,
        page_table,
        output,
        workspace,
        softmax_scale,
    )
    # print('our result:', output)

    # Compare results
    print("\nComparing results...")
    try:
        torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)
        print("Test PASSED!")
    except AssertionError as e:
        print(f"Test FAILED: {e}")

        # Debug info
        diff = (output - output_ref).abs()
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Ref range: [{output_ref.min().item():.4f}, {output_ref.max().item():.4f}]")
        raise

    # Benchmark
    print("\nRunning benchmark...")
    # Warmup
    for _ in range(16):
        runtime_kernel_blackwell_mla.mla_attention(
            q_nope_pe,
            ckv_kpe_cache,
            kv_lens,
            page_table,
            output,
            workspace,
            softmax_scale,
        )

    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    repetitions = 100
    starter.record()
    for _ in range(repetitions):
        runtime_kernel_blackwell_mla.mla_attention(
            q_nope_pe,
            ckv_kpe_cache,
            kv_lens,
            page_table,
            output,
            workspace,
            softmax_scale,
        )
    ender.record()
    torch.cuda.synchronize()
    total_time = starter.elapsed_time(ender)
    avg_time = total_time / repetitions
    print(f"Average time over {repetitions} runs: {avg_time:.4f} ms")


# def test_mla_attention_various_configs():
#     """Test MLA attention with various configurations."""
#     configs = [
#         {"batch_size": 1, "max_seq_len": 128, "page_size": 128},
#         {"batch_size": 4, "max_seq_len": 1024, "page_size": 128},
#         {"batch_size": 8, "max_seq_len": 2048, "page_size": 64},
#         {"batch_size": 2, "max_seq_len": 4096, "page_size": 128},
#     ]

#     for config in configs:
#         try:
#             test_mla_attention_vs_pytorch(**config)
#             if HAS_FLASHINFER:
#                 test_mla_attention_vs_flashinfer(**config)
#         except Exception as e:
#             print(f"Failed with config {config}: {e}")




if __name__ == "__main__":
    print("=" * 60)
    print("MLA Attention Kernel Tests")
    print("=" * 60)

    # Run PyTorch comparison test
    test_mla_attention_vs_pytorch(batch_size=4, max_seq_len=1024, page_size=128)

    # Run FlashInfer comparison test if available
    if HAS_FLASHINFER:
        test_mla_attention_vs_flashinfer(batch_size=4, max_seq_len=1024, page_size=128)
    else:
        print("\nSkipping FlashInfer comparison - install with: pip install flashinfer")

    # Optionally run sweep
    # test_mla_correctness_sweep()
