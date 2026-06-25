"""Test the ``layers.PagedAttention`` catalog module via PersistentKernel test_mode.

This is the Phase-2 test for the qwen3 production attention kernel
(``pk.paged_attention_layer`` -> ``multitoken_paged_attention_task_impl``).
It exercises a SINGLE PREFILL of a small sequence — the kernel handles
both prefill and decode but prefill is the more demanding path
(num_tokens > 1, causal mask, multiple page writes).

What we exercise
----------------

1. Build a small ``PagedAttention`` module on CUDA (bf16).
2. Allocate a fused QKV input row buffer for ``seq_len`` new tokens.
3. Allocate paged k/v caches sized to PK's ``(max_num_pages, page_size,
   num_kv_heads, head_dim)``. Use 1 page large enough to hold the whole
   sequence — keeps the meta-tensor wiring trivial.
4. Allocate cos/sin RoPE tables and the positions/ output buffer.
5. Run the PyTorch reference on a fresh copy of the same inputs/cache so
   forward() and compile() share identical pre-state.
6. Seed the runtime meta tensors (``qo_indptr_buffer``,
   ``paged_kv_indptr_buffer``, ``paged_kv_indices_buffer``,
   ``paged_kv_last_page_len_buffer``) for the single-prefill scenario.
7. Inside ``with pk.compile_scope():`` register the task via
   ``m.compile(...)``. Compile, launch once, sync.
8. Diff against the reference at bf16-attention tolerance (~0.5).

Why these dims
--------------

* ``head_dim = 64``: small enough to keep smem comfortable, large
  enough to exercise the rotary embedding's half-split. (The qwen3
  production deployment uses 128; both work — the kernel templates on
  ``HEAD_DIM``.)
* ``num_heads = 4, num_kv_heads = 2`` (GQA group=2). Standard GQA
  ratio — small enough for fast compile, large enough to exercise the
  ``NUM_QO_PER_KV`` path.
* ``batch = 1``: the kernel grid is per-request; one request keeps the
  meta-tensor seeding simple.
* ``seq_len = 8``: small prefill. Number of new-Q tokens AND the total
  attended sequence length (we're not appending to a pre-existing
  cache — that decode case has its own test under the plain
  ``Attention`` module).
* ``page_size = 64``: minimum that satisfies the kernel's
  ``PAGE_SIZE % KV_TILE_SIZE == 0`` static assert with
  ``KV_TILE_SIZE == 64``.
* ``max_num_pages = 1``: one page per layer suffices for ``seq_len <=
  page_size``.

DO NOT execute this file as part of Phase 2 — Phase 4 runs it on a free
GPU. The ``mirage`` conda env is required.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
# The catalog only re-exports a couple of leaf modules through
# ``mirage.mpk.layers`` today; ``PagedAttention`` lives in its own
# subpackage and Phase-2 agents do NOT touch ``__init__.py``.
from mirage.mpk.layers.attention.paged_attention import PagedAttention


def test_paged_attention_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Dimensions (see module docstring for the constraints)
    # ------------------------------------------------------------------
    batch_size = 1                  # number of requests
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    seq_len = 8                     # # of new Q tokens (full prefill)
    page_size = 64                  # must be a multiple of 64 (KV_TILE_SIZE)
    max_num_pages = 1               # one page per layer suffices
    max_seq_len = page_size         # total attended cap
    max_num_batched_tokens = seq_len
    layer_idx = 0

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    fused_qkv_size = q_size + kv_size + kv_size   # [Q | K | V]

    # ------------------------------------------------------------------
    # Build module and seed weights with a non-trivial scale.
    # ------------------------------------------------------------------
    m = PagedAttention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        layer_idx=layer_idx,
        prefix="test_",
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        m.q_norm.data.copy_(
            torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
        )
        m.k_norm.data.copy_(
            torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
        )

    # ------------------------------------------------------------------
    # Build the fused QKV input. Shape: (max_num_batched_tokens,
    # fused_qkv_size) — first axis is the *flat* token index across all
    # in-flight requests; for one request it's just seq_len.
    # ------------------------------------------------------------------
    fused_qkv = torch.randn(
        max_num_batched_tokens, fused_qkv_size,
        dtype=dtype, device=device,
    )

    # ------------------------------------------------------------------
    # Paged KV cache. Shape: (max_num_pages, page_size, num_kv_heads,
    # head_dim). Zero-initialise; the kernel will write its writes.
    # ------------------------------------------------------------------
    k_cache_paged = torch.zeros(
        max_num_pages, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    v_cache_paged = torch.zeros(
        max_num_pages, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )

    # ------------------------------------------------------------------
    # RoPE tables. (max_seq_len, head_dim).
    # ------------------------------------------------------------------
    positions_arr = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000 ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    ))
    freqs = positions_arr.unsqueeze(1) * inv_freq.unsqueeze(0)  # (max_seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                       # (max_seq_len, head_dim)
    cos_tab = emb.cos().to(dtype)
    sin_tab = emb.sin().to(dtype)

    # ------------------------------------------------------------------
    # PyTorch reference. Use a CONTIGUOUS (B, max_seq_len, H_kv, D)
    # cache for the reference — the math is identical to the paged
    # layout when there is one request and one page (each new token's
    # absolute position == its page-local position).
    # ------------------------------------------------------------------
    k_cache_ref = torch.zeros(
        batch_size, max_seq_len, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    v_cache_ref = torch.zeros(
        batch_size, max_seq_len, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    # Per-request reshape of the flat-token QKV buffer.
    qkv_per_req = fused_qkv.view(batch_size, seq_len, fused_qkv_size)
    # Positions for the new Q tokens — full prefill, so [0, seq_len).
    positions_per_req = torch.arange(
        seq_len, dtype=torch.int32, device=device
    ).unsqueeze(0).expand(batch_size, -1).contiguous()  # (B, T)

    ref = m.forward(
        qkv=qkv_per_req,
        cos=cos_tab,
        sin=sin_tab,
        k_cache=k_cache_ref,
        v_cache=v_cache_ref,
        positions=positions_per_req,
    )  # (B, T, H*D)
    ref_flat = ref.view(max_num_batched_tokens, num_heads * head_dim)

    # Output buffer the test driver reads back from.
    out_buf = torch.zeros(
        max_num_batched_tokens, num_heads * head_dim,
        dtype=dtype, device=device,
    )

    # ------------------------------------------------------------------
    # Build PersistentKernel in test mode.
    #
    # We pre-seed ALL the meta tensors the paged kernel reads:
    #   * qo_indptr_buffer: [0, seq_len] — one request with `seq_len` new
    #     Q tokens.
    #   * paged_kv_indptr_buffer: [0, 1] — one request with 1 page.
    #   * paged_kv_indices_buffer: [0] — the single request uses page 0.
    #   * paged_kv_last_page_len_buffer: [seq_len] — `seq_len` valid
    #     entries in the last page (== total seq len since we have one
    #     page).
    #
    # max_seq_length here = page_size (== total attended cap). PK
    # asserts ``meta_tensors["tokens"].shape[1] == max_seq_length`` so
    # the auto-allocated ``tokens`` buffer is sized accordingly.
    # ------------------------------------------------------------------
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_seq_length"] = max_seq_len
    params["max_num_batched_tokens"] = max_num_batched_tokens
    params["max_num_batched_requests"] = batch_size
    params["max_num_pages"] = max_num_pages
    params["page_size"] = page_size

    # Pre-seed the meta tensors BEFORE PK init —
    # _apply_test_mode_meta_defaults only inserts defaults for missing
    # keys.
    qo_indptr_buffer = torch.tensor(
        [0, seq_len], dtype=torch.int32, device=device,
    )
    # Pad to max_num_batched_requests + 1 == 2 — already the right size.
    paged_kv_indptr_buffer = torch.tensor(
        [0, 1], dtype=torch.int32, device=device,
    )
    paged_kv_indices_buffer = torch.tensor(
        [0], dtype=torch.int32, device=device,
    )
    paged_kv_last_page_len_buffer = torch.tensor(
        [seq_len], dtype=torch.int32, device=device,
    )
    params["meta_tensors"] = {
        "qo_indptr_buffer": qo_indptr_buffer,
        "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
        "paged_kv_indices_buffer": paged_kv_indices_buffer,
        "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
    }

    pk = PersistentKernel(**params)

    # ------------------------------------------------------------------
    # Attach the tensors that aren't owned by the module.
    # ------------------------------------------------------------------
    input_dt = pk.attach_input(fused_qkv, name="qkv_in")
    k_cache_dt = pk.attach_input(k_cache_paged, name="k_cache_paged")
    v_cache_dt = pk.attach_input(v_cache_paged, name="v_cache_paged")
    cos_dt = pk.attach_input(cos_tab, name="cos_tab")
    sin_dt = pk.attach_input(sin_tab, name="sin_tab")

    with pk.compile_scope():
        _ = m.compile(
            input=input_dt,
            k_cache=k_cache_dt,
            v_cache=v_cache_dt,
            cos=cos_dt,
            sin=sin_dt,
            output=out_buf,
            # Force the canonical grid; this matches
            # demo/qwen3/demo.py:593 and the kernel's per-(request,
            # kv-head) parallelisation.
            grid_dim=(batch_size, num_kv_heads, 1),
            block_dim=(128, 1, 1),
        )

    # ------------------------------------------------------------------
    # Compile and run once.
    # ------------------------------------------------------------------
    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Compare.
    # ------------------------------------------------------------------
    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref_flat[0, :8]: {ref_flat[0, :8]}")

    max_diff = (out_buf.float() - ref_flat.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        # bf16 fused attention is noisy; 0.5 matches the bar used by
        # test_attention.py and test_qwen3_mlp_testmode.py for similar
        # fused pipelines. If Phase 4 finds this too strict we can
        # loosen to 1.0.
        torch.testing.assert_close(out_buf, ref_flat, atol=0.5, rtol=0.5)
        print("PASSED: layers.PagedAttention compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: layers.PagedAttention compile() disagrees with "
              f"forward()\n{e}")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_paged_attention_testmode()
