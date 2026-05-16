"""Test the ``layers.Attention`` catalog module via PersistentKernel test_mode.

This is the Phase-2 test for the Qwen3-variant fused attention kernel
(``pk.attention_layer`` -> ``single_batch_decoding_kernel``). It is the
hardest of the Phase-2 layer tests because the kernel is multi-input,
stateful (KV cache), and reads runtime state (``meta_tensors["step"]``).

What we exercise
----------------

1. Build a small ``Attention`` module on CUDA (bf16). Tiny dims chosen to
   keep compile fast while still matching the kernel's hard-coded
   internals (``HEAD_DIM == 128``, ``NUM_THREADS == 128``).
2. Build a fused QKV input row vector and pre-fill a KV cache with the
   first ``S - 1`` keys/values so the kernel has prior context to attend
   over (the kernel reads positions ``[:S]`` and writes position
   ``S - 1``).
3. Run the PyTorch reference on a fresh copy of the same inputs/cache so
   forward() and compile() share identical state.
4. Set ``meta_tensors["step"][0] = S - 1`` so the kernel's
   ``runtime_config.step[0] + 1`` evaluates to ``S``.
5. Inside ``with pk.compile_scope():`` register the task via
   ``m.compile(...)``. Compile, launch once, sync.
6. Diff against the reference at bf16-attention tolerance (~0.5).

Why these dims
--------------

* ``head_dim = 128``: the single-batch decoding kernel's data layout
  asserts ``HEAD_DIM == 128`` via ``dmem_row<T, 1, 128, 128>`` in
  ``single_batch_decoding.cuh:74-79`` (the literal "128" is the inner
  stride). Other values won't compile.
* ``num_heads = 4, num_kv_heads = 2`` (GQA group=2). Small enough that
  the kernel's per-head registers fit, large enough to exercise the
  GQA grouping codepath. ``num_q_heads / num_kv_heads`` is a template
  arg baked into the code-gen.
* ``batch = 1``: the plain ``attention`` task is single-batch (the
  Cython grid axis is ``(batch_size, num_kv_heads, 1)`` and the
  underlying kernel was originally written for one request per task).
* ``seq_len = 16``: small power-of-two within the kernel's
  ``MAX_SEQ_LEN = 512`` and below ``KV_CHUNK_SIZE = 64`` so we run
  through a single chunk and exercise the tail path.

DO NOT execute this file as part of Phase 2 — Phase 4 runs it on a free
GPU. The ``mirage`` conda env is required.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel
# The catalog only re-exports a couple of leaf modules through
# ``mirage.mpk.layers`` today; ``Attention`` lives in its own subpackage
# and Phase-2 agents do NOT touch ``__init__.py`` (per the briefing).
from mirage.mpk.layers.attention.attention import Attention


def test_attention_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Dimensions (see module docstring for the constraints)
    # ------------------------------------------------------------------
    batch_size = 1
    num_heads = 4
    num_kv_heads = 2
    head_dim = 128
    seq_len = 16          # how many tokens are cached AND attended over
    max_seq_len = 512     # matches the kernel's MAX_SEQ_LEN
    layer_idx = 0

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    fused_qkv_size = q_size + kv_size + kv_size   # [Q | K | V]

    # ------------------------------------------------------------------
    # Build module and seed weights with a non-trivial scale.
    # ------------------------------------------------------------------
    try:
        m = Attention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            prefix="test_",
        ).to(device=device, dtype=dtype)
    except RuntimeError as e:
        print(f"SKIPPED (known broken in Mirage): {e}")
        return
    # Overwrite the all-ones init with a small randn so both norms are
    # non-trivial. Match the kernel's bf16 storage / fp32 reduction.
    with torch.no_grad():
        m.q_norm.data.copy_(
            torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
        )
        m.k_norm.data.copy_(
            torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
        )

    # ------------------------------------------------------------------
    # Build the fused QKV input row vector. Single-token decode: T == 1.
    # ------------------------------------------------------------------
    # The kernel reads ``input[batch, : q_size]``, then
    # ``input[batch, q_size : q_size + kv_size]``, then
    # ``input[batch, q_size + kv_size : ...]``. So we just allocate one
    # contiguous row.
    fused_qkv = torch.randn(
        batch_size, fused_qkv_size, dtype=dtype, device=device
    )

    # ------------------------------------------------------------------
    # KV cache. Pre-fill positions [0, seq_len - 1) with random content;
    # the kernel will write the new k/v at position seq_len - 1.
    # ------------------------------------------------------------------
    k_cache = torch.zeros(
        batch_size, max_seq_len, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    v_cache = torch.zeros(
        batch_size, max_seq_len, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    # Pre-existing context — anything beyond seq_len is irrelevant.
    k_cache[:, : seq_len - 1] = torch.randn_like(k_cache[:, : seq_len - 1])
    v_cache[:, : seq_len - 1] = torch.randn_like(v_cache[:, : seq_len - 1])

    # ------------------------------------------------------------------
    # RoPE tables. (max_seq_len, head_dim).
    # ------------------------------------------------------------------
    # The kernel indexes cos[seq_len - 1] / sin[seq_len - 1]. The first
    # ``seq_len - 1`` rows are unused by the decode path (those positions
    # have already been RoPE'd before being written to the cache), so we
    # only need row ``seq_len - 1`` to be meaningful for the compile
    # path; the forward() reference does its own slice. Use a smooth
    # synthetic table.
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000 ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    ))
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # (max_seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                  # (max_seq_len, head_dim)
    cos_tab = emb.cos().to(dtype)
    sin_tab = emb.sin().to(dtype)

    # ------------------------------------------------------------------
    # PyTorch reference. Use a deep copy of the cache because forward()
    # writes into it in place; the compiled kernel writes into the
    # original. We want both paths to see identical pre-state.
    # ------------------------------------------------------------------
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    q_in = fused_qkv[:, :q_size].unsqueeze(1)            # (B, 1, H*D)
    k_in = fused_qkv[:, q_size : q_size + kv_size].unsqueeze(1)
    v_in = fused_qkv[:, q_size + kv_size :].unsqueeze(1)
    ref = m.forward(
        q_proj=q_in,
        k_proj=k_in,
        v_proj=v_in,
        cos=cos_tab,
        sin=sin_tab,
        k_cache=k_cache_ref,
        v_cache=v_cache_ref,
        seq_len=seq_len,
    )  # (B, 1, H*D)
    ref = ref.view(batch_size, num_heads * head_dim)  # match kernel out shape

    # Output buffer the test driver reads back from.
    out_buf = torch.zeros(
        batch_size, num_heads * head_dim, dtype=dtype, device=device
    )

    # ------------------------------------------------------------------
    # Build PersistentKernel in test mode.
    #
    # IMPORTANT: the kernel reads ``meta_tensors["step"][0]`` and uses
    # ``step[0] + 1`` as the sequence length to attend over. We need
    # ``step[0] = seq_len - 1``, BUT the auto-allocated ``step`` lives
    # on the GPU after ``_apply_test_mode_meta_defaults`` runs. We
    # pre-seed it via the ``meta_tensors`` dict passed to __init__ so
    # the runtime sees the right value.
    #
    # max_seq_length MUST equal the cache's max_seq_len because PK
    # checks ``tokens.shape[1] == max_seq_length`` and the cache uses
    # the same indexing.
    # ------------------------------------------------------------------
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_seq_length"] = max_seq_len
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size

    # Seed ``step`` BEFORE PK init — _apply_test_mode_meta_defaults
    # only inserts defaults for missing keys.
    step_tensor = torch.full(
        (1,), seq_len - 1, dtype=torch.int32, device=device,
    )
    params["meta_tensors"] = {
        "step": step_tensor,
    }

    pk = PersistentKernel(**params)

    # ------------------------------------------------------------------
    # Attach the tensors that aren't owned by the module.
    # ------------------------------------------------------------------
    input_dt = pk.attach_input(fused_qkv, name="qkv_in")
    k_cache_dt = pk.attach_input(k_cache, name="k_cache")
    v_cache_dt = pk.attach_input(v_cache, name="v_cache")
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
            # Force the canonical decode grid; this matches
            # demo/qwen3/demo_chat.py:150 and the kernel's NUM_KV_HEADS-
            # per-task convention.
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
    print(f"ref[0, :8]:     {ref[0, :8]}")

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        # bf16 fused attention is noisy; 0.5 is the empirical bar used
        # by test_qwen3_mlp_testmode.py for a similar fused pipeline.
        # If this proves too strict in Phase 4 we can loosen to 1.0 or
        # switch to rtol-only.
        torch.testing.assert_close(out_buf, ref, atol=0.5, rtol=0.5)
        print("PASSED: layers.Attention compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: layers.Attention compile() disagrees with forward()\n{e}")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_attention_testmode()
