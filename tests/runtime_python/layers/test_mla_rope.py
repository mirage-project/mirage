"""Numerical test: ``layers.mla.MLARopeQ`` and ``MLARopeK`` via test_mode.

CAVEAT — known mismatch between forward() and compile():

The CATALOG ``forward()`` implements rotate-half RoPE (HuggingFace
style):  output[..., :half] = x[..., :half] * cos - x[..., half:] * sin,
output[..., half:] = x[..., half:] * cos + x[..., :half] * sin.

The compiled KERNEL (deepseek_mla_rope_sm100.cuh) implements GPT-J
interleaved RoPE:  output[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin,
output[..., 1::2] = x[..., 1::2] * cos + x[..., 0::2] * sin.

These produce mathematically different outputs for the SAME cos/sin
table, by design (DeepSeek V3 ships GPT-J interleaved RoPE; the catalog
forward() picks the wrong convention).

As a workaround we either:
* construct cos/sin tables that satisfy the rotate-half convention but
  permute Q/K accordingly (complex),
* OR test the compiled path against a GPT-J reference (validates the
  kernel but not the catalog's forward()).

We do the second here for the smoke half, and we XFAIL the catalog's
``forward()`` against ``compile()`` directly.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.mla.rope import MLARopeQ, MLARopeK
from mirage.mpk.persistent_kernel import PersistentKernel


def _build_gptj_cossin(seq_len, rope_dim, device):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim
    ))
    angles = torch.outer(pos, inv_freq)
    cos = angles.cos().repeat_interleave(2, dim=-1).to(torch.bfloat16)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(torch.bfloat16)
    return cos, sin


def _gptj_rotate(x_bf16, cos, sin):
    out = x_bf16.float().clone()
    x0 = x_bf16[..., 0::2].float()
    x1 = x_bf16[..., 1::2].float()
    c = cos[..., 0::2].float()
    s = sin[..., 0::2].float()
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x1 * c + x0 * s
    return out.to(torch.bfloat16)


def test_mla_rope_q_split_vs_gptj():
    """Validate ``MLARopeQ(variant='split').compile`` against GPT-J ref."""
    device = "cuda"
    torch.manual_seed(0)

    seq_len = 16
    num_heads = 4
    rope_dim = 64
    q_tile = 16

    q_pe = torch.randn(seq_len, num_heads * rope_dim,
                       dtype=torch.bfloat16, device=device) * 0.1
    q_pe_ref = q_pe.clone()
    cos, sin = _build_gptj_cossin(seq_len, rope_dim, device)

    # GPT-J reference: rotate each per-head 64-wide slice independently.
    q_view = q_pe_ref.view(seq_len, num_heads, rope_dim)
    for h in range(num_heads):
        q_view[:, h, :] = _gptj_rotate(q_view[:, h, :], cos, sin)

    m = MLARopeQ(num_heads=num_heads, variant="split", q_tile_size=q_tile).to(device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = seq_len
    params["max_num_batched_requests"] = 1
    params["max_seq_length"] = seq_len
    params["max_num_pages"] = 1
    params["page_size"] = seq_len
    # Seed meta tensors for position lookup.
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    prompt_lengths = torch.tensor([seq_len], dtype=torch.int32, device=device)
    tokens = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)
    params["meta_tensors"] = {
        "tokens": tokens,
        "step": step,
        "prompt_lengths": prompt_lengths,
        "qo_indptr_buffer": qo_indptr,
    }
    pk = PersistentKernel(**params)

    q_pe_dt = pk.attach_input(q_pe, name="ropeq_split_qpe")
    cos_dt = pk.attach_input(cos, name="ropeq_split_cos")
    sin_dt = pk.attach_input(sin, name="ropeq_split_sin")

    with pk.compile_scope():
        _ = m.compile(
            q_pe_dt, cos_dt, sin_dt,
            grid_dim=(1, num_heads, (seq_len + q_tile - 1) // q_tile),
            block_dim=(128, 1, 1),
        )

    print("Compiling MLARopeQ(split) test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    diff = (q_pe.float() - q_pe_ref.float()).abs().max().item()
    print(f"Max abs diff (kernel vs GPT-J ref): {diff}")
    try:
        torch.testing.assert_close(q_pe, q_pe_ref, atol=1e-2, rtol=1e-2)
        print("PASSED: MLARopeQ(split) compile() matches GPT-J reference")
    except AssertionError as e:
        print(f"FAILED: MLARopeQ(split) kernel vs GPT-J ref:\n{e}")
        pk.finalize()
        sys.exit(1)
    pk.finalize()


def test_mla_rope_k_vs_gptj():
    """Validate ``MLARopeK.compile`` against GPT-J ref. K-PE has no head axis."""
    device = "cuda"
    torch.manual_seed(1)

    seq_len = 16
    rope_dim = 64
    k_pe_stride = 128  # standalone k_pe defaults to row stride 128
    q_tile = 16

    k_pe = torch.zeros(seq_len, k_pe_stride, dtype=torch.bfloat16, device=device)
    k_pe[:, :rope_dim] = torch.randn(seq_len, rope_dim, dtype=torch.bfloat16,
                                     device=device) * 0.1
    k_pe_ref = k_pe.clone()
    cos, sin = _build_gptj_cossin(seq_len, rope_dim, device)
    k_pe_ref[:, :rope_dim] = _gptj_rotate(k_pe_ref[:, :rope_dim], cos, sin)

    m = MLARopeK(q_tile_size=q_tile).to(device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = seq_len
    params["max_num_batched_requests"] = 1
    params["max_seq_length"] = seq_len
    params["max_num_pages"] = 1
    params["page_size"] = seq_len
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    prompt_lengths = torch.tensor([seq_len], dtype=torch.int32, device=device)
    tokens = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)
    params["meta_tensors"] = {
        "tokens": tokens,
        "step": step,
        "prompt_lengths": prompt_lengths,
        "qo_indptr_buffer": qo_indptr,
    }
    pk = PersistentKernel(**params)

    k_pe_dt = pk.attach_input(k_pe, name="ropek_kpe")
    cos_dt = pk.attach_input(cos, name="ropek_cos")
    sin_dt = pk.attach_input(sin, name="ropek_sin")

    with pk.compile_scope():
        _ = m.compile(
            k_pe_dt, cos_dt, sin_dt,
            grid_dim=(1, 1, (seq_len + q_tile - 1) // q_tile),
            block_dim=(128, 1, 1),
        )

    print("Compiling MLARopeK test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    diff = (k_pe.float() - k_pe_ref.float()).abs().max().item()
    print(f"Max abs diff (kernel vs GPT-J ref): {diff}")
    try:
        torch.testing.assert_close(k_pe, k_pe_ref, atol=1e-2, rtol=1e-2)
        print("PASSED: MLARopeK compile() matches GPT-J reference")
    except AssertionError as e:
        print(f"FAILED: MLARopeK kernel vs GPT-J ref:\n{e}")
        pk.finalize()
        sys.exit(1)
    pk.finalize()


def test_mla_rope_forward_matches_gptj_ref():
    """``MLARopeQ.forward()`` and ``MLARopeK.forward()`` use the GPT-J
    interleaved rotation (matching the kernel convention). Verify both
    references produce identical output to an external GPT-J reference.
    """
    from mirage.mpk.layers.mla.rope import MLARopeQ, MLARopeK

    torch.manual_seed(0)
    T, H, D = 4, 2, 64
    q_pe = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    k_pe = torch.randn(T, D, dtype=torch.bfloat16, device="cuda")
    half = D // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    positions = torch.arange(T, dtype=torch.long, device="cuda")
    freqs = positions.float()[:, None] * inv_freq.to("cuda")[None, :]
    cos = freqs.cos().repeat_interleave(2, dim=-1).to(torch.bfloat16)
    sin = freqs.sin().repeat_interleave(2, dim=-1).to(torch.bfloat16)

    def gptj_rot(x):
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rot = torch.stack((-odd, even), dim=-1).flatten(-2)
        return rot

    rq = MLARopeQ(num_heads=H, variant="split").to("cuda")
    rk = MLARopeK().to("cuda")

    if D == 64:
        cos_e_q = cos[positions].unsqueeze(1)
        sin_e_q = sin[positions].unsqueeze(1)
    q_ref_external = (q_pe * cos_e_q) + (gptj_rot(q_pe) * sin_e_q)
    q_ref_module = rq.forward(q_pe, cos, sin, positions)
    diff_q = (q_ref_module.float() - q_ref_external.float()).abs().max().item()
    print(f"Q forward() vs GPT-J external ref: max abs diff = {diff_q}")
    assert diff_q < 1e-2, f"MLARopeQ.forward() mismatch: {diff_q}"

    cos_e_k = cos[positions]
    sin_e_k = sin[positions]
    k_ref_external = (k_pe * cos_e_k) + (gptj_rot(k_pe) * sin_e_k)
    k_ref_module = rk.forward(k_pe, cos, sin, positions)
    diff_k = (k_ref_module.float() - k_ref_external.float()).abs().max().item()
    print(f"K forward() vs GPT-J external ref: max abs diff = {diff_k}")
    assert diff_k < 1e-2, f"MLARopeK.forward() mismatch: {diff_k}"
    print("PASSED: MLARopeQ/K forward() now match the GPT-J interleaved "
          "convention used by deepseek_mla_rope_sm100.cuh.")


if __name__ == "__main__":
    test_mla_rope_q_split_vs_gptj()
    test_mla_rope_k_vs_gptj()
    test_mla_rope_forward_matches_gptj_ref()
    print("All MLA RoPE tests completed.")
