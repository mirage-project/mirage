"""DeepSeek MLA RoPE task correctness through MPK test_mode.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python \
        tests/runtime_python/blackwell/sm100_mla/test_deepseek_mla_rope_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _rotate_gptj(x, cos, sin):
    out = x.float().clone()
    x0 = x[..., 0::2].float()
    x1 = x[..., 1::2].float()
    c = cos[..., 0::2].float()
    s = sin[..., 0::2].float()
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x1 * c + x0 * s
    return out.to(torch.bfloat16)


def test_deepseek_mla_rope_testmode():
    device = "cuda"
    torch.manual_seed(7)

    seq_len = 32
    num_heads = 4
    fused_head_dim = 576
    rope_dim = 64
    k_pe_stride = 128
    q_tile = 16

    q_fused = (
        torch.randn(seq_len, num_heads * fused_head_dim,
                    dtype=torch.bfloat16, device=device) * 0.1
    )
    q_split = (
        torch.randn(seq_len, num_heads * rope_dim,
                    dtype=torch.bfloat16, device=device) * 0.1
    )
    k_pe = torch.zeros(seq_len, k_pe_stride, dtype=torch.bfloat16,
                       device=device)
    k_pe[:, :rope_dim] = (
        torch.randn(seq_len, rope_dim, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    q_fused_ref = q_fused.clone()
    q_split_ref = q_split.clone()
    k_pe_ref = k_pe.clone()

    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32)
        / rope_dim
    ))
    angles = torch.outer(pos, inv_freq)
    cos = angles.cos().repeat_interleave(2, dim=-1).to(torch.bfloat16)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(torch.bfloat16)

    q_fused_view = q_fused_ref.view(seq_len, num_heads, fused_head_dim)
    q_split_view = q_split_ref.view(seq_len, num_heads, rope_dim)
    for h in range(num_heads):
        q_tail = q_fused_view[:, h, fused_head_dim - rope_dim:]
        q_fused_view[:, h, fused_head_dim - rope_dim:] = _rotate_gptj(
            q_tail, cos, sin)
        q_split_view[:, h, :] = _rotate_gptj(q_split_view[:, h, :], cos, sin)
    k_pe_ref[:, :rope_dim] = _rotate_gptj(k_pe_ref[:, :rope_dim], cos, sin)

    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    prompt_lengths = torch.tensor([seq_len], dtype=torch.int32, device=device)
    tokens = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)

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
    params["meta_tensors"] = {
        "tokens": tokens,
        "step": step,
        "prompt_lengths": prompt_lengths,
        "qo_indptr_buffer": qo_indptr,
    }
    pk = PersistentKernel(**params)

    q_fused_dt = pk.attach_input(q_fused, name="q_fused")
    q_split_dt = pk.attach_input(q_split, name="q_split")
    k_pe_dt = pk.attach_input(k_pe, name="k_pe")
    cos_dt = pk.attach_input(cos, name="rope_cos")
    sin_dt = pk.attach_input(sin, name="rope_sin")

    pk.deepseek_mla_rope_q_layer(
        q_nope_pe=q_fused_dt,
        q_pe=q_split_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        num_heads=num_heads,
        has_split_q=True,
        grid_dim=(1, num_heads, (seq_len + q_tile - 1) // q_tile),
        block_dim=(128, 1, 1),
        q_tile_size=q_tile,
    )
    pk.deepseek_mla_rope_k_layer(
        k_pe=k_pe_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        grid_dim=(1, 1, (seq_len + q_tile - 1) // q_tile),
        block_dim=(128, 1, 1),
        q_tile_size=q_tile,
    )

    folder = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    torch.testing.assert_close(q_fused, q_fused_ref, rtol=0, atol=0)
    torch.testing.assert_close(q_split, q_split_ref, rtol=0, atol=0)
    torch.testing.assert_close(k_pe, k_pe_ref, rtol=0, atol=0)
    print("PASSED: deepseek_mla_rope_sm100 matches vLLM/SGLang GPT-J RoPE reference")
    pk.finalize()


if __name__ == "__main__":
    test_deepseek_mla_rope_testmode()
