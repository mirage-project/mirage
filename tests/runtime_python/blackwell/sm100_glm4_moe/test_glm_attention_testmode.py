"""Test-mode test for GLM-4.6 paged attention: per-head qk-norm (eps 1e-5)
fused with PARTIAL RoPE (rotary_dim 64 of head_dim 128) in
multitoken_paged_attention_sm100 (TASK_ATTN_SM100).

Two layers in one task graph over the same 8-token prefill request:
  - partial: rotary_dim=64, theta 1e6, qk_norm_eps=1e-5 (the GLM-4.6 config;
             exercises the new ROTARY_DIM template path: rotate dims 0-63
             pairing i <-> i+32, pass dims 64-127 through)
  - full:    default rotary_dim (=head_dim) and eps 1e-6 — regression guard
             that the ROTARY_DIM change leaves existing models bit-compatible

Both use GQA 8 q / 2 kv heads and are checked against the shared PyTorch
reference (causal attention, norm-then-rope, matching the HF
modeling_glm4_moe math).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import glm_attention_prefill_ref

HEAD_DIM = 128
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
GROUP = NUM_Q_HEADS // NUM_KV_HEADS
FUSED_DIM = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
T = 8  # prefill length == max_num_batched_tokens


def make_cos_sin(max_seq, rotary_dim, theta, device):
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
        / rotary_dim))
    angles = torch.outer(
        torch.arange(max_seq, dtype=torch.float32, device=device), inv_freq)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def fuse_qkv(q, k, v):
    """q [T, NQ, D], k/v [T, NKV, D] -> kv-head-interleaved [T, FUSED_DIM]."""
    rows = []
    for t in range(q.shape[0]):
        parts = []
        for g in range(NUM_KV_HEADS):
            parts.append(q[t, g * GROUP:(g + 1) * GROUP].reshape(-1))
            parts.append(k[t, g])
            parts.append(v[t, g])
        rows.append(torch.cat(parts))
    return torch.stack(rows).contiguous()


def main():
    torch.manual_seed(0)
    device = "cuda"

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["max_num_batched_tokens"] = T
    params["max_num_batched_requests"] = 1
    params["page_size"] = 64
    params["max_num_pages"] = 4
    params["max_seq_length"] = 256
    params["meta_tensors"] = {
        "prompt_lengths": torch.tensor([T], dtype=torch.int32, device=device),
    }
    pk = PersistentKernel(**params)

    # (name, rotary_dim_arg, table_rotary_dim, theta, eps)
    configs = [
        ("partial", 64, 64, 1e6, 1e-5),
        ("full", 0, HEAD_DIM, 1e4, 1e-6),
    ]

    cases = []
    for name, rd_arg, rd, theta, eps in configs:
        q = torch.randn(T, NUM_Q_HEADS, HEAD_DIM,
                        dtype=torch.bfloat16, device=device)
        k = torch.randn(T, NUM_KV_HEADS, HEAD_DIM,
                        dtype=torch.bfloat16, device=device)
        v = torch.randn(T, NUM_KV_HEADS, HEAD_DIM,
                        dtype=torch.bfloat16, device=device)
        qkv = fuse_qkv(q, k, v)
        qn_w = (0.5 + torch.rand(HEAD_DIM, device=device)).to(torch.bfloat16)
        kn_w = (0.5 + torch.rand(HEAD_DIM, device=device)).to(torch.bfloat16)
        cos, sin = make_cos_sin(params["max_seq_length"], rd, theta, device)
        k_cache = torch.zeros(params["max_num_pages"], params["page_size"],
                              NUM_KV_HEADS, HEAD_DIM,
                              dtype=torch.bfloat16, device=device)
        v_cache = torch.zeros_like(k_cache)
        out = torch.zeros(T, NUM_Q_HEADS * HEAD_DIM,
                          dtype=torch.bfloat16, device=device)

        dts = {}
        for tname, tt in (("qkv", qkv), ("k_cache", k_cache),
                          ("v_cache", v_cache), ("qn", qn_w), ("kn", kn_w),
                          ("cos", cos), ("sin", sin), ("out", out)):
            dts[tname] = pk.attach_input(tt, name=f"{name}_{tname}")

        pk.paged_attention_layer(
            input=dts["qkv"], k_cache=dts["k_cache"], v_cache=dts["v_cache"],
            q_norm=dts["qn"], k_norm=dts["kn"],
            cos_pos_embed=dts["cos"], sin_pos_embed=dts["sin"],
            output=dts["out"],
            grid_dim=(1, NUM_KV_HEADS, 1), block_dim=(128, 1, 1),
            rotary_dim=rd_arg, qk_norm_eps=eps,
        )
        cases.append((name, q, k, v, qn_w, kn_w, cos, sin, rd, eps, out))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    for name, q, k, v, qn_w, kn_w, cos, sin, rd, eps, out in cases:
        ref = glm_attention_prefill_ref(
            q, k, v, qn_w, kn_w, cos[:T], sin[:T], rd, eps=eps
        ).reshape(T, NUM_Q_HEADS * HEAD_DIM).to(torch.bfloat16)
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"[{name}] out max diff: {diff:.3e}")
        try:
            torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        except AssertionError as e:
            print(f"[{name}] FAILED: {e}")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("PASSED: GLM partial-RoPE paged attention matches reference")


if __name__ == "__main__":
    main()
