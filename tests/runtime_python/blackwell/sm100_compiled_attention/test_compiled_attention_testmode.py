"""Test-mode gate for the COMPILED attention path (prep -> generated core ->
finalize) at real Qwen3 head shapes (16 q / 8 kv heads, head_dim 128).

One request, one T-token prefill step. The hybrid path appends every new
token's K/V (qk-normed + roped) to the cache but computes attention only for
the LAST token -- the only row decode-only generation consumes -- so the
reference is full attention for token T-1 over positions 0..T-1.
"""
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import mirage as mi
from mirage import PersistentKernel

NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
GROUP = NUM_Q_HEADS // NUM_KV_HEADS
FUSED_DIM = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
T = 8
S_MAX = 256


def make_cos_sin(max_seq, rotary_dim, theta, device):
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
        / rotary_dim))
    angles = torch.outer(
        torch.arange(max_seq, dtype=torch.float32, device=device), inv_freq)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def fuse_qkv(q, k, v):
    rows = []
    for t in range(q.shape[0]):
        parts = []
        for g in range(NUM_KV_HEADS):
            parts.append(q[t, g * GROUP:(g + 1) * GROUP].reshape(-1))
            parts.append(k[t, g])
            parts.append(v[t, g])
        rows.append(torch.cat(parts))
    return torch.stack(rows).contiguous()


def norm_rope_ref(x, w, cos, sin, eps):
    """x [*, D] fp32; RMSNorm then NeoX rope with cos/sin [*, D]."""
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    xf = xf * r * w.float()
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    c, s = cos.float(), sin.float()
    out = torch.cat([x1 * c[..., :half] - x2 * s[..., :half],
                     x2 * c[..., half:] + x1 * s[..., half:]], dim=-1)
    return out


def main():
    torch.manual_seed(0)
    device = "cuda"
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["max_num_batched_tokens"] = T
    params["max_num_batched_requests"] = 1
    params["page_size"] = S_MAX
    params["max_num_pages"] = 1
    params["max_seq_length"] = S_MAX
    params["meta_tensors"] = {
        "prompt_lengths": torch.tensor([T], dtype=torch.int32, device=device),
    }
    pk = PersistentKernel(**params)

    eps = 1e-6
    q = torch.randn(T, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16,
                    device=device)
    k = torch.randn(T, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16,
                    device=device)
    v = torch.randn(T, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16,
                    device=device)
    qkv = fuse_qkv(q, k, v)
    qn_w = (0.5 + torch.rand(HEAD_DIM, device=device)).to(torch.bfloat16)
    kn_w = (0.5 + torch.rand(HEAD_DIM, device=device)).to(torch.bfloat16)
    cos, sin = make_cos_sin(S_MAX, HEAD_DIM, 1e4, device)
    k_cache = torch.zeros(1, S_MAX, NUM_KV_HEADS, HEAD_DIM,
                          dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros_like(k_cache)
    out = torch.zeros(T, NUM_Q_HEADS * HEAD_DIM, dtype=torch.bfloat16,
                      device=device)
    q_staged_t = torch.zeros(NUM_KV_HEADS * 1, 8, HEAD_DIM,
                             dtype=torch.bfloat16, device=device)
    mask_t = torch.full((NUM_KV_HEADS * 1, 1, S_MAX), -30000.0,
                        dtype=torch.bfloat16, device=device)
    pad_t = torch.zeros(NUM_KV_HEADS * 1, 8, HEAD_DIM, dtype=torch.bfloat16,
                        device=device)
    kt_staged_t = torch.zeros(NUM_KV_HEADS * 1, HEAD_DIM, S_MAX,
                              dtype=torch.bfloat16, device=device)
    v_staged_t = torch.zeros(NUM_KV_HEADS * 1, S_MAX, HEAD_DIM,
                             dtype=torch.bfloat16, device=device)

    dts = {}
    for name, tt in (("qkv", qkv), ("k_cache", k_cache),
                     ("v_cache", v_cache), ("qn", qn_w), ("kn", kn_w),
                     ("cos", cos), ("sin", sin), ("out", out),
                     ("q_staged", q_staged_t), ("mask", mask_t),
                     ("pad", pad_t), ("kt_staged", kt_staged_t),
                     ("v_staged", v_staged_t)):
        dts[name] = pk.attach_input(tt, name=f"ca_{name}")
    pk.attention_prep_layer(
        input=dts["qkv"], k_cache=dts["k_cache"], v_cache=dts["v_cache"],
        q_norm=dts["qn"], k_norm=dts["kn"],
        cos_pos_embed=dts["cos"], sin_pos_embed=dts["sin"],
        q_staged=dts["q_staged"], mask_staged=dts["mask"],
        kt_staged=dts["kt_staged"], v_staged=dts["v_staged"],
        grid_dim=(1, NUM_KV_HEADS, 1), block_dim=(128, 1, 1),
        qk_norm_eps=eps,
    )
    pk.generated_attention_layer(
        q_staged=dts["q_staged"], kt_staged=dts["kt_staged"],
        v_staged=dts["v_staged"], mask_staged=dts["mask"],
        attn_pad=dts["pad"],
        grid_dim=(NUM_KV_HEADS * 1, 1, 1), block_dim=(256, 1, 1),
    )
    pk.attention_finalize_layer(
        attn_pad=dts["pad"], output=dts["out"],
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
    )

    print("Compiling test kernel...")
    # output_dir=None -> temp dir; compiling into the source tree leaves
    # per-rank artifacts behind.
    pk.compile(output_dir=None)
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Prep contract checks (host-visible staging buffers).
    assert q_staged_t[:, 2:].abs().max().item() == 0.0, \
        "q_staged pad rows must be zeroed"
    assert int((mask_t[0, 0] == 0).sum().item()) == T, \
        "mask must have exactly the new positions live"

    # Reference: last token attends over all T positions.
    kn = norm_rope_ref(k.float(), kn_w, cos[:T, None, :].expand(T, NUM_KV_HEADS, HEAD_DIM),
                       sin[:T, None, :].expand(T, NUM_KV_HEADS, HEAD_DIM), eps)  # [T, KV, D]
    qn = norm_rope_ref(q[T - 1].float(), qn_w,
                       cos[T - 1].expand(NUM_Q_HEADS, HEAD_DIM),
                       sin[T - 1].expand(NUM_Q_HEADS, HEAD_DIM), eps)  # [NQ, D]
    ref_rows = []
    for h in range(NUM_Q_HEADS):
        kvh = h // GROUP
        logits = (qn[h] @ kn[:, kvh].T) / (HEAD_DIM ** 0.5)  # [T]
        p = torch.softmax(logits, dim=-1)
        ref_rows.append(p @ v[:, kvh].float())
    ref = torch.cat(ref_rows)  # [NQ * D]

    got = out[T - 1].float()
    diff = (got - ref).abs().max().item()
    rel = diff / ref.abs().max().item()
    print(f"last-token attention: max abs diff {diff:.4e}  rel {rel:.4e}")
    pk.finalize()
    if rel > 2e-2:
        print("FAILED")
        sys.exit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
