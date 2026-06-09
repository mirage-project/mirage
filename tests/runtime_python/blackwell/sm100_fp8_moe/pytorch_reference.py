"""
PyTorch reference implementations for FP8 MoE layers (W13, W2).

Self-contained: includes the small UE8M0 / dequantization helpers
needed for the references so this file can be imported without
pulling test_fp8_moe_gemm.py (avoiding circular imports).

These match the layer signatures used by `pk.moe_w13_fp8_layer` and
`pk.moe_w2_fp8_layer` in the MPK persistent kernel.
"""

import torch


# ----------------------------------------------------------------
# Comparison metrics (shared by W13 / W2 test_mode files)
# ----------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two tensors flattened to 1D (float32)."""
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = af.norm() * bf.norm()
    if denom.item() == 0.0:
        return 1.0 if af.norm().item() == bf.norm().item() else 0.0
    return (af @ bf / denom).item()


def rel_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean relative error: mean(|a-b|) / mean(|b|)."""
    af = a.float()
    bf = b.float()
    denom = bf.abs().mean().item()
    return (af - bf).abs().mean().item() / max(denom, 1e-12)


# ----------------------------------------------------------------
# FP8 quantization (per-128-block float32 scale along last dim) — the
# layer's input/weight contract (kernel does the UE8M0 floor internally).
# ----------------------------------------------------------------

def quantize_fp8_2d(x: torch.Tensor):
    """Quantize a 2D tensor [M, K] to FP8 E4M3 + per-128-block float32 scale."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    M, K = x.shape
    assert K % 128 == 0
    x_b = x.reshape(M, K // 128, 128)
    amax = x_b.abs().amax(dim=2)
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_fp8 = (x_b / scale.unsqueeze(2)).reshape(M, K).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


def quantize_fp8_3d(x: torch.Tensor):
    """Quantize a 3D tensor [A, B, K] to FP8 E4M3 + per-128-block float32 scale."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    A, B, K = x.shape
    assert K % 128 == 0
    x_b = x.reshape(A, B, K // 128, 128)
    amax = x_b.abs().amax(dim=3)
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_fp8 = (x_b / scale.unsqueeze(3)).reshape(A, B, K).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


# ----------------------------------------------------------------
# Routing — round-robin so each token routes to NUM_TOPK distinct experts.
# Mirrors the kernel's contract: routing_indices[e, token] = topk_slot
# (1-indexed) if token routed to local expert e else 0; mask[0..count-1] are
# the activated local expert IDs and mask[num_local_experts] = count.
# ----------------------------------------------------------------

def make_routing(batch_size, num_local_experts, num_topk, device):
    routing = torch.zeros(num_local_experts, batch_size,
                          dtype=torch.int32, device=device)
    token_to_experts = {}
    for i in range(batch_size):
        experts = [(i * num_topk + s) % num_local_experts
                   for s in range(num_topk)]
        token_to_experts[i] = experts
        for slot, e in enumerate(experts):
            routing[e, i] = slot + 1
    activated = [e for e in range(num_local_experts) if routing[e].any()]
    mask = torch.zeros(num_local_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        mask[idx] = e
    mask[num_local_experts] = len(activated)
    return routing, mask, token_to_experts


# ----------------------------------------------------------------
# FP8 dequantization helpers (lifted from test_fp8_moe_gemm.py)
# ----------------------------------------------------------------

def _dequantize_fp8(x_fp8: torch.Tensor, scale: torch.Tensor, block_k: int = 128):
    """Dequantize FP8 with per-block float32 scales (last-dim block size = 128)."""
    shape = x_fp8.shape
    K_dim = shape[-1]
    num_blocks = K_dim // block_k
    x_blocks = x_fp8.reshape(*shape[:-1], num_blocks, block_k).float()
    return (x_blocks * scale.unsqueeze(-1)).reshape(*shape)


def _float32_to_ue8m0_approx(scale: torch.Tensor):
    """Convert float32 scale to UE8M0-approximated value (power-of-2 floor).

    Matches the kernel's conversion: ue8m0 = (__float_as_uint(sf) >> 23) & 0xFF
    """
    bits = scale.view(torch.int32)
    ue8m0 = (bits >> 23) & 0xFF
    return 2.0 ** (ue8m0.float() - 127.0)


# ----------------------------------------------------------------
# moe_w13_fp8_ref
# ----------------------------------------------------------------

def moe_w13_fp8_ref(input_fp8, input_scale, weight_fp8, weight_scale,
                    batch_size, token_to_experts, use_ue8m0=True):
    """Pure PyTorch reference for the FP8 MoE W13 group GEMM.

    Args:
        input_fp8:    [B, K] FP8 E4M3 (B = padded batch, K = hidden_size)
        input_scale:  [B, K/128] float32 per-128-block scales
        weight_fp8:   [E, 2*I, K] FP8 E4M3 (per-expert W13 = concat[gate, up])
        weight_scale: [E, 2*I, K/128] float32 per-128-block scales
        batch_size:   number of *active* tokens (<= B)
        token_to_experts: dict[int, list[int]] token_idx -> list of expert ids
                          (length = num_topk)
        use_ue8m0:    if True, round scales to UE8M0 (power-of-2) — matches kernel

    Returns:
        output: [B, num_topk, 2*I] bfloat16
    """
    if use_ue8m0:
        i_scale = _float32_to_ue8m0_approx(input_scale)
        w_scale = _float32_to_ue8m0_approx(weight_scale)
    else:
        i_scale = input_scale
        w_scale = weight_scale

    input_deq = _dequantize_fp8(input_fp8, i_scale).bfloat16()
    B = input_fp8.shape[0]
    OUTPUT_SIZE = weight_fp8.shape[1]
    # num_topk: derive from token_to_experts (all entries equal length)
    num_topk = len(next(iter(token_to_experts.values()))) if token_to_experts else 1

    output = torch.zeros(B, num_topk, OUTPUT_SIZE,
                         dtype=torch.bfloat16, device=input_fp8.device)

    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            w_deq = _dequantize_fp8(weight_fp8[e], w_scale[e]).bfloat16()
            output[i, slot] = (input_deq[i:i+1] @ w_deq.T).squeeze(0)

    return output


# ----------------------------------------------------------------
# moe_w2_fp8_ref
# ----------------------------------------------------------------

def moe_w2_fp8_ref(input_fp8_3d, input_scale_3d, weight_fp8, weight_scale,
                   batch_size, token_to_experts, use_ue8m0=True):
    """Pure PyTorch reference for the FP8 MoE W2 group GEMM.

    Args:
        input_fp8_3d:   [B, num_topk, I] FP8 E4M3 (post-SiLU·Mul output, quantized)
        input_scale_3d: [B, num_topk, I/128] float32 per-128-block scales
        weight_fp8:     [E, K, I] FP8 E4M3 (per-expert W2; K = hidden, I = intermediate)
        weight_scale:   [E, K, I/128] float32 per-128-block scales
        batch_size:     number of *active* tokens (<= B)
        token_to_experts: dict[int, list[int]] token_idx -> list of expert ids
        use_ue8m0:      if True, round scales to UE8M0 (power-of-2) — matches kernel

    Returns:
        output: [B, num_topk, K] bfloat16
    """
    if use_ue8m0:
        i_scale = _float32_to_ue8m0_approx(input_scale_3d)
        w_scale = _float32_to_ue8m0_approx(weight_scale)
    else:
        i_scale = input_scale_3d
        w_scale = weight_scale

    # Dequantize 3D input [B, topk, I]
    B, T, I = input_fp8_3d.shape
    input_deq = _dequantize_fp8(
        input_fp8_3d.reshape(B * T, I), i_scale.reshape(B * T, I // 128)
    ).reshape(B, T, I).bfloat16()

    W2_OUTPUT_SIZE = weight_fp8.shape[1]
    output = torch.zeros(B, T, W2_OUTPUT_SIZE,
                         dtype=torch.bfloat16, device=input_fp8_3d.device)

    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            w_deq = _dequantize_fp8(weight_fp8[e], w_scale[e]).bfloat16()
            output[i, slot] = (input_deq[i, slot:slot+1] @ w_deq.T).squeeze(0)

    return output
