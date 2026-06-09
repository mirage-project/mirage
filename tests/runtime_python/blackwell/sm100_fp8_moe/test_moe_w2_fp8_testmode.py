"""
Test mode: FP8 MoE W2 group GEMM through PersistentKernel test_mode.

Validates `pk.moe_w2_fp8_layer` end-to-end (Python -> codegen -> nvcc ->
runtime) against the pure-PyTorch reference in pytorch_reference.py.

Tolerances: abs<2.0, rel<0.05  (FP8 is loose, BF16 accumulation).
"""

import os
import sys
import math
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

# Reference (self-contained — no circular import via test_fp8_moe_gemm)
from pytorch_reference import moe_w2_fp8_ref


# ================================================================
# Config
# ================================================================
NUM_EXPERTS = 64
NUM_TOPK = 8
HIDDEN_SIZE = 7168          # K (output dim of W2)
INTERMEDIATE_SIZE = 2048    # I (reduction dim of W2)
BATCH_SIZE = 16


# ================================================================
# FP8 quantization (block_k=128 along last dim)
# ================================================================
def quantize_fp8_3d(x):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    A, B, K = x.shape
    assert K % 128 == 0
    x_b = x.reshape(A, B, K // 128, 128)
    amax = x_b.abs().amax(dim=3)
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_fp8 = (x_b / scale.unsqueeze(3)).reshape(A, B, K).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


# ================================================================
# Routing — round-robin to keep things deterministic
# ================================================================
def make_routing(batch_size, num_experts, num_topk, device):
    routing = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    token_to_experts = {}
    for i in range(batch_size):
        experts = [(i * num_topk + s) % num_experts for s in range(num_topk)]
        token_to_experts[i] = experts
        for slot, e in enumerate(experts):
            routing[e, i] = slot + 1
    activated = [e for e in range(num_experts) if routing[e].any()]
    mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        mask[idx] = e
    mask[num_experts] = len(activated)
    return routing, mask, token_to_experts


def test_moe_w2_fp8_testmode():
    device = "cuda"
    torch.manual_seed(100)

    print(f"\n{'=' * 70}")
    print(f"Test mode: moe_w2_fp8_layer")
    print(f"  E={NUM_EXPERTS}, B={BATCH_SIZE}, I={INTERMEDIATE_SIZE}, "
          f"K={HIDDEN_SIZE}, topk={NUM_TOPK}")

    # Inputs
    input_val = torch.randn(BATCH_SIZE, NUM_TOPK, INTERMEDIATE_SIZE,
                            device=device) * 0.1
    weight_val = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                             device=device) / math.sqrt(INTERMEDIATE_SIZE)

    input_fp8, input_scale = quantize_fp8_3d(input_val)
    weight_fp8, weight_scale = quantize_fp8_3d(weight_val)
    routing, mask, token_to_experts = make_routing(
        BATCH_SIZE, NUM_EXPERTS, NUM_TOPK, device)

    output = torch.zeros(BATCH_SIZE, NUM_TOPK, HIDDEN_SIZE,
                         dtype=torch.bfloat16, device=device)

    # Build PK
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = BATCH_SIZE
    params["max_num_batched_requests"] = BATCH_SIZE
    pk = PersistentKernel(**params)

    i_fp8 = pk.attach_input(input_fp8, name="input_fp8")
    i_sc = pk.attach_input(input_scale, name="input_scale")
    w_fp8 = pk.attach_input(weight_fp8, name="weight_fp8")
    w_sc = pk.attach_input(weight_scale, name="weight_scale")
    rt = pk.attach_input(routing, name="routing_indices")
    mk = pk.attach_input(mask, name="mask")
    out = pk.attach_input(output, name="output")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    # 7168/14 = 512 = 4*MMA_M: known-good N-split for W2 (matches pipeline test).
    pk.moe_w2_fp8_layer(
        input_fp8=i_fp8, input_scale=i_sc,
        weight_fp8=w_fp8, weight_scale=w_sc,
        moe_routing_indices=rt, moe_mask=mk, output=out,
        grid_dim=(8, 14, 1), block_dim=block_dim,
    )

    print("Compiling...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # Reference
    ref = moe_w2_fp8_ref(input_fp8, input_scale, weight_fp8, weight_scale,
                         BATCH_SIZE, token_to_experts, use_ue8m0=True)

    diff = (output.float() - ref.float()).abs()
    max_abs = diff.max().item()
    max_rel = max_abs / max(ref.float().abs().max().item(), 1e-6)

    print(f"\nOutput[0, 0, :8]:    {output[0, 0, :8]}")
    print(f"Reference[0, 0, :8]: {ref[0, 0, :8]}")
    print(f"\nMax abs diff: {max_abs:.6f}, Max rel err: {max_rel:.6f}")

    passed = (max_abs < 2.0 and max_rel < 0.05)
    print(f"\n{'PASSED' if passed else 'FAILED'}: moe_w2_fp8_layer test_mode "
          f"(abs={max_abs:.4f}, rel={max_rel:.4f})")
    pk.finalize()
    assert passed, f"abs={max_abs} rel={max_rel}"


if __name__ == "__main__":
    test_moe_w2_fp8_testmode()
