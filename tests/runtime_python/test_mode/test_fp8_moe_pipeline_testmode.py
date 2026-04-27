"""
Test: FP8 MoE pipeline layers via PersistentKernel test_mode.

Tests the FP8 block-scaled MoE pipeline with DeepSeek V3 configuration:
  1. test_moe_w13_fp8:          W13 FP8 group GEMM only
  2. test_moe_w2_fp8:           W2 FP8 group GEMM only
  3. test_moe_w13_silu_mul:     W13 FP8 + SiLU-Mul pipeline
  4. test_moe_w13_silu_mul_w2:  Full W13 FP8 + SiLU-Mul + W2 BF16 pipeline (single compile)

Run:
    python tests/runtime_python/test_mode/test_fp8_moe_pipeline_testmode.py
"""

import torch
import sys
import os
import math

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


# ================================================================
# FP8 quantization utilities
# ================================================================

def quantize_fp8_2d(x):
    """Quantize [M, K] to FP8 E4M3 with per-row per-128-K block scaling."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    M, K = x.shape
    assert K % 128 == 0
    x_blocked = x.reshape(M, K // 128, 128)
    amax = x_blocked.abs().amax(dim=2)
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_scaled = x_blocked / scale.unsqueeze(2)
    x_fp8 = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


def quantize_fp8_3d(x):
    """Quantize [A, B, K] to FP8 E4M3 with per-element per-128-K block scaling."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    A, B, K = x.shape
    assert K % 128 == 0
    x_blocked = x.reshape(A, B, K // 128, 128)
    amax = x_blocked.abs().amax(dim=3)
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_scaled = x_blocked / scale.unsqueeze(3)
    x_fp8 = x_scaled.reshape(A, B, K).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


def float32_to_ue8m0_approx(scale):
    """Convert float32 scale to UE8M0-approximated value (power-of-2 floor)."""
    bits = scale.view(torch.int32)
    ue8m0 = (bits >> 23) & 0xFF
    return 2.0 ** (ue8m0.float() - 127.0)


def dequantize_fp8_2d(x_fp8, scale):
    M, K = x_fp8.shape
    x = x_fp8.float().reshape(M, K // 128, 128)
    return (x * scale.unsqueeze(2)).reshape(M, K)


def dequantize_fp8_3d(x_fp8, scale):
    A, B, K = x_fp8.shape
    x = x_fp8.float().reshape(A, B, K // 128, 128)
    return (x * scale.unsqueeze(3)).reshape(A, B, K)


# ================================================================
# Common setup
# ================================================================

NUM_EXPERTS = 64
NUM_TOPK = 8
HIDDEN_SIZE = 7168        # K
INTERMEDIATE_SIZE = 2048  # I
N_W13 = 2 * INTERMEDIATE_SIZE  # 4096
BATCH_SIZE = 16


def make_routing(batch_size, num_experts, num_topk, device):
    """Round-robin routing: token i -> experts (i*topk+s) % E."""
    routing = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    for i in range(batch_size):
        for s in range(num_topk):
            eid = (i * num_topk + s) % num_experts
            routing[eid, i] = s + 1

    activated = [e for e in range(num_experts) if routing[e].any()]
    mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        mask[idx] = e
    mask[num_experts] = len(activated)
    return routing, mask


def make_pk(**extra_params):
    """Create a PersistentKernel with test_mode defaults."""
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=BATCH_SIZE,
        max_num_batched_requests=BATCH_SIZE,
    )
    params.update(extra_params)
    return PersistentKernel(**params)


# ================================================================
# Test 1: W13 FP8 only
# ================================================================

def test_moe_w13_fp8():
    """Test FP8 MoE W13 group GEMM: [B, K] @ [E, 2I, K].T -> [B, topk, 2I]."""
    device = "cuda"
    torch.manual_seed(42)

    print(f"\n{'='*70}")
    print(f"Test: FP8 MoE W13 Group GEMM")
    print(f"  E={NUM_EXPERTS}, B={BATCH_SIZE}, K={HIDDEN_SIZE}, N={N_W13}, topk={NUM_TOPK}")

    input_val = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device) * 0.1
    weight_val = torch.randn(NUM_EXPERTS, N_W13, HIDDEN_SIZE, device=device) / math.sqrt(HIDDEN_SIZE)

    input_fp8, input_scale = quantize_fp8_2d(input_val)
    weight_fp8, weight_scale = quantize_fp8_3d(weight_val)
    routing, mask = make_routing(BATCH_SIZE, NUM_EXPERTS, NUM_TOPK, device)
    output = torch.zeros(BATCH_SIZE, NUM_TOPK, N_W13, dtype=torch.bfloat16, device=device)

    # Reference
    input_deq = dequantize_fp8_2d(input_fp8, float32_to_ue8m0_approx(input_scale))
    weight_deq = dequantize_fp8_3d(weight_fp8, float32_to_ue8m0_approx(weight_scale))
    ref = torch.zeros_like(output, dtype=torch.float32)
    for i in range(BATCH_SIZE):
        for s in range(NUM_TOPK):
            eid = (i * NUM_TOPK + s) % NUM_EXPERTS
            ref[i, s] = input_deq[i] @ weight_deq[eid].T
    ref = ref.bfloat16()

    # Build PK
    pk = make_pk()
    i_fp8 = pk.attach_input(input_fp8, name="input_fp8")
    i_sc = pk.attach_input(input_scale, name="input_scale")
    w_fp8 = pk.attach_input(weight_fp8, name="weight_fp8")
    w_sc = pk.attach_input(weight_scale, name="weight_scale")
    rt = pk.attach_input(routing, name="routing_indices")
    mk = pk.attach_input(mask, name="mask")
    out = pk.attach_input(output, name="output")

    pk.moe_w13_fp8_layer(
        input_fp8=i_fp8, input_scale=i_sc,
        weight_fp8=w_fp8, weight_scale=w_sc,
        moe_routing_indices=rt, moe_mask=mk, output=out,
        grid_dim=(8, 16, 1), block_dim=(256, 1, 1),
    )

    print("Compiling...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"\nOutput[0, 0, :8]:    {output[0, 0, :8]}")
    print(f"Reference[0, 0, :8]: {ref[0, 0, :8]}")
    max_abs = (output.float() - ref.float()).abs().max().item()
    max_rel = max_abs / max(ref.float().abs().max().item(), 1e-6)
    print(f"\nMax abs diff: {max_abs:.6f}, Max rel err: {max_rel:.6f}")

    passed = max_rel < 0.1
    print(f"\n{'PASSED' if passed else 'FAILED'}: FP8 MoE W13 group GEMM")
    pk.finalize()
    return passed


# ================================================================
# Test 2: W2 FP8 only
# ================================================================

def test_moe_w2_fp8():
    """Test FP8 MoE W2 group GEMM: [B, topk, I] @ [E, K, I].T -> [B, topk, K]."""
    device = "cuda"
    torch.manual_seed(100)

    print(f"\n{'='*70}")
    print(f"Test: FP8 MoE W2 Group GEMM")
    print(f"  E={NUM_EXPERTS}, B={BATCH_SIZE}, I={INTERMEDIATE_SIZE}, K={HIDDEN_SIZE}, topk={NUM_TOPK}")

    input_val = torch.randn(BATCH_SIZE, NUM_TOPK, INTERMEDIATE_SIZE, device=device) * 0.1
    weight_val = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device) / math.sqrt(INTERMEDIATE_SIZE)

    input_fp8, input_scale = quantize_fp8_3d(input_val)
    weight_fp8, weight_scale = quantize_fp8_3d(weight_val)
    routing, mask = make_routing(BATCH_SIZE, NUM_EXPERTS, NUM_TOPK, device)
    output = torch.zeros(BATCH_SIZE, NUM_TOPK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

    # Reference
    input_deq = dequantize_fp8_3d(input_fp8, float32_to_ue8m0_approx(input_scale))
    weight_deq = dequantize_fp8_3d(weight_fp8, float32_to_ue8m0_approx(weight_scale))
    ref = torch.zeros_like(output, dtype=torch.float32)
    for i in range(BATCH_SIZE):
        for s in range(NUM_TOPK):
            eid = (i * NUM_TOPK + s) % NUM_EXPERTS
            ref[i, s] = input_deq[i, s] @ weight_deq[eid].T
    ref = ref.bfloat16()

    # Build PK
    pk = make_pk()
    i_fp8 = pk.attach_input(input_fp8, name="input_fp8")
    i_sc = pk.attach_input(input_scale, name="input_scale")
    w_fp8 = pk.attach_input(weight_fp8, name="weight_fp8")
    w_sc = pk.attach_input(weight_scale, name="weight_scale")
    rt = pk.attach_input(routing, name="routing_indices")
    mk = pk.attach_input(mask, name="mask")
    out = pk.attach_input(output, name="output")

    # 7168/14 = 512 = 4*MMA_M: good N-split for W2
    pk.moe_w2_fp8_layer(
        input_fp8=i_fp8, input_scale=i_sc,
        weight_fp8=w_fp8, weight_scale=w_sc,
        moe_routing_indices=rt, moe_mask=mk, output=out,
        grid_dim=(8, 14, 1), block_dim=(256, 1, 1),
    )

    print("Compiling...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"\nOutput[0, 0, :8]:    {output[0, 0, :8]}")
    print(f"Reference[0, 0, :8]: {ref[0, 0, :8]}")
    max_abs = (output.float() - ref.float()).abs().max().item()
    max_rel = max_abs / max(ref.float().abs().max().item(), 1e-6)
    print(f"\nMax abs diff: {max_abs:.6f}, Max rel err: {max_rel:.6f}")

    passed = max_rel < 0.1
    print(f"\n{'PASSED' if passed else 'FAILED'}: FP8 MoE W2 group GEMM")
    pk.finalize()
    return passed


# ================================================================
# Test 3: W13 FP8 + SiLU-Mul
# ================================================================

def test_moe_w13_silu_mul():
    """Test W13 FP8 -> SiLU-Mul pipeline: [B,K] -> [B,topk,2I] -> [B,topk,I]."""
    device = "cuda"
    torch.manual_seed(200)

    print(f"\n{'='*70}")
    print(f"Test: FP8 MoE W13 + SiLU-Mul pipeline")
    print(f"  E={NUM_EXPERTS}, B={BATCH_SIZE}, K={HIDDEN_SIZE}, I={INTERMEDIATE_SIZE}, topk={NUM_TOPK}")

    input_val = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device) * 0.1
    weight_val = torch.randn(NUM_EXPERTS, N_W13, HIDDEN_SIZE, device=device) / math.sqrt(HIDDEN_SIZE)

    input_fp8, input_scale = quantize_fp8_2d(input_val)
    weight_fp8, weight_scale = quantize_fp8_3d(weight_val)
    routing, mask = make_routing(BATCH_SIZE, NUM_EXPERTS, NUM_TOPK, device)

    w13_out = torch.zeros(BATCH_SIZE, NUM_TOPK, N_W13, dtype=torch.bfloat16, device=device)
    silu_out = torch.zeros(BATCH_SIZE, NUM_TOPK, INTERMEDIATE_SIZE, dtype=torch.bfloat16, device=device)

    # Reference
    input_deq = dequantize_fp8_2d(input_fp8, float32_to_ue8m0_approx(input_scale))
    weight_deq = dequantize_fp8_3d(weight_fp8, float32_to_ue8m0_approx(weight_scale))
    ref_w13 = torch.zeros(BATCH_SIZE, NUM_TOPK, N_W13, dtype=torch.float32, device=device)
    for i in range(BATCH_SIZE):
        for s in range(NUM_TOPK):
            eid = (i * NUM_TOPK + s) % NUM_EXPERTS
            ref_w13[i, s] = input_deq[i] @ weight_deq[eid].T
    ref_gate = ref_w13[:, :, :INTERMEDIATE_SIZE]
    ref_up = ref_w13[:, :, INTERMEDIATE_SIZE:]
    ref_silu = (torch.nn.functional.silu(ref_gate) * ref_up).bfloat16()

    # Build PK
    qo_indptr = torch.zeros(BATCH_SIZE + 1, dtype=torch.int32, device=device)
    qo_indptr[BATCH_SIZE] = BATCH_SIZE
    pk = make_pk(meta_tensors={"qo_indptr_buffer": qo_indptr})

    i_fp8 = pk.attach_input(input_fp8, name="input_fp8")
    i_sc = pk.attach_input(input_scale, name="input_scale")
    w_fp8 = pk.attach_input(weight_fp8, name="weight_fp8")
    w_sc = pk.attach_input(weight_scale, name="weight_scale")
    rt = pk.attach_input(routing, name="routing_indices")
    mk = pk.attach_input(mask, name="mask")
    w13_dt = pk.attach_input(w13_out, name="w13_out")
    silu_dt = pk.attach_input(silu_out, name="silu_out")

    pk.moe_w13_fp8_layer(
        input_fp8=i_fp8, input_scale=i_sc,
        weight_fp8=w_fp8, weight_scale=w_sc,
        moe_routing_indices=rt, moe_mask=mk, output=w13_dt,
        grid_dim=(8, 16, 1), block_dim=(256, 1, 1),
    )
    pk.moe_silu_mul_layer(
        input=w13_dt, output=silu_dt,
        grid_dim=(BATCH_SIZE, NUM_TOPK, 1), block_dim=(256, 1, 1),
    )

    print("Compiling...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"\nSiLU-Mul[0, 0, :8]:  {silu_out[0, 0, :8]}")
    print(f"Reference[0, 0, :8]: {ref_silu[0, 0, :8]}")
    max_abs = (silu_out.float() - ref_silu.float()).abs().max().item()
    max_rel = max_abs / max(ref_silu.float().abs().max().item(), 1e-6)
    print(f"\nMax abs diff: {max_abs:.6f}, Max rel err: {max_rel:.6f}")

    passed = max_rel < 0.1
    print(f"\n{'PASSED' if passed else 'FAILED'}: W13 FP8 + SiLU-Mul pipeline")
    pk.finalize()
    return passed


# ================================================================
# Test 4: Full pipeline W13 FP8 + SiLU-Mul + W2 BF16 (single compile)
# ================================================================

def test_moe_w13_silu_mul_w2():
    """Test full MoE pipeline in a single compile: W13 FP8 -> SiLU-Mul -> W2 BF16."""
    device = "cuda"
    torch.manual_seed(300)

    print(f"\n{'='*70}")
    print(f"Test: Full MoE pipeline (W13 FP8 + SiLU-Mul + W2 BF16, single compile)")
    print(f"  E={NUM_EXPERTS}, B={BATCH_SIZE}, K={HIDDEN_SIZE}, I={INTERMEDIATE_SIZE}, topk={NUM_TOPK}")

    input_val = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device) * 0.1
    w13_val = torch.randn(NUM_EXPERTS, N_W13, HIDDEN_SIZE, device=device) / math.sqrt(HIDDEN_SIZE)
    w2_val = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device,
                          dtype=torch.bfloat16) * 0.01

    input_fp8, input_scale = quantize_fp8_2d(input_val)
    w13_fp8, w13_scale = quantize_fp8_3d(w13_val)
    routing, mask = make_routing(BATCH_SIZE, NUM_EXPERTS, NUM_TOPK, device)

    w13_out = torch.zeros(BATCH_SIZE, NUM_TOPK, N_W13, dtype=torch.bfloat16, device=device)
    silu_out = torch.zeros(BATCH_SIZE, NUM_TOPK, INTERMEDIATE_SIZE, dtype=torch.bfloat16, device=device)
    w2_out = torch.zeros(BATCH_SIZE, NUM_TOPK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

    # Reference
    input_deq = dequantize_fp8_2d(input_fp8, float32_to_ue8m0_approx(input_scale))
    w13_deq = dequantize_fp8_3d(w13_fp8, float32_to_ue8m0_approx(w13_scale))

    ref_w13 = torch.zeros(BATCH_SIZE, NUM_TOPK, N_W13, dtype=torch.float32, device=device)
    for i in range(BATCH_SIZE):
        for s in range(NUM_TOPK):
            eid = (i * NUM_TOPK + s) % NUM_EXPERTS
            ref_w13[i, s] = input_deq[i] @ w13_deq[eid].T
    ref_gate = ref_w13[:, :, :INTERMEDIATE_SIZE]
    ref_up = ref_w13[:, :, INTERMEDIATE_SIZE:]
    ref_silu = (torch.nn.functional.silu(ref_gate) * ref_up).bfloat16()

    ref_w2 = torch.zeros(BATCH_SIZE, NUM_TOPK, HIDDEN_SIZE, dtype=torch.float32, device=device)
    for i in range(BATCH_SIZE):
        for s in range(NUM_TOPK):
            eid = (i * NUM_TOPK + s) % NUM_EXPERTS
            ref_w2[i, s] = ref_silu[i, s].float() @ w2_val[eid].float().T
    ref_w2 = ref_w2.bfloat16()

    # Build PK — all three layers in a single compile
    qo_indptr = torch.zeros(BATCH_SIZE + 1, dtype=torch.int32, device=device)
    qo_indptr[BATCH_SIZE] = BATCH_SIZE
    pk = make_pk(meta_tensors={"qo_indptr_buffer": qo_indptr})

    t_ifp8 = pk.attach_input(input_fp8, name="input_fp8")
    t_isc = pk.attach_input(input_scale, name="input_scale")
    t_w13fp8 = pk.attach_input(w13_fp8, name="w13_fp8")
    t_w13sc = pk.attach_input(w13_scale, name="w13_scale")
    t_w2 = pk.attach_input(w2_val, name="w2")
    t_rt = pk.attach_input(routing, name="routing_indices")
    t_mk = pk.attach_input(mask, name="mask")
    t_w13out = pk.attach_input(w13_out, name="w13_out")
    t_siluout = pk.attach_input(silu_out, name="silu_out")
    t_w2out = pk.attach_input(w2_out, name="w2_out")

    pk.moe_w13_fp8_layer(
        input_fp8=t_ifp8, input_scale=t_isc,
        weight_fp8=t_w13fp8, weight_scale=t_w13sc,
        moe_routing_indices=t_rt, moe_mask=t_mk, output=t_w13out,
        grid_dim=(8, 16, 1), block_dim=(256, 1, 1),
    )
    pk.moe_silu_mul_layer(
        input=t_w13out, output=t_siluout,
        grid_dim=(BATCH_SIZE, NUM_TOPK, 1), block_dim=(256, 1, 1),
    )
    pk.moe_w2_linear_layer(
        input=t_siluout, weight=t_w2,
        moe_routing_indices=t_rt, moe_mask=t_mk, output=t_w2out,
        grid_dim=(8, 16, 1), block_dim=(256, 1, 1),
    )

    print("Compiling...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"\nW2 output[0, 0, :8]:  {w2_out[0, 0, :8]}")
    print(f"Reference[0, 0, :8]:  {ref_w2[0, 0, :8]}")
    max_abs = (w2_out.float() - ref_w2.float()).abs().max().item()
    max_rel = max_abs / max(ref_w2.float().abs().max().item(), 1e-6)
    print(f"\nMax abs diff: {max_abs:.6f}, Max rel err: {max_rel:.6f}")

    passed = max_abs < 1.0
    print(f"\n{'PASSED' if passed else 'FAILED'}: Full MoE pipeline (W13 FP8 + SiLU + W2 BF16)")
    pk.finalize()
    return passed


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    all_passed = True
    all_passed &= test_moe_w13_fp8()
    all_passed &= test_moe_w2_fp8()
    all_passed &= test_moe_w13_silu_mul()
    all_passed &= test_moe_w13_silu_mul_w2()

    print(f"\n{'='*70}")
    if all_passed:
        print("All FP8 MoE pipeline tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
