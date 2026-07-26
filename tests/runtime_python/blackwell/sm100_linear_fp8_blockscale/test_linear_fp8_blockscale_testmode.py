"""Test mode: fp32-scale quantize -> preserved-block-scale dense FP8 GEMM.

Exercises the full pipeline for the new task -- Python layer API
(quantize_fp8_layer(scale_ue8m0=False) + linear_fp8_blockscale_layer), task
registration, C++ code generation, nvcc compilation and runtime dispatch --
with both GEMM variants (plain and residual-fused) in a single compile.

Run:
    python tests/runtime_python/blackwell/sm100_linear_fp8_blockscale/\
test_linear_fp8_blockscale_testmode.py
"""

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

BLOCK = 128
FP8_MAX = 448.0
EPS = 1e-10
BATCH_SIZE = 16
HIDDEN_SIZE = 2048   # K
OUTPUT_SIZE = 2048   # N (the grid splits it into N/128 tasks)
TOPK = 8             # 3-D quantize shape (the MoE activation), M3-I2b
INTER_SIZE = 512

torch.backends.cuda.matmul.allow_tf32 = False


def quantize_activation(x_bf16):
    """Inline PyTorch reference for MPK's fp32-scale per-token-group quantizer
    (docs/qwen35/vllm-graph.md 3.4): absmax = max(max|x|, 1e-10),
    scale = absmax / 448, x / scale, clamp before the RN-even e4m3 cast."""
    m, k = x_bf16.shape
    xf = x_bf16.float().reshape(m, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    scale = absmax / FP8_MAX
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(m, k).to(torch.float8_e4m3fn), scale.contiguous()


def quantize_weight_blocks(w_bf16):
    """Inline reference for the checkpoint's weight format: one float32 scale
    per 128x128 block, stored [N/128, K/128]."""
    n, k = w_bf16.shape
    wf = w_bf16.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    absmax = wf.abs().amax(dim=(1, 3)).clamp(min=EPS)
    scale = absmax / FP8_MAX
    q = (wf / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(n, k).to(torch.float8_e4m3fn), scale.contiguous()


def dequant_groups(q, scale):
    m, k = q.shape
    return (q.float().reshape(m, k // BLOCK, BLOCK) * scale.unsqueeze(-1)).reshape(m, k)


def dequant_blocks(q, scale):
    n, k = q.shape
    wf = q.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    return (wf * scale[:, None, :, None]).reshape(n, k)


def main():
    device = "cuda"
    torch.manual_seed(20260726)

    x = torch.randn(BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    w_plain = torch.randn(
        OUTPUT_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    w_resid = torch.randn(
        OUTPUT_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn(
        BATCH_SIZE, OUTPUT_SIZE, dtype=torch.bfloat16, device=device
    )

    w_plain_q, w_plain_s = quantize_weight_blocks(w_plain)
    w_resid_q, w_resid_s = quantize_weight_blocks(w_resid)

    # Runtime buffers the megakernel fills.
    x_q = torch.zeros(
        BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, device=device
    )
    x_scale = torch.zeros(
        BATCH_SIZE, HIDDEN_SIZE // BLOCK, dtype=torch.float32, device=device
    )
    out_plain = torch.zeros(
        BATCH_SIZE, OUTPUT_SIZE, dtype=torch.bfloat16, device=device
    )
    out_resid = torch.zeros(
        BATCH_SIZE, OUTPUT_SIZE, dtype=torch.bfloat16, device=device
    )

    # --- M3-I2b row-partition gate ---------------------------------------
    # Same input, same kernel, but grid.x splits the token axis instead of
    # handing every task the whole tensor. Must be byte-identical: a group's
    # fp8 bytes and fp32 scale come from that group's own 128 elements and the
    # kernel's row loop carries no cross-row state. 2-D and 3-D, because the
    # qwen3.5 builder quantizes both [mbt, hidden] and [mbt, topk, inter].
    x_q_rp = torch.zeros_like(x_q)
    x_scale_rp = torch.zeros_like(x_scale)
    x3 = torch.randn(
        BATCH_SIZE, TOPK, INTER_SIZE, dtype=torch.bfloat16, device=device
    )
    x3_q = torch.zeros(
        BATCH_SIZE, TOPK, INTER_SIZE, dtype=torch.float8_e4m3fn, device=device
    )
    x3_scale = torch.zeros(
        BATCH_SIZE, TOPK, INTER_SIZE // BLOCK, dtype=torch.float32, device=device
    )
    x3_q_rp = torch.zeros_like(x3_q)
    x3_scale_rp = torch.zeros_like(x3_scale)

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
    pk = PersistentKernel(**params)
    assert pk.target_cc >= 100, "linear_fp8_blockscale_sm100 requires Blackwell"
    block_dim = (256, 1, 1)

    x_dt = pk.attach_input(x, name="x")
    x_q_dt = pk.attach_input(x_q, name="x_q")
    x_scale_dt = pk.attach_input(x_scale, name="x_scale")
    w_plain_dt = pk.attach_input(w_plain_q, name="w_plain")
    w_plain_s_dt = pk.attach_input(w_plain_s, name="w_plain_scale")
    w_resid_dt = pk.attach_input(w_resid_q, name="w_resid")
    w_resid_s_dt = pk.attach_input(w_resid_s, name="w_resid_scale")
    residual_dt = pk.attach_input(residual, name="residual")
    out_plain_dt = pk.attach_input(out_plain, name="out_plain")
    out_resid_dt = pk.attach_input(out_resid, name="out_resid")

    # fp32-scale activation quantization (scale_ue8m0=False), the variant the
    # preserved-block-scale GEMM consumes.
    pk.quantize_fp8_layer(
        input=x_dt,
        output_fp8=x_q_dt,
        output_scale=x_scale_dt,
        grid_dim=(BATCH_SIZE, 1, 1),
        block_dim=(128, 1, 1),
        scale_ue8m0=False,
    )
    # M3-I2b: the same quantize with the token axis split across the grid.
    pk.quantize_fp8_layer(
        input=x_dt,
        output_fp8=pk.attach_input(x_q_rp, name="x_q_rp"),
        output_scale=pk.attach_input(x_scale_rp, name="x_scale_rp"),
        grid_dim=(BATCH_SIZE, 1, 1),
        block_dim=(128, 1, 1),
        scale_ue8m0=False,
        row_partition=(0, -1, -1),
    )
    x3_dt = pk.attach_input(x3, name="x3")
    pk.quantize_fp8_layer(
        input=x3_dt,
        output_fp8=pk.attach_input(x3_q, name="x3_q"),
        output_scale=pk.attach_input(x3_scale, name="x3_scale"),
        grid_dim=(BATCH_SIZE, 1, 1),
        block_dim=(128, 1, 1),
        scale_ue8m0=False,
    )
    pk.quantize_fp8_layer(
        input=x3_dt,
        output_fp8=pk.attach_input(x3_q_rp, name="x3_q_rp"),
        output_scale=pk.attach_input(x3_scale_rp, name="x3_scale_rp"),
        grid_dim=(BATCH_SIZE, 1, 1),
        block_dim=(128, 1, 1),
        scale_ue8m0=False,
        row_partition=(0, -1, -1),
    )
    pk.linear_fp8_blockscale_layer(
        input_fp8=x_q_dt,
        input_scale=x_scale_dt,
        weight_fp8=w_plain_dt,
        weight_scale=w_plain_s_dt,
        output=out_plain_dt,
        grid_dim=(OUTPUT_SIZE // BLOCK, 1, 1),
        block_dim=block_dim,
    )
    pk.linear_fp8_blockscale_layer(
        input_fp8=x_q_dt,
        input_scale=x_scale_dt,
        weight_fp8=w_resid_dt,
        weight_scale=w_resid_s_dt,
        output=out_resid_dt,
        grid_dim=(OUTPUT_SIZE // BLOCK, 1, 1),
        block_dim=block_dim,
        residual=residual_dt,
    )

    pk.compile(output_dir="./test_output_linear_fp8_blockscale")
    pk()
    torch.cuda.synchronize()

    # --- PyTorch reference ---
    ref_x_q, ref_x_scale = quantize_activation(x)
    assert torch.equal(x_q.view(torch.uint8), ref_x_q.view(torch.uint8)), (
        "the fp32-scale quantize task disagrees with the reference primitive"
    )
    torch.testing.assert_close(x_scale, ref_x_scale, rtol=1e-6, atol=0.0)

    # --- M3-I2b row-partition gate: BYTE equality, not tolerance ----------
    assert torch.equal(x_q_rp.view(torch.uint8), x_q.view(torch.uint8)), (
        "2-D row_partition=(0,-1,-1) changed the fp8 bytes"
    )
    assert torch.equal(x_scale_rp.view(torch.int32), x_scale.view(torch.int32)), (
        "2-D row_partition=(0,-1,-1) changed the fp32 block scales"
    )
    assert torch.equal(x3_q_rp.view(torch.uint8), x3_q.view(torch.uint8)), (
        "3-D row_partition=(0,-1,-1) changed the fp8 bytes"
    )
    assert torch.equal(x3_scale_rp.view(torch.int32), x3_scale.view(torch.int32)), (
        "3-D row_partition=(0,-1,-1) changed the fp32 block scales"
    )
    # ... and the 3-D unpartitioned path itself still matches the reference,
    # so "both agree" cannot be two copies of the same wrong answer.
    ref3_q, ref3_s = quantize_activation(x3.reshape(-1, INTER_SIZE))
    assert torch.equal(
        x3_q.reshape(-1, INTER_SIZE).view(torch.uint8), ref3_q.view(torch.uint8)
    ), "3-D quantize disagrees with the reference primitive"
    torch.testing.assert_close(
        x3_scale.reshape(-1, INTER_SIZE // BLOCK), ref3_s, rtol=1e-6, atol=0.0
    )
    print("  row_partition: 2-D and 3-D byte-identical to the whole-tensor grid")

    x_deq = dequant_groups(ref_x_q, ref_x_scale)
    ref_plain = x_deq @ dequant_blocks(w_plain_q, w_plain_s).t()
    ref_resid = x_deq @ dequant_blocks(w_resid_q, w_resid_s).t() + residual.float()

    for name, out, ref in (
        ("plain", out_plain, ref_plain),
        ("residual", out_resid, ref_resid),
    ):
        err = (out.float() - ref).norm().item() / ref.norm().item()
        floor = (
            (ref.to(torch.bfloat16).float() - ref).norm().item() / ref.norm().item()
        )
        print(
            f"  {name}: max_abs_diff={(out.float() - ref).abs().max().item():.4e} "
            f"frob_rel={err:.3e} bf16_output_floor={floor:.3e} "
            f"ratio={err / floor:.2f}"
        )
        assert err <= 1.6 * floor, (
            f"{name}: frob_rel {err:.3e} exceeds 1.6x the bf16 output-rounding "
            f"floor {floor:.3e}"
        )
        torch.testing.assert_close(
            out, ref.to(torch.bfloat16), rtol=1e-2, atol=1e-2
        )

    pk.finalize()
    print("LINEAR_FP8_BLOCKSCALE TEST-MODE PIPELINE PASSED")


if __name__ == "__main__":
    main()
