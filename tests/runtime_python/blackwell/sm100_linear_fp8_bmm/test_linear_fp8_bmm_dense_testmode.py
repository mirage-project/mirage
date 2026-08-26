"""End-to-end test_mode coverage for the DENSE-scale FP8 BMM kernel
(`linear_fp8_bmm_dense_sm100_layer`, float32 block scales).

This is the dense (float32-block-scale) variant of the per-head FP8 batched
matmul. DSV3 uses it for the decode BMM2 / o-unabsorption via the
dense=True builder dispatch (builder.py `_bmm_decode_o_path`):
    input (bs, Hl, 512) fp8 + f32 scale [bs, Hl, nk=4]
    weight (Hl, 128, 512) fp8 + f32 block scale [Hl, 1, nk=4]
    output (bs, Hl, 128) bf16
  -> output[n, h, :] = input[n, h, :] @ weight[h, :, :]^T  (per head)
  grid = (1, Hl, 1)  (grid.x must be 1: per-head D_out=128=BN); block=(256,1,1).
Hl = num_local_q_heads = 128 // world_size (128/64/32/16 for tp=1/2/4/8).

Scale layout (confirmed from src/kernel/task_register.cc:5831 +
fp8_gemm_dense_qout_sm100_common.cuh L301-302):
  input_scale  [N, Hl, nk] float32 row-major; per-head row stride = Hl*nk
  weight_scale [Hl, D_out/128=1, nk] float32 128x128-block

Run:
  CUDA_VISIBLE_DEVICES=<idle-gpu> \
    python tests/runtime_python/blackwell/sm100_linear_fp8_bmm/test_linear_fp8_bmm_dense_testmode.py
"""
import os
import sys
import torch

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from pytorch_reference import (  # noqa: E402
    quantize_bmm_input_f32,
    quantize_bmm_weight_f32,
    bmm_reference_dense_f32,
)

FOLDER = os.environ.get("MPK_TEST_OUTPUT_DIR", "/tmp/mpk_test_bmm_dense")
os.makedirs(FOLDER, exist_ok=True)


def _make_pk(batch_size: int) -> PersistentKernel:
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=batch_size,
        max_num_batched_requests=batch_size,
    )
    return PersistentKernel(**params)


def _run_case(label: str, batch: int, num_heads: int, d_in: int = 512,
              d_out: int = 128, tol: float = 0.05) -> bool:
    """Compile + run linear_fp8_bmm_dense_sm100_layer end-to-end and compare to
    the dense-f32-scale per-head reference."""
    assert batch <= 16, f"BMM kernel caps batch <= 16, got {batch}"
    assert d_in % 128 == 0 and d_out % 128 == 0
    grid_x = 1                       # required: per-head D_out=128=BN
    grid_y = num_heads
    nk = d_in // 128

    print(f"\n{'='*72}")
    print(f"Test: {label}")
    print(f"  N={batch}  H={num_heads}  D_in={d_in}  D_out={d_out}  nk={nk}  "
          f"grid=({grid_x}, {grid_y}, 1)")

    device = "cuda"
    torch.manual_seed(42)
    input_bf16 = (torch.randn(batch, num_heads, d_in,
                              dtype=torch.bfloat16, device=device) * 0.1
                  ).contiguous()
    weight_bf16 = (torch.randn(num_heads, d_out, d_in,
                               dtype=torch.bfloat16, device=device)
                   / (d_in ** 0.5)).contiguous()

    input_fp8, input_scale = quantize_bmm_input_f32(input_bf16)     # [N,H,nk]
    weight_fp8, weight_scale = quantize_bmm_weight_f32(weight_bf16)  # [H,1,nk]
    output = torch.zeros(batch, num_heads, d_out, dtype=torch.bfloat16,
                         device=device)

    ref = bmm_reference_dense_f32(input_fp8, input_scale,
                                  weight_fp8, weight_scale)

    pk = _make_pk(batch)
    i_fp8 = pk.attach_input(input_fp8, name="bmm_input_fp8")
    i_sc = pk.attach_input(input_scale, name="bmm_input_scale")
    w_fp8 = pk.attach_input(weight_fp8, name="bmm_weight_fp8")
    w_sc = pk.attach_input(weight_scale, name="bmm_weight_scale")
    o = pk.attach_input(output, name="bmm_output")

    pk.linear_fp8_bmm_dense_sm100_layer(
        input_fp8=i_fp8, input_scale=i_sc,
        weight_fp8=w_fp8, weight_scale=w_sc,
        output=o,
        grid_dim=(grid_x, grid_y, 1),
        block_dim=(256, 1, 1),
    )

    print("Compiling...")
    pk.compile(output_dir=FOLDER)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    assert torch.isfinite(output).all(), "BMM output has non-finite values"

    diff = (output.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        output.float().flatten(), ref.float().flatten(), dim=0).item()
    denom = ref.float().abs().mean().item() + 1e-12
    rel = mean_abs / denom
    print(f"  output[0,0,:8]: {output[0,0,:8].tolist()}")
    print(f"  ref   [0,0,:8]: {ref[0,0,:8].tolist()}")
    print(f"  max-abs:  {max_abs:.4f}  (tol {tol})")
    print(f"  mean-abs: {mean_abs:.6f}  rel-mean: {rel*100:.4f}%")
    print(f"  cos-sim:  {cos:.6f}")

    pk.finalize()
    ok = cos >= 0.99 and (max_abs <= tol or rel <= 0.05)
    print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    return ok


# ===========================================================================
# DSV3 union-of-axes matrix — dense BMM is used only for BMM2 (Din=512,
# Dout=128). Same union as the UE8M0 BMM: every tp in {1,2,4,8} x every bs.
# ===========================================================================
_UNION = (
    [(1, bs) for bs in (1, 2, 4, 8, 16)]
    + [(tp, 16) for tp in (2, 4, 8)]
    + [(8, 1)]
)


if __name__ == "__main__":
    results = {}
    for tp, bs in _UNION:
        hl = 128 // tp
        label = f"dense bmm2 tp{tp} bs{bs} (H={hl}, Din=512, Dout=128)"
        try:
            results[label] = _run_case(label, batch=bs, num_heads=hl)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[label] = False
    print("\n" + "=" * 72)
    print("Summary (linear_fp8_bmm_dense_sm100 f32-scale):")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")
    fail = sum(1 for v in results.values() if not v)
    print(f"  {len(results) - fail}/{len(results)} PASS")
    sys.exit(0 if fail == 0 else 1)
