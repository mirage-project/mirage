"""End-to-end test_mode coverage for the MPK FP8 BMM kernel
(`linear_fp8_bmm_sm100_layer`, UE8M0 swapAB scales).

Drives the full MPK compilation pipeline (Python layer -> register_task ->
codegen -> nvcc -> persistent runtime) at real DSV3 decode-path shapes.

DSV3 has two real uses of this per-head batched matmul (builder.py
`_bmm_decode_q_path` L1300 and `_bmm_decode_o_path` L1433):
  * Q-up / kv_b_k absorption: input (bs, Hl, 128) -> output (bs, Hl, 512),
    weight (Hl, 512, 128).  Din=128, Dout=512, grid=(512//128=4, Hl, 1).
  * BMM2 / o-unabsorb (kv_b_v):  input (bs, Hl, 512) -> output (bs, Hl, 128),
    weight (Hl, 128, 512).  Din=512, Dout=128, grid=(1, Hl, 1).
Hl = num_local_q_heads = 128 // world_size (128/64/32/16 for tp=1/2/4/8).

BMM contract (kernel + register at src/kernel/task_register.cc:5591):
  - input  [N, H, D_in] fp8 + UE8M0 packed scale [N, H, packed_K]
  - weight [H, D_out, D_in] fp8 + scale [H, D_out, packed_K]
  - output [N, H, D_out] = input @ weight^T per head
  - grid = (D_out / 128, H, 1) — one head per CTA; block = (256, 1, 1)
  - N <= 16; D_in % 128 == 0; D_out % 128 == 0

Run:
  CUDA_VISIBLE_DEVICES=<idle-gpu> \
    python tests/runtime_python/blackwell/sm100_linear_fp8_bmm/test_linear_fp8_bmm_testmode.py
"""
import os
import sys
import torch

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from pytorch_reference import (  # noqa: E402
    quantize_bmm_input,
    quantize_bmm_weight,
    dequant_bmm_input,
    dequant_bmm_weight,
    bmm_reference_from_dequant,
)

FOLDER = os.environ.get("MPK_TEST_OUTPUT_DIR", "/tmp/mpk_test_bmm")
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


def _run_case(label: str, batch: int, num_heads: int, d_in: int,
              d_out: int, tol: float = 0.05) -> bool:
    """Compile + run linear_fp8_bmm_sm100_layer end-to-end and compare to the
    FP8-dequant per-head reference."""
    assert batch <= 16, f"BMM kernel caps batch <= 16, got {batch}"
    assert d_in % 128 == 0, f"d_in={d_in} must be % 128"
    assert d_out % 128 == 0, f"d_out={d_out} must be % 128"
    grid_x = d_out // 128
    grid_y = num_heads

    print(f"\n{'='*72}")
    print(f"Test: {label}")
    print(f"  N={batch}  H={num_heads}  D_in={d_in}  D_out={d_out}  "
          f"grid=({grid_x}, {grid_y}, 1)")

    device = "cuda"
    torch.manual_seed(42)
    input_bf16 = (torch.randn(batch, num_heads, d_in,
                              dtype=torch.bfloat16, device=device) * 0.1
                  ).contiguous()
    weight_bf16 = (torch.randn(num_heads, d_out, d_in,
                               dtype=torch.bfloat16, device=device)
                   / (d_in ** 0.5)).contiguous()

    input_fp8, input_scale = quantize_bmm_input(input_bf16)
    weight_fp8, weight_scale = quantize_bmm_weight(weight_bf16)
    output = torch.zeros(batch, num_heads, d_out, dtype=torch.bfloat16,
                         device=device)

    # FP8-dequant reference: kernel computes on the quantized operands.
    input_dq = dequant_bmm_input(input_fp8, input_scale)
    weight_dq = dequant_bmm_weight(weight_fp8, weight_scale)
    ref = bmm_reference_from_dequant(input_dq, weight_dq)

    pk = _make_pk(batch)
    i_fp8 = pk.attach_input(input_fp8, name="bmm_input_fp8")
    i_sc = pk.attach_input(input_scale, name="bmm_input_scale")
    w_fp8 = pk.attach_input(weight_fp8, name="bmm_weight_fp8")
    w_sc = pk.attach_input(weight_scale, name="bmm_weight_scale")
    o = pk.attach_input(output, name="bmm_output")

    pk.linear_fp8_bmm_sm100_layer(
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
    # fp8 tolerance: cosine > 0.99 OR relative <= 5% (decision log).
    ok = cos >= 0.99 and (max_abs <= tol or rel <= 0.05)
    print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    return ok


# ===========================================================================
# DSV3 union-of-axes matrix (decision log "Test-matrix size policy"):
#   {tp=1} x {bs=1,2,4,8,16}  U  {bs=16} x {tp=2,4,8}  U  {tp=8, bs=1}
# = 9 configs, hitting every tp in {1,2,4,8} and every bs in {1,2,4,8,16}.
# Hl = 128 // tp. bs capped <= 16 (decode-only swapAB kernel).
# Run for BOTH real DSV3 uses: Q-up (Din=128, Dout=512) and BMM2 (Din=512,
# Dout=128).
# ===========================================================================
_UNION = (
    [(1, bs) for bs in (1, 2, 4, 8, 16)]
    + [(tp, 16) for tp in (2, 4, 8)]
    + [(8, 1)]
)
# (label_suffix, d_in, d_out)
_SHAPES = (
    ("qup", 128, 512),   # kv_b_k Q-up absorption
    ("bmm2", 512, 128),  # kv_b_v o-unabsorption
)


def _matrix():
    cases = []
    for tag, d_in, d_out in _SHAPES:
        for tp, bs in _UNION:
            hl = 128 // tp
            cases.append((f"{tag} tp{tp} bs{bs} (H={hl}, "
                          f"Din={d_in}, Dout={d_out})", bs, hl, d_in, d_out))
    return cases


if __name__ == "__main__":
    results = {}
    for label, bs, hl, d_in, d_out in _matrix():
        try:
            results[label] = _run_case(label, batch=bs, num_heads=hl,
                                       d_in=d_in, d_out=d_out)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[label] = False
    print("\n" + "=" * 72)
    print("Summary (linear_fp8_bmm_sm100 UE8M0):")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")
    fail = sum(1 for v in results.values() if not v)
    print(f"  {len(results) - fail}/{len(results)} PASS")
    sys.exit(0 if fail == 0 else 1)
