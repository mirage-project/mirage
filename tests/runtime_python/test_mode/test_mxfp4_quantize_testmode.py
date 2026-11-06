"""
Test: MXFP4 quantize (TASK_QUANTIZE_MXFP4_SM100) via PersistentKernel test_mode.

The MPK quantize task should produce byte-identical packed e2m1 + e8m0 scales to
the validated standalone quantizer on the same bf16 input. Covers both scale
layouts the task infers from output_scale's leading dim:
  * interleaved (mma_n=0)  -> for the 1d2d GEMM path
  * swapAB per-tile (mma_n>0) -> for the small-batch swapAB GEMM path

Run:
    python tests/runtime_python/test_mode/test_mxfp4_quantize_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

_MXFP4_DIR = os.path.join(
    os.path.dirname(__file__), "..", "blackwell", "sm100_linear_mxfp4"
)
sys.path.insert(0, _MXFP4_DIR)
import runtime_kernel_blackwell_linear_mxfp4 as mxfp4_ext  # noqa: E402


def make_pk(batch_size, **extra):
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
    params.update(extra)
    return PersistentKernel(**params)


def test_quantize_mxfp4(batch_size, hidden_size, mma_n):
    device = "cuda"
    torch.manual_seed(0)
    layout = "swapAB" if mma_n > 0 else "interleaved"
    print(f"\n{'='*70}\nTest: MXFP4 quantize {layout} "
          f"(B={batch_size}, H={hidden_size}, mma_n={mma_n})")

    # bf16-rounded values so the MPK task (bf16 in) and the oracle (float32 in)
    # quantize the exact same numbers.
    x_bf16_src = (torch.randn(batch_size, hidden_size, device=device) * 0.5).to(
        torch.bfloat16
    )
    x_f32 = x_bf16_src.float()  # oracle reads float32

    # Oracle: the validated standalone quantizer. Its outputs give the exact
    # shapes (incl. row padding) the task writes into.
    q_ref, sf_ref = mxfp4_ext.quantize_mxfp4_sm100(x_f32, mma_n)
    torch.cuda.synchronize()

    q_out = torch.zeros_like(q_ref)
    sf_out = torch.zeros_like(sf_ref)
    # Attach the input at its TRUE batch size; the quantize task infers batch
    # from input.dim[0] (and the scale layout/MMA_N from output_scale.dim[0]),
    # padding rows internally. Passing a pre-padded input would make it infer
    # the wrong batch (and hence the wrong MMA_N).
    x_bf16 = x_bf16_src

    pk = make_pk(batch_size)
    t_in = pk.attach_input(x_bf16, name="qin")
    t_q = pk.attach_input(q_out, name="qout")
    t_sf = pk.attach_input(sf_out, name="sfout")
    pk.quantize_mxfp4_layer(
        input=t_in, output_fp4=t_q, output_scale=t_sf,
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )

    print("Compiling...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # Compare the real (non-padded) packed rows, and only the scale bytes the
    # kernels actually write. Both scale tensors are allocated with torch::empty
    # by the standalone, so unwritten slots are garbage and must be excluded.
    #   interleaved [tiles,k_outer,32,4,4]: written = [..., :, :, :2]
    #       (rows packed across the 32x4 grid; inner-4 slots 0,1 = 2 active e8m0)
    #   swapAB      [n_tiles,k_outer,32,4,4] with mma_n<=32: per tile only the
    #       first mma_n rows (within_32) at row_group 0, k_inner 0,1 are written.
    q_match = torch.equal(q_out[:batch_size], q_ref[:batch_size])
    if mma_n == 0:
        sf_match = torch.equal(sf_out[..., :2], sf_ref[..., :2])
    else:
        assert mma_n <= 32
        a = sf_out[:, :, :mma_n, 0, :2]
        b = sf_ref[:, :, :mma_n, 0, :2]
        sf_match = torch.equal(a, b)
    print(f"packed match: {q_match}, active-scale match: {sf_match}")
    passed = q_match and sf_match
    print(f"{'PASSED' if passed else 'FAILED'}: MXFP4 quantize {layout}")
    pk.finalize()
    return passed


if __name__ == "__main__":
    ok = True
    ok &= test_quantize_mxfp4(batch_size=128, hidden_size=512, mma_n=0)   # interleaved
    ok &= test_quantize_mxfp4(batch_size=64, hidden_size=512, mma_n=8)    # swapAB
    print(f"\n{'='*70}")
    if not ok:
        sys.exit(1)
    print("MXFP4 quantize tests PASSED!")
