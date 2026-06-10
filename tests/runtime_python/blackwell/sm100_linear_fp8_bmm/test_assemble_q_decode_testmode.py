"""End-to-end test_mode coverage for the assemble_q_decode_sm100 kernel.

assemble_q_decode_sm100 interleaves the BMM-absorbed q_nope (N, H, 512) with
q_pe (N, H, 64) into the per-head [nope|pe] layout (N, H, 576) that the MLA
decode TMA expects.

Run:
  CUDA_VISIBLE_DEVICES=<idle-gpu> \
    python tests/runtime_python/blackwell/sm100_linear_fp8_bmm/test_assemble_q_decode_testmode.py
"""
import os
import sys
import torch

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402
from mirage.mpk.models.deepseek_v3 import tasks as dsv3_tasks

FOLDER = os.environ.get("MPK_TEST_OUTPUT_DIR", "/tmp/mpk_test_assemble_q")
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


def _run_case(label: str, N: int, H: int, D_NOPE: int = 512, D_PE: int = 64):
    """Allocate (N, H, D_NOPE) + (N, H, D_PE) random inputs and verify the
    kernel writes (N, H, D_NOPE+D_PE) = per-head [nope|pe] layout."""
    print(f"\n{'='*72}")
    print(f"Test: {label}")
    print(f"  N={N}  H={H}  D_NOPE={D_NOPE}  D_PE={D_PE}  D_TOTAL={D_NOPE+D_PE}")
    device = "cuda"
    torch.manual_seed(42)
    q_nope_abs = torch.randn(N, H, D_NOPE, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(N, H, D_PE, dtype=torch.bfloat16, device=device)
    q_nope_pe = torch.zeros(N, H, D_NOPE + D_PE, dtype=torch.bfloat16,
                            device=device)

    # Python reference: per-head concat
    ref = torch.empty_like(q_nope_pe)
    ref[:, :, :D_NOPE] = q_nope_abs
    ref[:, :, D_NOPE:] = q_pe

    pk = _make_pk(N)
    qn = pk.attach_input(q_nope_abs, name="q_nope_abs")
    qp = pk.attach_input(q_pe, name="q_pe")
    qo = pk.attach_input(q_nope_pe, name="q_nope_pe")
    dsv3_tasks.assemble_q_decode_sm100_layer(
        pk,
        q_nope_abs=qn, q_pe=qp, q_nope_pe=qo,
        grid_dim=(N, 1, 1), block_dim=(128, 1, 1))

    print("Compiling...")
    pk.compile(output_dir=FOLDER)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    diff = (q_nope_pe.float() - ref.float()).abs()
    max_abs = diff.max().item()
    print(f"  max-abs:    {max_abs:.6f}")
    print(f"  out[0,0,:4]: {q_nope_pe[0,0,:4].tolist()}")
    print(f"  ref[0,0,:4]: {ref[0,0,:4].tolist()}")
    print(f"  out[0,0,510:514]: {q_nope_pe[0,0,510:514].tolist()}")
    print(f"  ref[0,0,510:514]: {ref[0,0,510:514].tolist()}")
    pk.finalize()
    ok = max_abs < 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    return ok


def test_smoke():
    return _run_case("smoke N=1 H=4", N=1, H=4)


def test_dsv3_decode():
    return _run_case("dsv3 decode N=1 H=128", N=1, H=128)


def test_dsv3_tp4():
    return _run_case("dsv3 TP4 N=1 H=32", N=1, H=32)


def test_dsv3_mtp():
    return _run_case("dsv3 MTP N=4 H=128", N=4, H=128)


def test_dsv3_decode_b16():
    return _run_case("dsv3 N=16 H=128", N=16, H=128)


if __name__ == "__main__":
    results = {}
    for fn in (test_smoke, test_dsv3_tp4, test_dsv3_decode,
               test_dsv3_mtp, test_dsv3_decode_b16):
        try:
            results[fn.__name__] = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[fn.__name__] = False
    print("\n" + "=" * 72)
    print("Summary:")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")
    fail = sum(1 for v in results.values() if not v)
    sys.exit(0 if fail == 0 else 1)
