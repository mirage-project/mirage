"""End-to-end test_mode coverage for the assemble_q_decode_sm100 kernel.

assemble_q_decode_sm100 interleaves the BMM-absorbed q_nope (N, H, 512) with
q_pe (N, H, 64) into the per-head [nope|pe] layout (N, H, 576) that the MLA
decode TMA expects (builder.py `_bmm_decode_q_path` L1312, pe_only=True).

Two modes:
  * full (pe_only=False): writes nope into [0:512] AND q_pe into [512:576].
  * pe_only=True: writes ONLY q_pe into the tail [512:576]; the nope region is
    left untouched (in production the BMM already wrote nope into
    q_nope_pe[:, :, :512] via the slice-view fuse). The reference therefore
    preserves the pre-existing nope content and only checks the tail.

Reference is a trivial per-head concat (no FP math) -> byte-exact.

DSV3 sweep (decision log): H in {128,64,32,16} (tp=1,2,4,8) x N=bs in
{1,2,4,8,16}. Per-element copy -> full bs sweep at every H is cheap, so the
full 4x5 cross-product is run (20 configs) for the full mode + a pe_only
spot-check at each H (bs=16).

Run:
  CUDA_VISIBLE_DEVICES=<idle-gpu> \
    python tests/runtime_python/blackwell/sm100_linear_fp8_bmm/test_assemble_q_decode_testmode.py
"""
import os
import sys
import torch

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402

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


def _run_case(label: str, N: int, H: int, D_NOPE: int = 512, D_PE: int = 64,
              pe_only: bool = False) -> bool:
    """Allocate (N, H, D_NOPE) + (N, H, D_PE) random inputs and verify the
    kernel writes (N, H, D_NOPE+D_PE) = per-head [nope|pe] layout.

    pe_only=True: only the [D_NOPE:] tail is written; the nope region must
    retain its pre-existing (here: random sentinel) content."""
    print(f"\n{'='*72}")
    print(f"Test: {label}")
    print(f"  N={N}  H={H}  D_NOPE={D_NOPE}  D_PE={D_PE}  "
          f"D_TOTAL={D_NOPE+D_PE}  pe_only={pe_only}")
    device = "cuda"
    torch.manual_seed(42)
    q_nope_abs = torch.randn(N, H, D_NOPE, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(N, H, D_PE, dtype=torch.bfloat16, device=device)

    if pe_only:
        # Pre-fill the output with a distinct sentinel so we can confirm the
        # kernel (a) writes the q_pe tail and (b) leaves the nope region as-is.
        q_nope_pe = (torch.randn(N, H, D_NOPE + D_PE, dtype=torch.bfloat16,
                                 device=device) * 7.0).contiguous()
        ref = q_nope_pe.clone()
        ref[:, :, D_NOPE:] = q_pe  # only the tail is overwritten
    else:
        q_nope_pe = torch.zeros(N, H, D_NOPE + D_PE, dtype=torch.bfloat16,
                                device=device)
        ref = torch.empty_like(q_nope_pe)
        ref[:, :, :D_NOPE] = q_nope_abs
        ref[:, :, D_NOPE:] = q_pe

    pk = _make_pk(N)
    qn = pk.attach_input(q_nope_abs, name="q_nope_abs")
    qp = pk.attach_input(q_pe, name="q_pe")
    qo = pk.attach_input(q_nope_pe, name="q_nope_pe")
    pk.assemble_q_decode_sm100_layer(
        q_nope_abs=qn, q_pe=qp, q_nope_pe=qo,
        grid_dim=(N, 1, 1), block_dim=(128, 1, 1),
        pe_only=pe_only)

    print("Compiling...")
    pk.compile(output_dir=FOLDER)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    diff = (q_nope_pe.float() - ref.float()).abs()
    max_abs = diff.max().item()
    print(f"  max-abs:    {max_abs:.6f}")
    print(f"  out[0,0,510:514]: {q_nope_pe[0,0,510:514].tolist()}")
    print(f"  ref[0,0,510:514]: {ref[0,0,510:514].tolist()}")
    pk.finalize()
    ok = max_abs < 1e-6  # pure copy -> byte-exact
    print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    return ok


# ===========================================================================
# DSV3 sweep: H in {128,64,32,16} (tp=1,2,4,8) x N=bs in {1,2,4,8,16}.
# Per-element copy -> full cross-product is cheap; run all 20 for full mode,
# plus a pe_only spot-check at every H (bs=16).
# ===========================================================================
_HS = (128, 64, 32, 16)
_NS = (1, 2, 4, 8, 16)


def _matrix():
    cases = []
    tp_for = {128: 1, 64: 2, 32: 4, 16: 8}
    for H in _HS:
        tp = tp_for[H]
        for N in _NS:
            cases.append((f"full tp{tp} N{N} H{H}", N, H, False))
    # pe_only spot-check at each H, bs=16.
    for H in _HS:
        tp = tp_for[H]
        cases.append((f"pe_only tp{tp} N16 H{H}", 16, H, True))
    return cases


if __name__ == "__main__":
    results = {}
    for label, N, H, pe_only in _matrix():
        try:
            results[label] = _run_case(label, N=N, H=H, pe_only=pe_only)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[label] = False
    print("\n" + "=" * 72)
    print("Summary (assemble_q_decode_sm100):")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")
    fail = sum(1 for v in results.values() if not v)
    print(f"  {len(results) - fail}/{len(results)} PASS")
    sys.exit(0 if fail == 0 else 1)
